# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2014 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================
# Authors: Simone Benatti
# =============================================================================
#
# The vehicle reference frame has Z up, X towards the front of the vehicle, and
# Y pointing to the left.
#
# MPC solution (acceleration and steering speed) is applied to the vehicle.
# The steering speed from MPC is fed into driver's steering input directly.
# ChSpeedController is used to enforce the acceleration from MPC.
# 
# Uses the custom_track for the maps in /maps directory. 
# Not using f1tenth-racetrack
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml
import time
import torch
import gpytorch
from argparse import Namespace
from datetime import datetime
from EGP.regulators.pure_pursuit import *
from EGP.regulators.path_follow_mpc import *
from EGP.models.extended_kinematic import ExtendedKinematicModel
from EGP.models.GP_model_ensembling import GPEnsembleModel
from EGP.models.GP_model_ensembling_frenet import GPEnsembleModelFrenet
from EGP.models.configs import *
from EGP.helpers.closest_point import *
from EGP.helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *
from utilities import load_map, friction_func, centerline_to_frenet

# --------------
step_size = 2e-3
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
# Program parameters
model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
# currently only "custom_track" works for frenet
map_name = 'custom_track'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
map_ind = 8 ####You also need to select the corresponding ind if not custom_track!!!!
use_dyn_friction = False
gp_mpc_type = 'cartesian'  # cartesian, frenet
# render_every = 30  # render graphics every n sim steps
constant_speed = True
constant_speed_ref = 4.0
constant_friction = 1.0
num_laps_learn = 2
num_laps = num_laps_learn + 1
SAVE_MODEL = True
SAVE_DIR = './data/'
MAP_DIR = './f1tenth-racetrack/'
t_end = 300
# --------------

env = ChronoEnv(step_size=step_size, throttle_value=throttle_value)

# Creating the single-track Motion planner and Controller

# Init Pure-Pursuit regulator
work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance

# Load map config file
with open('EGP/configs/config_%s.yaml' % 'SaoPaulo') as file:  # map_name -- SaoPaulo
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
conf.wpt_path="./EGP"+conf.wpt_path
# print("conf.wpt_path",conf.wpt_path)

if not map_name == 'custom_track':
    map_info = np.genfromtxt('map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=7, reverse=False)
    print("waypoints\n",waypoints.shape)

    # Check if the waypoints are of the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
    if waypoints.shape[1] == 4:
        print('centerline')
        waypoints = centerline_to_frenet(waypoints)
        env.reduced_rate = 2
        w0=waypoints[0,3]-np.pi
    else: # raceline
        print('raceline')
        env.reduced_rate = 5
        w0=waypoints[0,3]

    # sample every env.reduced_rate 10 waypoints for patch in visualization
    reduced_waypoints = waypoints[::env.reduced_rate, :] 
    friction = [constant_friction for i in range(reduced_waypoints.shape[0])]

elif map_name == 'custom_track':
    centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 50, 2 * 25 * np.pi + 50, 2 * 25 * np.pi + 100],
                                      [0.0, 0.0, -50.0, -50.0, 0.0],
                                      [0.0, 50.0, 50.0, 0.0, 0.0],
                                      [1 / 25, 0.0, 1 / 25, 0.0, 1 / 25],
                                      [0.0, np.pi, np.pi, 0.0, 0.0]]).T

    centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 25, 25 * (3.0 * np.pi / 2.0) + 25, 25 * (3.0 * np.pi / 2.0) + 50,
                                        25 * (2.0 * np.pi + np.pi / 2.0) + 50, 25 * (2.0 * np.pi + np.pi / 2.0) + 125, 25 * (3.0 * np.pi) + 125,
                                        25 * (3.0 * np.pi) + 200],
                                        [0.0, 0.0, -25.0, -50.0, -50.0, -100.0, -100.0, -75.0, 0.0],
                                        [0.0, 50.0, 50.0, 75.0, 100.0, 100.0, 25.0, 0.0, 0.0],
                                        [1 / 25, 0.0, -1 / 25, 0.0, 1 / 25, 0.0, 1 / 25, 0.0, 1/25],
                                        [0.0, np.pi, np.pi, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0, 0.0, 0.0]]).T

    track = Track(centerline_descriptor=centerline_descriptor, track_width=10.0, reference_speed=constant_speed_ref)
    waypoints = track.get_reference_trajectory()
    print('waypoints\n',waypoints.shape)
    friction = [constant_friction for i in range(waypoints.shape[0])]
    reduced_waypoints = waypoints
    w0=waypoints[0,3]

# maps has its own speed reference but we want just constant speed reference
if constant_speed:
    waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * constant_speed_ref

# speed PID gains
# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
# pure pursuit gains
# Kp = 1
# Ki = 0.01
# Kd = 0
# steeringPID_Gain=[2,0,0]
# ext_kinematic gains
Kp = 5
Ki = 0.01
Kd = 0
# steeringPID_Gain=[0.5,0,0]

env.make(friction=friction, waypoints=waypoints,
         reduced_waypoints=reduced_waypoints, speedPID_Gain=[Kp, Ki, Kd],
         steeringPID_Gain=[0.5,0,0], x0=waypoints[0,1], y0=waypoints[0,2],
         w0=w0)

# ---------------
# Simulation loop
# ---------------

if model_in_first_lap == "pure_pursuit":
    # Init Pure-Pursuit regulator
    work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance
    # work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 15, 'vgain': 1.0} # tlad: look ahead distance
    if map_name != 'custom_track':
        conf.wpt_path = MAP_DIR + conf.wpt_path
    conf.wpt_xind = 1
    conf.wpt_yind = 2
    conf.wpt_vind = -2
    planner_pp = PurePursuitPlanner(conf, env.vehicle_params.WB)
    planner_pp.waypoints = waypoints.copy()

    ballT = env.vis.GetSceneManager().addSphereSceneNode(0.1)
    ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)

    plt.figure()
    plt.plot(planner_pp.waypoints[:,1], planner_pp.waypoints[:,2], label="waypoints")
    plt.show()

dlk = np.sqrt((waypoints[1, 1] - waypoints[0, 1]) ** 2 + (waypoints[1, 2] - waypoints[0, 2]) ** 2)
if gp_mpc_type == 'frenet':
    planner_gp_mpc_frenet_config = MPCConfigGPFrenet()
    planner_gp_mpc_frenet_config.dlk = dlk
    reset_config(planner_gp_mpc_frenet_config, env.vehicle_params)
    planner_gp_mpc_frenet = STMPCPlanner(model=GPEnsembleModelFrenet(config=planner_gp_mpc_frenet_config, track=track),
                                        waypoints=waypoints.copy(),
                                        config=planner_gp_mpc_frenet_config, track=track)
    planner_gp_mpc_frenet.trajectry_interpolation = 1

elif gp_mpc_type == 'cartesian':
    planner_gp_mpc_config = MPCConfigGP()
    planner_gp_mpc_config.dlk = dlk
    reset_config(planner_gp_mpc_config, env.vehicle_params)
    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel(config=planner_gp_mpc_config),
                                  waypoints=waypoints.copy(),config=planner_gp_mpc_config)

planner_ekin_mpc_config = MPCConfigEXT()
planner_ekin_mpc_config.dlk = dlk
reset_config(planner_ekin_mpc_config, env.vehicle_params)
planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=planner_ekin_mpc_config), 
                                waypoints=waypoints.copy(),
                                config=planner_ekin_mpc_config) #path_follow_mpc.py
# reset_config(planner_ekin_mpc, env.vehicle_params)
# reset_config(planner_ekin_mpc.config, env.vehicle_params)

speed    = 0
steering = 0 # [-1,1]

execution_time_start = time.time()

# init logger
# For plotting
log = {'time': [], 'x': [], 'y': [], 'lap_n': [], 'vx': [], 'v_ref': [], 'vx_mean': [], 'vx_var': [], 'vy_mean': [],
        'vy_var': [], 'theta_mean': [], 'theta_var': [], 'true_vx': [], 'true_vy': [], 'true_yaw_rate': [], 'tracking_error': []}

# dataset for GP training
log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'Y0': [], 'Y1': [], 'Y2': [],
                'X0[t-1]': [], 'X1[t-1]': [], 'X2[t-1]': [], 'X3[t-1]': [], 'X4[t-1]': [], 'X5[t-1]': [], 'Y0[t-1]': [], 'Y1[t-1]': [],
                'Y2[t-1]': []}

gp_model_trained = 0
gather_data = 0
logged_data = 0

mean, lower, upper = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

print('Model used: %s' % model_in_first_lap)

######################################################
'''
while not done: # every control time step
    if gp_model_trained <= 1:
        # just run ext_kinematic for the first 2 laps
    else:
        # use the learned GP models using planner_gp_mpc_frenet
    
    # run the sim for one control period using env.step()
    # save params to "log" every 5 control steps by counting logged_data

    # add point by planner_gp_mpc_frenet.add_point(vehicle_state, u) <- This is not happening

    # Sample X, Y every 2 control time step to Y_sample, X_sample
    # and add new datapoint by planner_gp_mpc_frenet.model.add_new_datapoint(X_sample, Y_sample)
    # using gather_data_every=2 and counting gather_data

    # If in the first two laps, learn GP by train_gp_min_variance
    # Save the dataset to "log_dataset"
'''
#########################################################

ballT = env.vis.GetSceneManager().addSphereSceneNode(0.1)

u_speed_command = []
u_steering_command = []

plt.ion()
figure, ax = plt.subplots(1, 1, figsize=(10, 10))
figure_c, ax_c = plt.subplots(1, 1, figsize=(10, 10))
ref, = ax.plot([], [], 'g', label="ref path")
mpc_pred, = ax.plot([], [], 'r', label="pred path")
mpc_out, = ax.plot([], [], 'b', label="out path")
mpc_fx_input, = ax_c.plot([], [], 'k', label="fx input")
mpc_deltav_input, = ax_c.plot([], [], 'm', label="deltav")
ax.legend()
plt.title("MPC prediction")

while env.lap_counter < num_laps:
    # Render scene
    env.render()

    vehicle_state = env.my_hmmwv.state
    # print("==========================================")
    # print("THETA BEFORE: ", vehicle_state[3])
    # CONVERT THETA FROM -pi to pi to 0 to 2pi
    vehicle_state[3] = (vehicle_state[3] + 2*np.pi) % (2*np.pi)
    # print("THETA AFTER: ", vehicle_state[3])
    # print("==========================================")

    if map_name == 'custom_track':
        pose_frenet = track.cartesian_to_frenet(np.array([vehicle_state[0], vehicle_state[1], vehicle_state[3]]))  # np.array([x,y,yaw])

        vehicle_state_frenet = np.array([pose_frenet[0],  # s
                                        pose_frenet[1],  # ey
                                        vehicle_state[2],  # vx
                                        pose_frenet[2],  # eyaw
                                        vehicle_state[4],  # vy
                                        vehicle_state[-2],  # yaw rate
                                        vehicle_state[-1],  # steering angle
                                        ])
        print(f"X: {vehicle_state[0]}  Y: {vehicle_state[1]}  S: {pose_frenet[0]}")

        if pose_frenet[0] < 0:
            print("Negative s!!")

    tracking_error = 0.0
    u = [0.0, 0.0]

    if gp_model_trained < num_laps_learn:
    # if gp_model_trained < 1:
    # just run ext_kinematic for the first 2 laps (gp_model_trained=0 or 1)
        print("Initial model")
        if model_in_first_lap == "ext_kinematic":
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, env.mpc_ox, env.mpc_oy = planner_ekin_mpc.plan(vehicle_state)
            u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration

            # Increase the speed reference every lap
            if constant_speed:
                if waypoints[:, 5][0] < 18.7:
                    waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.003
                else:
                    waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.00015
            planner_ekin_mpc.waypoints = waypoints.copy()

        elif model_in_first_lap == "pure_pursuit":
            # Regulator step pure pursuit
            pos = env.my_hmmwv.GetVehicle().GetPos()
            planner_pp.waypoints = waypoints.copy()
            speed, steer_angle = planner_pp.plan(pos.x, pos.y, vehicle_state[3],
                                                 work['tlad'], work['vgain'])
            # print("pure pursuit input speed", speed, "steering angle [rad]", steer_angle)

            pT = chrono.ChVectorD(planner_pp.lookahead_point[0], planner_pp.lookahead_point[1],  pos.z)
            ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))

            # error_steer = steer_angle - vehicle_state[-1]
            # u[1] = 10.0 * error_steer

            # error_drive = speed - vehicle_state[2]
            # u[0] = 12.0 * error_drive

            u[1] = steer_angle
            u[0] = speed

            if env.lap_counter == 0:
                u[0] += np.random.randn(1)[0] * 0.2
                u[1] += np.random.randn(1)[0] * 0.01
    else:
        print("gp_model_trained", gp_model_trained)
        if gp_mpc_type == 'cartesian':
            planner_gp_mpc.waypoints = waypoints.copy()
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(vehicle_state)
            u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration

            # if waypoints[:, 5][0] <= 5.5:
            #     u[0] += np.random.randn(1)[0] * 0.000005
            #     u[1] += np.random.randn(1)[0] * 0.0005

            ref.set_data(mpc_ref_path_x, mpc_ref_path_y)
            mpc_pred.set_data(mpc_pred_x, mpc_pred_y)
            mpc_out.set_data(mpc_ox, mpc_oy)
            ax.legend()
            ax.set_xlim(min(mpc_ref_path_x)-10, max(mpc_ref_path_x)+10)
            ax.set_ylim(min(mpc_ref_path_y)-10, max(mpc_ref_path_y)+10)
            figure.canvas.draw()
            figure.canvas.flush_events()

            mpc_fx_input.set_data(np.arange(0, planner_gp_mpc.config.TK) *  planner_gp_mpc.config.DTK, planner_gp_mpc.input_o[0])
            mpc_deltav_input.set_data(np.arange(0, planner_gp_mpc.config.TK) *  planner_gp_mpc.config.DTK, planner_gp_mpc.input_o[1])
            ax_c.legend()
            ax_c.set_xlim(0, planner_gp_mpc.config.TK * planner_gp_mpc.config.DTK)
            ax_c.set_ylim(-1, 1)
            figure_c.canvas.draw()
            figure_c.canvas.flush_events()

            plt.show()

            _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                                                                    np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
        elif gp_mpc_type == 'frenet':
            u, mpc_ref_path_s, mpc_ref_path_ey, mpc_pred_s, mpc_pred_ey, mpc_os, mpc_oey = planner_gp_mpc_frenet.plan(vehicle_state_frenet)
            u[0] = u[0] / planner_gp_mpc_frenet.config.MASS  # Force to acceleration

            mpc_ref_path_x = np.zeros(mpc_ref_path_s.shape)
            mpc_ref_path_y = np.zeros(mpc_ref_path_s.shape)
            mpc_pred_x = np.zeros(mpc_ref_path_s.shape)
            mpc_pred_y = np.zeros(mpc_ref_path_s.shape)
            mpc_out_x = np.zeros(mpc_ref_path_s.shape)
            mpc_out_y = np.zeros(mpc_ref_path_s.shape)

            for i in range(mpc_ref_path_s.shape[0]):
                pose_cartesian = track.frenet_to_cartesian(np.array([mpc_pred_s[i], mpc_pred_ey[i], 0.0]))  # [s, ey, eyaw]
                mpc_pred_x[i] = pose_cartesian[0]
                mpc_pred_y[i] = pose_cartesian[1]
                pose_cartesian = track.frenet_to_cartesian(np.array([mpc_ref_path_s[i], mpc_ref_path_ey[i], 0.0]))  # [s, ey, eyaw]
                mpc_ref_path_x[i] = pose_cartesian[0]
                mpc_ref_path_y[i] = pose_cartesian[1]
                pose_cartesian = track.frenet_to_cartesian(np.array([mpc_os[i], mpc_oey[i], 0.0]))
                mpc_out_x[i] = pose_cartesian[0]
                mpc_out_y[i] = pose_cartesian[1]

            env.mpc_ox = mpc_pred_x
            env.mpc_oy = mpc_pred_y

            ref.set_data(mpc_ref_path_x, mpc_ref_path_y)
            mpc_pred.set_data(mpc_pred_x, mpc_pred_y)
            mpc_out.set_data(env.mpc_ox, env.mpc_oy)
            ax.legend()
            ax.set_xlim(min(mpc_ref_path_x)-10, max(mpc_ref_path_x)+10)
            ax.set_ylim(min(mpc_ref_path_y)-10, max(mpc_ref_path_y)+10)
            figure.canvas.draw()
            figure.canvas.flush_events()

            mpc_fx_input.set_data(np.arange(0, planner_gp_mpc_frenet.config.TK) *  planner_gp_mpc_frenet.config.DTK, planner_gp_mpc_frenet.input_o[0])
            mpc_deltav_input.set_data(np.arange(0, planner_gp_mpc_frenet.config.TK) *  planner_gp_mpc_frenet.config.DTK, planner_gp_mpc_frenet.input_o[1])
            ax_c.legend()
            ax_c.set_xlim(0, planner_gp_mpc_frenet.config.TK * planner_gp_mpc_frenet.config.DTK)
            ax_c.set_ylim(-1, 1)
            figure_c.canvas.draw()
            figure_c.canvas.flush_events()

            plt.show()

        else:
            print("Choose gp_mpc_type between cartesian and frenet!")

    if gp_model_trained:
        if gp_mpc_type == 'cartesian':
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean, lower, upper = planner_gp_mpc.model.scale_and_predict_model_step(vehicle_state, [u[0] * planner_gp_mpc.config.MASS, u[1]])
        elif gp_mpc_type == 'frenet':
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean, lower, upper = planner_gp_mpc_frenet.model.scale_and_predict_model_step(vehicle_state_frenet, [u[0] * planner_gp_mpc_frenet.config.MASS, u[1]])

    if (gp_model_trained < num_laps_learn) and (model_in_first_lap == "pure_pursuit"):
        speed = u[0]
        steering = u[1]
        # print("pure pursuit input speed", speed, "steering input [rad]", steering)
    else:
        speed = vehicle_state[2] + u[0]*planner_ekin_mpc.config.DTK
        steering = env.driver_inputs.m_steering*env.vehicle_params.MAX_STEER + u[1]*planner_ekin_mpc.config.DTK # [-max steer,max steer]

        # Check whether the input and the visualized steering angle are the same
        # steering = 0.5*env.vehicle_params.MAX_STEER
        # print("env.vehicle_params.MAX_STEER in main_GP_MPC",env.vehicle_params.MAX_STEER )
        u_speed_command.append(speed)
        u_steering_command.append(steering)
        # print("speed input:", speed, "steering input [-1,1]:", steering/env.vehicle_params.MAX_STEER, "[rad]:", steering, "[deg]:", steering*180/np.pi)

    # Forcing positive speed
    # speed = constant_speed_ref

    # Run the simulation for one control period
    for i in range(int(env.control_step)):
        env.step(steering, speed)
        # Render scene
        env.render()
    laptime = env.time

    # # Increase the speed reference every lap
    # if constant_speed:
    #     if waypoints[:, 5][0] < 18.7:
    #         waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.0003
    #     else:
    #         waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.00015

    # Logging
    logged_data += 1
    if logged_data > 5:
        log['time'].append(laptime)
        log['lap_n'].append(env.lap_counter)
        log['x'].append(env.my_hmmwv.state[0])
        log['y'].append(env.my_hmmwv.state[1])
        log['vx'].append(env.my_hmmwv.state[2])
        log['v_ref'].append(waypoints[:, 5][0])
        log['vx_mean'].append(float(mean[0]))
        log['vx_var'].append(float(abs(mean[0] - lower[0])))
        log['vy_mean'].append(float(mean[1]))
        log['vy_var'].append(float(abs(mean[1] - lower[1])))
        log['theta_mean'].append(float(mean[2]))
        log['theta_var'].append(float(abs(mean[2] - lower[2])))
        log['true_vx'].append(env.my_hmmwv.state[2])
        log['true_vy'].append(env.my_hmmwv.state[4])
        log['true_yaw_rate'].append(env.my_hmmwv.state[5])
        log['tracking_error'].append(tracking_error)
        logged_data = 0

    # learning GPs
    u[0] = u[0] * planner_ekin_mpc.config.MASS  # Acceleration to force

    # if (gp_mpc_type == 'frenet') and (planner_gp_mpc_frenet.it > 0):
    #     planner_gp_mpc_frenet.add_point(vehicle_state_frenet, u)
    # elif (gp_mpc_type == 'cartesian') and (planner_gp_mpc.it > 0):
    #     print("adding point")
    #     planner_gp_mpc.add_point(vehicle_state, u)

    # obtain this transition every control time period, not simulation time step
    vx_transition = env.my_hmmwv.state[2]- vehicle_state[2]# + np.random.randn(1)[0] * 0.00001  
    vy_transition = env.my_hmmwv.state[4]- vehicle_state[4]# + np.random.randn(1)[0] * 0.00001 
    yaw_rate_transition = env.my_hmmwv.state[5]- vehicle_state[5] # + np.random.randn(1)[0] * 0.00001 

    gather_data_every = 2
    gather_data += 1
    if gather_data >= gather_data_every:
        if gp_mpc_type == 'cartesian':
            Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])
            X_sample = np.array([float(vehicle_state[2]), float(vehicle_state[4]),
                                float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1])]) #vx, vy, yaw_rate, steering_angle, Force, steering_speed
            # print("adding new data point")
            planner_gp_mpc.model.add_new_datapoint(X_sample, Y_sample)
        elif gp_mpc_type == 'frenet':
            Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])
            X_sample = np.array([float(vehicle_state_frenet[2]), float(vehicle_state_frenet[4]),
                                float(vehicle_state_frenet[5]), float(vehicle_state_frenet[6]), float(u[0]), float(u[1])]) #vx, vy, yaw_rate, steering_angle, Force, steering_speed
            planner_gp_mpc_frenet.model.add_new_datapoint(X_sample, Y_sample)
        gather_data = 0

    if env.lap_counter - 1 == gp_model_trained:
    # if (env.lap_counter - 1 == gp_model_trained) and (env.lap_counter == 1):
        print('lap counter', env.lap_counter)
        gp_model_trained += 1
        print("GP training...")
        num_of_new_samples = 250

        if gp_mpc_type == 'cartesian':
            print(f"{len(planner_gp_mpc.model.x_measurements[0])}")
            planner_gp_mpc.model.train_gp_new(num_of_new_samples)
            log_dataset['X0'] = planner_gp_mpc.model.x_samples[0]
            log_dataset['X1'] = planner_gp_mpc.model.x_samples[1]
            log_dataset['X2'] = planner_gp_mpc.model.x_samples[2]
            log_dataset['X3'] = planner_gp_mpc.model.x_samples[3]
            log_dataset['X4'] = planner_gp_mpc.model.x_samples[4]
            log_dataset['X5'] = planner_gp_mpc.model.x_samples[5]
            log_dataset['Y0'] = planner_gp_mpc.model.y_samples[0]
            log_dataset['Y1'] = planner_gp_mpc.model.y_samples[1]
            log_dataset['Y2'] = planner_gp_mpc.model.y_samples[2]
        elif gp_mpc_type == 'frenet':
            print(f"{len(planner_gp_mpc_frenet.model.x_measurements[0])}")
            planner_gp_mpc_frenet.model.train_gp_new(num_of_new_samples)
            log_dataset['X0'] = planner_gp_mpc_frenet.model.x_samples[0]
            log_dataset['X1'] = planner_gp_mpc_frenet.model.x_samples[1]
            log_dataset['X2'] = planner_gp_mpc_frenet.model.x_samples[2]
            log_dataset['X3'] = planner_gp_mpc_frenet.model.x_samples[3]
            log_dataset['X4'] = planner_gp_mpc_frenet.model.x_samples[4]
            log_dataset['X5'] = planner_gp_mpc_frenet.model.x_samples[5]
            log_dataset['Y0'] = planner_gp_mpc_frenet.model.y_samples[0]
            log_dataset['Y1'] = planner_gp_mpc_frenet.model.y_samples[1]
            log_dataset['Y2'] = planner_gp_mpc_frenet.model.y_samples[2]
        print("GP training done")
        print('Model used: GP')
        print('Reference speed: %f' % waypoints[:, 5][0])

        with open(SAVE_DIR+'log01', 'w') as f:
            json.dump(log, f)
        with open(SAVE_DIR+'testing_dataset', 'w') as f:
            json.dump(log_dataset, f)

    # if env.time > t_end:
    #     print("env.time",env.time)
    #     break

execution_time_end = time.time()
print('Sim elapsed time:', laptime, 'Execution elapsed time:', execution_time_end - execution_time_start)
with open(SAVE_DIR+'log01', 'w') as f:
    json.dump(log, f)
with open(SAVE_DIR+'log_dataset', 'w') as f:
    json.dump(log_dataset, f)

if SAVE_MODEL:
    if gp_mpc_type == 'cartesian':
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        torch.save(planner_gp_mpc.model.gp_model.state_dict(), 'gp' + dt_string + '.pth')
        torch.save(planner_gp_mpc.model.gp_likelihood.state_dict(), 'gp_likelihood' + dt_string + '.pth')
    elif gp_mpc_type == 'frenet':
        # planner_gp_mpc_frenet.model.save_model() #'GPEnsembleModelFrenet' object has no attribute 'save_model'
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        torch.save(planner_gp_mpc_frenet.model.gp_model.state_dict(), SAVE_DIR + 'gp' + dt_string + '.pth')
        torch.save(planner_gp_mpc_frenet.model.gp_likelihood.state_dict(), SAVE_DIR + 'gp_likelihood' + dt_string + '.pth')

plt.figure()
plt.plot(env.t_stepsize, env.speed)
plt.title("longitudinal speed")
plt.xlabel("time [s]")
plt.ylabel("longitudinal speed [m/s]")
plt.savefig("longitudinal_speed.png")

plt.figure()
color = [i for i in range(len(env.x_trajectory))]
plt.scatter(env.x_trajectory, env.y_trajectory, c=color,s=1, label="trajectory")
plt.title("trajectory")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.savefig("trajectory.png")

# plt.figure()
# plt.plot(u_speed_command, label="speed command")
# plt.legend()

# plt.figure()
# plt.plot(u_steering_command, label="steering command")
# plt.legend()

plt.show()

