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
# import pychrono.irrlicht as chronoirr
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml
import time
import torch
import gpytorch
from argparse import Namespace
from datetime import datetime
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembling_frenet import GPEnsembleModelFrenet
from models.configs import *
from helpers.closest_point import *
from helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *

# --------------
step_size = 2e-3
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
# Program parameters
model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
# currently only "custom_track" works for frenet
map_name = 'custom_track'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
use_dyn_friction = False
# gp_mpc_type = 'frenet'  # cartesian, frenet
# render_every = 30  # render graphics every n sim steps
constant_speed = True
constant_speed_ref = 50
constant_friction = 0.7
number_of_laps = 3
SAVE_MODEL = True
t_end = 250
# --------------

env = ChronoEnv(step_size, throttle_value)

# Creating the single-track Motion planner and Controller

# Init Pure-Pursuit regulator
work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 1.0}

# Load map config file
with open('configs/config_%s.yaml' % 'SaoPaulo') as file:  # map_name -- SaoPaulo
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
if not map_name == 'custom_track':

    if use_dyn_friction:
        tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
        tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)

        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    # Rotate the map for 90 degrees in anti-clockwise direction 
    # to match the map with the vehicle's initial orientation
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    waypoints[:, 1:3] = np.dot(waypoints[:, 1:3], rotation_matrix)

else:
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

    track = Track(centerline_descriptor=centerline_descriptor, track_width=10.0, reference_speed=15.0)
    waypoints = track.get_reference_trajectory()
    print('waypoints\n',waypoints.shape)

# maps has its own speed reference but we want just make it constant
if constant_speed:
    waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * constant_speed_ref

# Convert waypoints to ChBezierCurve
curve_points = [chrono.ChVectorD(waypoint[1], waypoint[2], 0.6) for waypoint in waypoints]
curve = chrono.ChBezierCurve(curve_points, True) # True = closed curve
    
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

friction = [constant_friction for i in range(waypoints.shape[0])]

# # Define the patch coordinates
# patch_coords = [[waypoint[1], waypoint[2], 0.0] for waypoint in waypoints]

# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
Kp = 3.8
Ki = 0
Kd = 0

env.make(config=MPCConfigEXT(), friction=friction, waypoints=waypoints,
         reduced_waypoints=waypoints, curve=curve, speedPID_Gain=[Kp, Ki, Kd], 
         ini_pos=chrono.ChVectorD(waypoints[0,1], waypoints[0,2], 0.5))

# ---------------
# Simulation loop
# ---------------

env.my_hmmwv.GetVehicle().EnableRealtime(True)
num_laps = 3  # Number of laps

# # Define the starting point and a tolerance distance
# # starting_point = chrono.ChVectorD(-70, 0, 0.6)  # Replace with the actual starting point coordinates
# tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

# Reset the simulation time
env.my_hmmwv.GetSystem().SetChTime(0) 

reset_config(env, env.vehicle_params)


planner_gp_mpc_frenet = STMPCPlanner(model=GPEnsembleModelFrenet(config=MPCConfigGPFrenet(), track=track), waypoints=waypoints,
                                             config=MPCConfigGPFrenet(), track=track)
planner_gp_mpc_frenet.trajectry_interpolation = 1


planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=env.config), 
                                waypoints=waypoints,
                                config=env.config) #path_follow_mpc.py


reset_config(planner_gp_mpc_frenet, env.vehicle_params)
reset_config(planner_ekin_mpc, env.vehicle_params)

# driver = veh.ChDriver(env.my_hmmwv.GetVehicle()) #This command does NOT work. Never use ChDriver!
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
    if gp_model_trained<= 1:
        # just run ext_kinematic for the first 2 laps
    else:
        # use the learned GP models using planner_gp_mpc_frenet
    
    # run the sim for one control period using env.step()
    # save params to "log" every 5 control steps by counting logged_data

    # add point by planner_gp_mpc_frenet.add_point(vehicle_state, u)

    # Sample X, Y every 2 control time step to Y_sample, X_sample
    # and add new datapoint by planner_gp_mpc_frenet.model.add_new_datapoint(X_sample, Y_sample)
    # using gather_data_every=2 and counting gather_data

    # If in the first two laps, learn GP by train_gp_min_variance
    # Save the dataset to "log_dataset"
'''
#########################################################

while env.lap_counter < num_laps:
    # Render scene
    env.render()

    vehicle_state = env.my_hmmwv.state
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

    tracking_error = 0.0

    if gp_model_trained <= 1:
    # if gp_model_trained < 1:
        # just run ext_kinematic for the first 2 laps (gp_model_trained=0 or 1)
        print("Initial model")
        if model_in_first_lap == "ext_kinematic":
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_ekin_mpc.plan(
                vehicle_state)
            u[0] = u[0] / planner_ekin_mpc.config.MASS  # Force to acceleration
    else:
        print("gp_model_trained",gp_model_trained)
        u, mpc_ref_path_s, mpc_ref_path_ey, mpc_pred_s, mpc_pred_ey, mpc_os, mpc_oey = planner_gp_mpc_frenet.plan(
            vehicle_state_frenet)

        u[0] = u[0] / planner_gp_mpc_frenet.config.MASS  # Force to acceleration

        mpc_ref_path_x = np.zeros(mpc_ref_path_s.shape)
        mpc_ref_path_y = np.zeros(mpc_ref_path_s.shape)
        mpc_pred_x = np.zeros(mpc_ref_path_s.shape)
        mpc_pred_y = np.zeros(mpc_ref_path_s.shape)

        for i in range(mpc_ref_path_s.shape[0]):
            pose_cartesian = track.frenet_to_cartesian(np.array([mpc_pred_s[i], mpc_pred_ey[i], 0.0]))  # [s, ey, eyaw]
            mpc_pred_x[i] = pose_cartesian[0]
            mpc_pred_y[i] = pose_cartesian[1]
            pose_cartesian = track.frenet_to_cartesian(np.array([mpc_ref_path_s[i], mpc_ref_path_ey[i], 0.0]))  # [s, ey, eyaw]
            mpc_ref_path_x[i] = pose_cartesian[0]
            mpc_ref_path_y[i] = pose_cartesian[1]

    if gp_model_trained:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, lower, upper = planner_gp_mpc_frenet.model.scale_and_predict_model_step(vehicle_state_frenet, [u[0] * planner_gp_mpc_frenet.config.MASS, u[1]])

    speed = vehicle_state[2] + u[0]*planner_ekin_mpc.config.DTK
    steering = env.driver_inputs.m_steering + u[1]*planner_ekin_mpc.config.DTK/env.config.MAX_STEER # [-1,1]
    # print("speed input", speed, "steering input", steering)

    # Forcing positive speed
    speed = constant_speed_ref

    # Run the simulation for one control period
    for i in range(int(env.control_step)):
        env.step(speed, steering)
        # Render scene
        env.render()
    laptime = env.time

    # Increase the speed reference every lap
    if constant_speed:
        if env.lap_counter >= 0 and waypoints[:, 5][0] < 18.7:
            waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.0003
        else:
            waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.00015

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
    u[0] = u[0] * planner_gp_mpc_frenet.config.MASS  # Acceleration to force

    if planner_gp_mpc_frenet.it > 0:
        planner_gp_mpc_frenet.add_point(vehicle_state_frenet, u)

    # obtain this transition every control time period, not simulation time step
    vx_transition = env.my_hmmwv.state[2] + np.random.randn(1)[0] * 0.00001 - vehicle_state[2] 
    vy_transition = env.my_hmmwv.state[4] + np.random.randn(1)[0] * 0.00001 - vehicle_state[4]
    yaw_rate_transition = env.my_hmmwv.state[5] + np.random.randn(1)[0] * 0.00001 - vehicle_state[5]

    gather_data_every = 2
    gather_data += 1
    if gather_data >= gather_data_every:
        Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])
        X_sample = np.array([float(vehicle_state[2]), float(vehicle_state[4]),
                                float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1])]) #vx, vy, yaw_rate, steering_angle, Force, steering_speed

        planner_gp_mpc_frenet.model.add_new_datapoint(X_sample, Y_sample)
        gather_data = 0

    if env.lap_counter - 1 == gp_model_trained:
    # if (env.lap_counter - 1 == gp_model_trained) and (env.lap_counter == 1):
        print('lap counter', env.lap_counter)
        gp_model_trained += 1
        print("GP training...")
        num_of_new_samples = 250

        print(f"{len(planner_gp_mpc_frenet.model.x_measurements[0])}")
        planner_gp_mpc_frenet.model.train_gp_min_variance(num_of_new_samples)

        print("GP training done")
        print('Model used: GP')
        print('Reference speed: %f' % waypoints[:, 5][0])

        log_dataset['X0'] = planner_gp_mpc_frenet.model.x_samples[0]
        log_dataset['X1'] = planner_gp_mpc_frenet.model.x_samples[1]
        log_dataset['X2'] = planner_gp_mpc_frenet.model.x_samples[2]
        log_dataset['X3'] = planner_gp_mpc_frenet.model.x_samples[3]
        log_dataset['X4'] = planner_gp_mpc_frenet.model.x_samples[4]
        log_dataset['X5'] = planner_gp_mpc_frenet.model.x_samples[5]
        log_dataset['Y0'] = planner_gp_mpc_frenet.model.y_samples[0]
        log_dataset['Y1'] = planner_gp_mpc_frenet.model.y_samples[1]
        log_dataset['Y2'] = planner_gp_mpc_frenet.model.y_samples[2]

        with open('log01', 'w') as f:
            json.dump(log, f)
        with open('testing_dataset', 'w') as f:
            json.dump(log_dataset, f)

    if env.time > t_end:
        print("env.time",env.time)
        break

execution_time_end = time.time()
print('Sim elapsed time:', laptime, 'Execution elapsed time:', execution_time_end - execution_time_start)
with open('log01', 'w') as f:
    json.dump(log, f)
with open('log_dataset', 'w') as f:
    json.dump(log_dataset, f)

if SAVE_MODEL:
    # planner_gp_mpc_frenet.model.save_model() #'GPEnsembleModelFrenet' object has no attribute 'save_model'
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

    torch.save(planner_gp_mpc_frenet.model.gp_model.state_dict(), 'gp' + dt_string + '.pth')
    torch.save(planner_gp_mpc_frenet.model.gp_likelihood.state_dict(), 'gp_likelihood' + dt_string + '.pth')


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


plt.show()

