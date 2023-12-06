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
# Read the map info from map_info.txt
#
# Identify the cornering stiffness
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from argparse import Namespace
import warnings
from EGP.regulators.pure_pursuit import *
from EGP.regulators.path_follow_mpc import *
from EGP.models.extended_kinematic import ExtendedKinematicModel
from EGP.models.configs import *
from EGP.helpers.closest_point import *
from EGP.helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *
from utilities import load_map, friction_func, centerline_to_frenet

# --------------
step_size = 2e-3 #simulation step size
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
SAVE_MODEL = True
MAP_DIR = './f1tenth-racetrack/'
SAVE_DIR = './data/'
t_end = 60
map_ind = 17 # 39 Zandvoort_raceline
map_scale = 10
map_reverse = False
patch_scale = 1.5 # map_ind 16: 1.5
ref_vx = 8.0
control_model = "pure_pursuit" # options: ext_kinematic, pure_pursuit
num_laps = 5  # Number of laps
const_friction = 1.0
# --------------

'''
ref_vx = 10
speed PID= [5,0,0]
steering PID = [0.5,0,0]
actual speed 7.3 m/s

If you use ref_vx=15, the vehicle goes left and right at turning
'''

# Load map config file
with open('EGP/maps/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

map_info = np.genfromtxt('map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=map_scale, reverse=map_reverse)
print("waypoints\n",waypoints.shape)

# Check if the waypoints are of the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
if waypoints.shape[1] == 4:
    print("centerline to frenet")
    waypoints = centerline_to_frenet(waypoints)
    # reduced_rate = 1
    # map_ind = 16
    reduced_rate = 2
    w0=waypoints[0,3]-np.pi
    if map_reverse:
        w0 = waypoints[0,3]
else: # raceline
    print("reduced_rate is 5")
    reduced_rate = 5
    w0=waypoints[0,3]
    if map_reverse:
        w0 = waypoints[0,3]-np.pi

waypoints[:, -2] = ref_vx

# sample every env.reduced_rate 10 waypoints for patch in visualization
reduced_waypoints = waypoints[::reduced_rate, :] 
s_max = np.max(reduced_waypoints[:, 0])
# friction = [friction_func(i,s_max) for i in range(reduced_waypoints.shape[0])]
# friction = [1.0 + i/waypoints.shape[0] for i in range(reduced_waypoints.shape[0])]
friction = [const_friction for i in range(reduced_waypoints.shape[0])]

if control_model == "ext_kinematic":
    Kp = 5
    Ki = 0.01
    Kd = 0
elif control_model == "pure_pursuit":
    Kp = 1
    Ki = 0.01
    Kd = 0

env = ChronoEnv().make(timestep=step_size, control_period=0.1, waypoints=waypoints, patch_scale=patch_scale,
         sample_rate_waypoints=reduced_rate, friction=friction, speedPID_Gain=[Kp, Ki, Kd],
         steeringPID_Gain=[0.5,0,0], x0=reduced_waypoints[0,1], 
         y0=reduced_waypoints[0,2], w0=w0)

# ---------------
# Simulation loop
# ---------------


# Making sure that some config parameters are obtained from chrono, not from MPCConfigEXT
if control_model == "ext_kinematic":
    planner_ekin_mpc_config = MPCConfigEXT()
    planner_ekin_mpc_config.dlk = np.sqrt((waypoints[1, 1] - waypoints[0, 1]) ** 2 + (waypoints[1, 2] - waypoints[0, 2]) ** 2)
    reset_config(planner_ekin_mpc_config, env.vehicle_params)
    planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=planner_ekin_mpc_config), 
                                    waypoints=waypoints.copy(),
                                    config=planner_ekin_mpc_config) #path_follow_mpc.py
    if planner_ekin_mpc.config.DTK != env.control_period:
        warnings.warn("planner_ekin_mpc.config.DTK != env.control_period. Setting DTK to be equal to env.control_period.")
        planner_ekin_mpc.config.DTK = env.control_period
        print("planner_ekin_mpc.config.DTK", planner_ekin_mpc.config.DTK)
elif control_model == "pure_pursuit":
    # Init Pure-Pursuit regulator
    work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance
    # work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 15, 'vgain': 1.0} # tlad: look ahead distance
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

speed    = 0
steering = 0
control_list = []
state_list = []

execution_time_start = time.time()

observation = {'poses_x': env.my_hmmwv.state[0],'poses_y': env.my_hmmwv.state[1],
                'vx':env.my_hmmwv.state[2], 'poses_theta': env.my_hmmwv.state[3],
                'vy':env.my_hmmwv.state[4],'yaw_rate':env.my_hmmwv.state[5],'steering':env.my_hmmwv.state[6]}
        
        
Fy_over_muFz_list = []
slip_angle_FL_list = []

veh.TMeasyTire.ExportParameterFile(veh.TMeasyTire("Test"), "tmp")

plt.ion()
figure, ax = plt.subplots(1, 1, figsize=(10, 10))
scatter  = ax.scatter([], [], c='k', s=1, label = "Front Left")
ax.legend()
plt.title("Fy/muFz vs slip angle")

while env.lap_counter < num_laps:
    # Render scene
    env.render()

    if (env.step_number % (env.control_step) == 0):
        if control_model == "ext_kinematic":
            # Solve MPC problem 
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, env.mpc_ox, env.mpc_oy = planner_ekin_mpc.plan(env.my_hmmwv.state)
            u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration
            speed = env.my_hmmwv.state[2] + u[0]*env.control_period
            steering = env.driver_inputs.m_steering*env.vehicle_params.MAX_STEER + u[1]*env.control_period # [-max steer,max steer]

            if waypoints[:, 5][0] < 18.7:
                waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.003
            else:
                waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.00015
            planner_ekin_mpc.waypoints = waypoints.copy()

        elif control_model == "pure_pursuit":
            planner_pp.waypoints = waypoints.copy()
            speed, steering = planner_pp.plan(observation['poses_x'], observation['poses_y'], observation['poses_theta'],
                                            work['tlad'], work['vgain'])
            print("pure pursuit input speed", speed, "steering angle ratio [-1,1]", steering/env.vehicle_params.MAX_STEER)  
            u = np.array([speed, steering])
            # Visualize the lookahead point of pure-pursuit
            if planner_pp.lookahead_point is not None:
                pT = chrono.ChVectorD(planner_pp.lookahead_point[0], planner_pp.lookahead_point[1], 0.0)
                ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))
            else:
                print("No lookahead point found in main_map_info.py!!!")

            # # Get tire force in global frame
            # vehicle = env.my_hmmwv
            # tf_FL = vehicle.GetVehicle().GetTire(0, veh.LEFT).ReportTireForce(env.terrain).force
            # # tf_FR = vehicle.GetVehicle().GetTire(0, veh.RIGHT).ReportTireForce(env.terrain).force
            # # tf_RL = vehicle.GetVehicle().GetTire(1, veh.LEFT).ReportTireForce(env.terrain).force
            # # tf_RR = vehicle.GetVehicle().GetTire(1, veh.RIGHT).ReportTireForce(env.terrain).force

            # # Get tire force in local frame
            # tf_FL = np.array([tf_FL.x, tf_FL.y, tf_FL.z]).reshape(3,1)
            # SpindleRot_FL = vehicle.GetVehicle().GetSpindleRot(0,0)
            # SpindleQuat_FL = np.array([SpindleRot_FL.e0, SpindleRot_FL.e1, SpindleRot_FL.e2, SpindleRot_FL.e3])
            # SpindleRot_FL = R.from_quat(SpindleQuat_FL).as_matrix()
            # tf_FL_local = np.dot(SpindleRot_FL, tf_FL)

        if env.time > 15:
            Fz, tf_FL_local = get_tire_force(env, 0, veh.LEFT)
            mu_Fz = const_friction * Fz
            Fy_over_muFz = tf_FL_local[1] / mu_Fz
            slip_angle_FL = env.my_hmmwv.GetVehicle().GetTire(0,veh.LEFT).GetSlipAngle()

            Fy_over_muFz_list.append(abs(Fy_over_muFz))
            slip_angle_FL_list.append(abs(slip_angle_FL*180/np.pi))

            print("tf_FL_local", tf_FL_local, "slip_angle_FL", slip_angle_FL)

            scatter.set_offsets(np.c_[slip_angle_FL_list, Fy_over_muFz_list])
            ax.legend()
            ax.set_xlim([0, 5])
            ax.set_ylim([-0.5, 1.1])
            figure.canvas.draw()
            figure.canvas.flush_events()

            plt.show()

            # Fz, tf_FR_local = get_tire_force(env, 0, veh.RIGHT)
            # slip_angle_FR = env.my_hmmwv.GetVehicle().GetTire(0,veh.RIGHT).GetSlipAngle()

            # Fz, tf_RL_local = get_tire_force(env, 1, veh.LEFT)
            # slip_angle_RL = env.my_hmmwv.GetVehicle().GetTire(1,veh.LEFT).GetSlipAngle()

            # Fz, tf_RR_local = get_tire_force(env, 1, veh.RIGHT)
            # slip_angle_RR = env.my_hmmwv.GetVehicle().GetTire(1,veh.RIGHT).GetSlipAngle()

            # print("tf_FL_local", tf_FL_local, "slip_angle_FL", slip_angle_FL)
            # print("tf_FR_local", tf_FR_local, "slip_angle_FR", slip_angle_FR) 
            # print("tf_RL_local", tf_RL_local, "slip_angle_RL", slip_angle_RL)
            # print("tf_RR_local", tf_RR_local, "slip_angle_RR", slip_angle_RR)



        # set the steering between -pi to pi
        if steering > np.pi:
            print("steering command BEFORE", steering)
            steering = steering - 2*np.pi
            print("steering command AFTER", steering)
        elif steering < -np.pi:
            print("steering command BEFORE", steering)
            steering = steering + 2*np.pi
            print("steering command AFTER", steering)
        # Debugging for toe-in angle
        # steering = 1
        # speed = 10.0

        control_list.append(u) # saving acceleration and steering speed
        state_list.append(env.my_hmmwv.state)
    
    observation, reward, done, info = env.step(steering, speed)


    if env.time > t_end:
        done = True
        print("Simulated for t_end. env.time:",env.time)
        break

    if env.lap_counter >= num_laps:
        done = True
        print("Completed num_laps. env.time:",env.time)
        break

plt.waitforbuttonpress()

execution_time_end = time.time()
print("execution time: ", execution_time_end - execution_time_start)

control_list = np.vstack(control_list)
state_list = np.vstack(state_list)

np.save("data/control.npy", control_list)
np.save("data/state.npy", state_list)

plt.figure()
plt.plot(env.t_stepsize, env.speed)
plt.title("longitudinal speed")
plt.xlabel("time [s]")
plt.ylabel("longitudinal speed [m/s]")
plt.savefig("longitudinal_speed.png")


plt.figure()
color = [i for i in range(len(env.x_trajectory))]
plt.scatter(env.x_trajectory, env.y_trajectory, c=color,s=1, label="trajectory")
plt.scatter(reduced_waypoints[0,1],reduced_waypoints[0,2], c='r',s=10, label="start")
plt.title("trajectory")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.savefig("trajectory.png")

plt.figure()
plt.plot(env.toein_FL, label="Front Left")
plt.plot(env.toein_FR, label="Front Right")
plt.plot(env.toein_RL, label="Rear Left")
plt.plot(env.toein_RR, label="Rear Right")
plt.plot(env.steering_driver, label="driver input steering")
plt.title("toe-in angle")
plt.legend()
plt.xlabel("time step")
plt.ylabel("toe-in angle [deg]")
plt.savefig("toe-in.png")

# plt.figure()
# plt.scatter(slip_angle_FL_list, Fy_over_muFz_list, label="Front Left")
# plt.title("slip angle vs Fy/muFz")
# plt.xlabel("slip angle [deg]")
# plt.ylabel("Fy/muFz")
# plt.legend()
# plt.xlim([0, 8])
# plt.savefig("slip_angle_vs_Fy_over_muFz.png")


plt.show()

