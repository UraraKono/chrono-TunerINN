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
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from argparse import Namespace
from EGP.regulators.pure_pursuit import *
from EGP.regulators.path_follow_mpc import *
from EGP.models.extended_kinematic import ExtendedKinematicModel
from EGP.models.configs import *
from EGP.helpers.closest_point import *
from EGP.helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *
# from chrono_env.data_gen_utils import load_map, friction_func
# from chrono_env.frenet_utils import centerline_to_frenet
from utilities import load_map, friction_func, centerline_to_frenet

# --------------
step_size = 2e-3 #simulation step size
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
SAVE_MODEL = True
MAP_DIR = './f1tenth-racetrack/'
SAVE_DIR = './data/'
t_end = 400
map_ind = 39 # 39 Zandvoort_raceline
ref_vx = 8.0
control_model = "pure_pursuit" # options: ext_kinematic, pure_pursuit
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
waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=7, reverse=False)
print("waypoints\n",waypoints.shape)

# Check if the waypoints are of the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
if waypoints.shape[1] == 4:
    waypoints = centerline_to_frenet(waypoints)
    reduced_rate = 1
else: # raceline
    reduced_rate = 5

waypoints[:, -2] = ref_vx

# sample every env.reduced_rate 10 waypoints for patch in visualization
reduced_waypoints = waypoints[::reduced_rate, :] 
s_max = np.max(reduced_waypoints[:, 0])
friction = [friction_func(i,s_max) for i in range(reduced_waypoints.shape[0])]
# friction = [1.0 + i/waypoints.shape[0] for i in range(reduced_waypoints.shape[0])]
# friction = [1.0 for i in range(reduced_waypoints.shape[0])]

# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
Kp = 5
Ki = 0.01
Kd = 0

env = ChronoEnv().make(timestep=step_size, control_period=0.1, waypoints=waypoints,
         sample_rate_waypoints=reduced_rate, friction=friction, speedPID_Gain=[Kp, Ki, Kd],
         steeringPID_Gain=[0.5,0,0], x0=reduced_waypoints[0,1], 
         y0=reduced_waypoints[0,2], w0=waypoints[0,3]-np.pi)

# ---------------
# Simulation loop
# ---------------

num_laps = 3  # Number of laps
lap_counter = 0

# Making sure that some config parameters are obtained from chrono, not from MPCConfigEXT
if control_model == "ext_kinematic":
    planner_ekin_mpc_config = MPCConfigEXT()
    reset_config(planner_ekin_mpc_config, env.vehicle_params)
    planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=planner_ekin_mpc_config), 
                                    waypoints=waypoints.copy(),
                                    config=planner_ekin_mpc_config) #path_follow_mpc.py
elif control_model == "pure_pursuit":
    # Init Pure-Pursuit regulator
    work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance
    conf.wpt_path = MAP_DIR + conf.wpt_path
    planner_pp = PurePursuitPlanner(conf, env.vehicle_params.WB)
    planner_pp.waypoints = waypoints.copy()

    ballT = env.vis.GetSceneManager().addSphereSceneNode(0.1)
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
        
        

while lap_counter < num_laps:
    # Render scene
    env.render()

    if (env.step_number % (env.control_step) == 0):
        if control_model == "ext_kinematic":
            # Solve MPC problem 
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, env.mpc_ox, env.mpc_oy = planner_ekin_mpc.plan(env.my_hmmwv.state)
            u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration
            speed = env.my_hmmwv.state[2] + u[0]*env.control_period
            steering = env.driver_inputs.m_steering*env.vehicle_params.MAX_STEER + u[1]*env.control_period # [-max steer,max steer]
        elif control_model == "pure_pursuit":
            planner_pp.waypoints = waypoints.copy()
            speed, steering = planner_pp.plan(observation['poses_x'], observation['poses_y'], observation['poses_theta'],
                                            work['tlad'], work['vgain'])  
            # Visualize the lookahead point of pure-pursuit
            pT = chrono.ChVectorD(planner_pp.lookahead_point[0], planner_pp.lookahead_point[1], 0.0)
            ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))


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
    
    observation, reward, done, info = env.step(speed, steering)


    if env.time > t_end:
        done = True
        print("Simulated for t_end. env.time:",env.time)
        break

    if lap_counter >= num_laps:
        done = True
        print("Completed num_laps. env.time:",env.time)
        break

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

plt.show()

