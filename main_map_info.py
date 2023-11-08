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
# import pychrono.irrlicht as chronoirr
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml
import time
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.extended_kinematic import ExtendedKinematicModel
from models.configs import *
from helpers.closest_point import *
from helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import get_vehicle_state
from chrono_env.data_gen_utils import load_map, friction_func
from chrono_env.frenet_utils import cartesian_to_frenet, frenet_to_cartesian, centerline_to_frenet

# --------------
step_size = 2e-3 #simulation step size
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
# Program parameters
# model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
# currently only "custom_track" works for frenet
# map_name = 'SaoPaulo'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
# use_dyn_friction = False
# gp_mpc_type = 'frenet'  # cartesian, frenet
# render_every = 30  # render graphics every n sim steps
# constant_speed = True
# constant_friction = 0.7
# number_of_laps = 20
SAVE_MODEL = True
MAP_DIR = './f1tenth-racetrack/'
t_end = 400
map_ind = 39 # 39 Zandvoort_raceline
# --------------

env = ChronoEnv(step_size, throttle_value)

# Load map config file
with open('maps/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=7, reverse=False)
print("waypoints\n",waypoints.shape)

# Convert waypoints to ChBezierCurve
curve_points = [chrono.ChVectorD(waypoint[1], waypoint[2], 0.6) for waypoint in waypoints]
curve = chrono.ChBezierCurve(curve_points, True) # True = closed curve

# Check if the waypoints are of the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
if waypoints.shape[1] == 4:
    waypoints = centerline_to_frenet(waypoints)
    env.reduced_rate = 1
else: # raceline
    env.reduced_rate = 5

# Rotate the map for 90 degrees in anti-clockwise direction 
# to match the map with the vehicle's initial orientation
rotation_matrix = np.array([[0, 1], [-1, 0]])
waypoints[:, 1:3] = np.dot(waypoints[:, 1:3], rotation_matrix)

# sample every env.reduced_rate 10 waypoints for patch in visualization
reduced_waypoints = waypoints[::env.reduced_rate, :] 
s_max = np.max(reduced_waypoints[:, 0])
friction = [friction_func(i,s_max) for i in range(reduced_waypoints.shape[0])]
# friction = [0.4 + i/waypoints.shape[0] for i in range(reduced_waypoints.shape[0])]

veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
Kp = 0.4*10
Ki = 0
Kd = 0

env.make(config=MPCConfigEXT(), friction=friction, waypoints=waypoints, 
         reduced_waypoints=reduced_waypoints, curve=curve, speedPID_Gain=[Kp, Ki, Kd],
           ini_pos=chrono.ChVectorD(reduced_waypoints[0,1], reduced_waypoints[0,2], 0.5))

# ---------------
# Simulation loop
# ---------------

env.my_hmmwv.GetVehicle().EnableRealtime(True)
num_laps = 3  # Number of laps
lap_counter = 0

# # Define the starting point and a tolerance distance
# # starting_point = chrono.ChVectorD(-70, 0, 0.6)  # Replace with the actual starting point coordinates
# tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

# Reset the simulation time
env.my_hmmwv.GetSystem().SetChTime(0)

# Making sure that some config parameters are obtained from chrono, not from MPCConfigEXT
env.config.LENGTH      = env.vehicle_params.LENGTH
env.config.WIDTH       = env.vehicle_params.WIDTH
env.config.LR          = env.vehicle_params.LR
env.config.LF          = env.vehicle_params.LF
env.config.WB          = env.vehicle_params.WB
env.config.MIN_STEER   = env.vehicle_params.MIN_STEER
env.config.MAX_STEER   = env.vehicle_params.MAX_STEER
env.config.MAX_STEER_V = env.vehicle_params.MAX_STEER_V
env.config.MAX_SPEED   = env.vehicle_params.MAX_SPEED
env.config.MIN_SPEED   = env.vehicle_params.MIN_SPEED
env.config.MAX_ACCEL   = env.vehicle_params.MAX_ACCEL
env.config.MAX_DECEL   = env.vehicle_params.MAX_DECEL
env.config.MASS        = env.vehicle_params.MASS    

env.planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=env.config), 
                                waypoints=waypoints,
                                config=env.config) #path_follow_mpc.py

# driver = veh.ChDriver(env.my_hmmwv.GetVehicle()) #This command does NOT work. Never use ChDriver!
speed    = 0
steering = 0
control_list = []
state_list = []

execution_time_start = time.time()

while lap_counter < num_laps:
    # Render scene
    env.render()

    if (env.step_number % (env.control_step) == 0):
        # Solve MPC problem
        # env.my_hmmwv.state = get_vehicle_state(env) 
        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, env.mpc_ox, env.mpc_oy = env.planner_ekin_mpc.plan(env.my_hmmwv.state)
        u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration
        # print("u", u)
        speed = env.my_hmmwv.state[2] + u[0]*env.planner_ekin_mpc.config.DTK
        steering = env.driver_inputs.m_steering + u[1]*env.planner_ekin_mpc.config.DTK/env.config.MAX_STEER # [-1,1]
        # print("steering input", steering)
        # Debugging for toe-in angle
        # steering = 1
        speed = 10.0

        control_list.append(u) # saving acceleration and steering speed
        state_list.append(env.my_hmmwv.state)
    
    env.step(speed, steering)


    if env.time > t_end:
        print("env.time",env.time)
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
plt.title("trajectory")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
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

