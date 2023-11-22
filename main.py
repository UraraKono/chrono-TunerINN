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
# This file uses the maps in /maps directory if map_name == 'custom_track' 
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
from argparse import Namespace
from EGP.regulators.pure_pursuit import *
from EGP.regulators.path_follow_mpc import *
from EGP.models.extended_kinematic import ExtendedKinematicModel
from EGP.models.configs import *
from EGP.helpers.closest_point import *
from EGP.helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *
# from chrono_env.data_gen_utils import friction_func
from utilities import friction_func

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
# constant_speed = True
constant_friction = 0.7
# number_of_laps = 20
SAVE_MODEL = True
t_end = 60
# --------------

env = ChronoEnv(step_size, throttle_value)

# Load map config file
with open('EGP/configs/config_%s.yaml' % 'SaoPaulo') as file:  # map_name -- SaoPaulo
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
 
# friction = [0.4 + i/waypoints.shape[0] for i in range(waypoints.shape[0])]
friction = [constant_friction for i in range(waypoints.shape[0])]
# s_max = np.max(reduced_waypoints[:, 0])
# friction = [friction_func(i,s_max) for i in range(reduced_waypoints.shape[0])]

# Define the patch coordinates
patch_coords = [[waypoint[1], waypoint[2], 0.0] for waypoint in waypoints]

# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
Kp = 5
Ki = 0
Kd = 0
# Kp = 3.8
# Ki = 0
# Kd = 0

env.make(config=MPCConfigEXT(), friction=friction,waypoints=waypoints,
         reduced_waypoints=waypoints, speedPID_Gain=[Kp, Ki, Kd],
         steeringPID_Gain=[1,0,0],x0=waypoints[0,1], y0=waypoints[0,2], w0=0)

# ---------------
# Simulation loop
# ---------------

env.my_hmmwv.GetVehicle().EnableRealtime(True)
num_laps = 5  # Number of laps
lap_counter = 0

# Reset the simulation time
env.my_hmmwv.GetSystem().SetChTime(0)

reset_config(env, env.vehicle_params)  

env.planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=env.config), 
                                waypoints=waypoints,
                                config=env.config) #path_follow_mpc.py

speed    = 0
steering = 0
control_list = []
state_list = []

execution_time_start = time.time()

while lap_counter < num_laps:
    # Render scene
    env.render()

    if (env.step_number % (env.control_step) == 0):
        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, env.mpc_ox, env.mpc_oy = env.planner_ekin_mpc.plan(env.my_hmmwv.state)
        u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration
        # print("u", u)
        speed = env.my_hmmwv.state[2] + u[0]*env.planner_ekin_mpc.config.DTK
        steering = env.driver_inputs.m_steering*env.config.MAX_STEER + u[1]*env.planner_ekin_mpc.config.DTK # [deg]
        # print("steering input", steering)
        # Debugging for toe-in angle
        # steering = 1*env.config.MAX_STEER
        # speed = 3.0

        control_list.append(u) # saving acceleration and steering speed
        state_list.append(env.my_hmmwv.state)
    
    env.step(steering, speed)

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
plt.scatter(waypoints[0,1],waypoints[0,2], c='r',s=5, label="start")
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
