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

#######################################################################
'''
I think purpursuit.py needs the distance between the waypoints
But if we choose fitenth-racetrack, we have different spacing for centerline/raceline
So we need to specify that
'''

# --------------
step_size = 2e-3 #simulation step size
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
SAVE_MODEL = True
MAP_DIR = './f1tenth-racetrack/'
SAVE_DIR = './data/'
t_end = 60
map_ind = 39 # 39 Zandvoort_raceline
ref_vx = 8.0
# Init Pure-Pursuit regulator
work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance
# --------------

env = ChronoEnv()

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
    env.reduced_rate = 1
    w0=waypoints[0,3]-np.pi
else: # raceline
    env.reduced_rate = 5
    w0=waypoints[0,3]

# Rotate the map for 90 degrees in anti-clockwise direction 
# to match the map with the vehicle's initial orientation
# rotation_matrix = np.array([[0, 1], [-1, 0]])
# waypoints[:, 1:3] = np.dot(waypoints[:, 1:3], rotation_matrix)

waypoints[:, -2] = ref_vx

# sample every env.reduced_rate 10 waypoints for patch in visualization
reduced_waypoints = waypoints[::env.reduced_rate, :] 
s_max = np.max(reduced_waypoints[:, 0])
friction = [friction_func(i,s_max) for i in range(reduced_waypoints.shape[0])]
print("friction len", len(friction))
# friction = [0.4 + i/waypoints.shape[0] for i in range(reduced_waypoints.shape[0])]

# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
Kp = 1
Ki = 0.01
Kd = 0

env.make(config=MPCConfigEXT(), friction=friction, waypoints=waypoints,sample_rate_waypoints=env.reduced_rate,
         reduced_waypoints=reduced_waypoints, speedPID_Gain=[Kp, Ki, Kd],
         steeringPID_Gain=[1,0,0], x0=reduced_waypoints[0,1], y0=reduced_waypoints[0,2], w0=w0)

# ---------------
# Simulation loop
# ---------------

env.my_hmmwv.GetVehicle().EnableRealtime(True)
num_laps = 3  # Number of laps
lap_counter = 0

# Reset the simulation time
env.my_hmmwv.GetSystem().SetChTime(0)

# Making sure that some config parameters are obtained from chrono, not from MPCConfigEXT
reset_config(env, env.vehicle_params) 

# env.planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=env.config), 
#                                 waypoints=waypoints,
#                                 config=env.config) #path_follow_mpc.py
conf.wpt_path = MAP_DIR + conf.wpt_path
planner_pp = PurePursuitPlanner(conf, env.vehicle_params.WB)
planner_pp.waypoints = waypoints.copy()

plt.figure()
plt.plot(planner_pp.waypoints[:,1], planner_pp.waypoints[:,2], label="waypoints")
plt.show()

speed    = 0
steering = 0
control_list = []
state_list = []

execution_time_start = time.time()
ballT = env.vis.GetSceneManager().addSphereSceneNode(0.1)

while lap_counter < num_laps:
    # Render scene
    env.render()

    if (env.step_number % (env.control_step) == 0):
        pos = env.my_hmmwv.GetVehicle().GetPos()
        planner_pp.waypoints = waypoints.copy()
        speed, steering = planner_pp.plan(pos.x, pos.y, env.my_hmmwv.state[3],
                                                work['tlad'], work['vgain'])
        print("pure pursuit input speed", speed, "steering angle [rad]", steering)

        pT = chrono.ChVectorD(planner_pp.lookahead_point[0], planner_pp.lookahead_point[1],  pos.z)
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

        # control_list.append(u) # saving acceleration and steering speed
        state_list.append(env.my_hmmwv.state)
    
    env.step(steering, speed)


    if env.time > t_end:
        print("env.time",env.time)
        break

execution_time_end = time.time()
print("execution time: ", execution_time_end - execution_time_start)

# control_list = np.vstack(control_list)
# state_list = np.vstack(state_list)

# np.save("data/control.npy", control_list)
# np.save("data/state.npy", state_list)

plt.figure()
plt.plot(env.t_stepsize, env.speed)
plt.title("longitudinal speed")
plt.xlabel("time [s]")
plt.ylabel("longitudinal speed [m/s]")
plt.savefig("longitudinal_speed.png")


plt.figure()
color = [i for i in range(len(env.x_trajectory))]
plt.scatter(env.x_trajectory, env.y_trajectory, c=color,s=1, label="trajectory")
plt.scatter(reduced_waypoints[0,1],reduced_waypoints[0,2], c='r',s=5, label="start")
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

