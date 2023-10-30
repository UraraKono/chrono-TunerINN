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
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import numpy as np
import json
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
import time
import yaml
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.extended_kinematic import ExtendedKinematicModel
from models.configs import *
from helpers.closest_point import *
from helpers.track import Track
from chrono_env.environment_new import ChronoEnv

# --------------
step_size = 2e-3
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
# Program parameters
model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
# currently only "custom_track" works for frenet
map_name = 'custom_track'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
use_dyn_friction = False
gp_mpc_type = 'frenet'  # cartesian, frenet
render_every = 30  # render graphics every n sim steps
constant_speed = True
constant_friction = 0.7
number_of_laps = 20
SAVE_MODEL = True
t_end = 10
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

    # Convert waypoints to ChBezierCurve
    curve_points = [chrono.ChVectorD(waypoint[1], waypoint[2], 0.6) for waypoint in waypoints]
    curve = chrono.ChBezierCurve(curve_points, True) # True = closed curve
    
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

friction = [0.4 + i/waypoints.shape[0] for i in range(waypoints.shape[0])]

# Define the patch coordinates
patch_coords = [[waypoint[1], waypoint[2], 0.0] for waypoint in waypoints]

# Kp = 0.6
# Ki = 0.2
# Kd = 0.3
Kp = 0.4*10
Ki = 0
Kd = 0

env.make(config=MPCConfigEXT(), friction=friction, patch_coords=patch_coords, 
         waypoints=waypoints, curve=curve, speedPID_Gain=[Kp, Ki, Kd])

# Create the terrain
# env.terrain, env.viz_patch = init_terrain(env, friction, patch_coords, waypoints)

# path = curve
# # print("path\n", path)

# npoints = path.getNumPoints()

# path_asset = chrono.ChLineShape()
# path_asset.SetLineGeometry(chrono.ChLineBezier(path))
# path_asset.SetName("test path")
# path_asset.SetColor(chrono.ChColor(0.8, 0.0, 0.0))
# path_asset.SetNumRenderPoints(max(2 * npoints, 400))
# env.viz_patch.GetGroundBody().AddVisualShape(path_asset)

# MPC_params = MPCConfig()
# Convert waypoints to ChBezierCurve
# mpc_curve_points = [chrono.ChVectorD(i/10 + 0.1, i/10 + 0.1, 0.6) for i in range(env.config.TK + 1)] #これはなにをやっているの？map情報からのwaypointガン無視してない？
# mpc_curve = chrono.ChBezierCurve(mpc_curve_points, True) # True = closed curve

# npoints = mpc_curve.getNumPoints()

# mpc_path_asset = chrono.ChLineShape()
# mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))
# mpc_path_asset.SetName("MPC path")
# mpc_path_asset.SetColor(chrono.ChColor(0.0, 0.0, 0.8))
# mpc_path_asset.SetNumRenderPoints(max(2 * npoints, 400))
# env.viz_patch.GetGroundBody().AddVisualShape(mpc_path_asset)

# What's the difference between path_asset and mpc_path_asset?

# Create the PID lateral controller
# steeringPID = veh.ChPathSteeringController(mpc_curve)
# steeringPID.SetLookAheadDistance(5) 
# steeringPID.SetGains(0.8, 0, 0)
# steeringPID.Reset(my_hmmwv.GetVehicle())

# speedPID = veh.ChSpeedController()

# speedPID.SetGains(Kp, Ki, Kd)
# speedPID.Reset(env.my_hmmwv.GetVehicle())

# I did this in env.make but not sure if I need to do it again here
# # Create the vehicle Irrlicht application
# env.vis = init_irrlicht_vis(env.my_hmmwv)

# ---------------
# Simulation loop
# ---------------

# steeringPID_output = 0

env.my_hmmwv.GetVehicle().EnableRealtime(True)
num_laps = 3  # Number of laps
lap_counter = 0

# Define the starting point and a tolerance distance
starting_point = chrono.ChVectorD(-100, 0, 0.6)  # Replace with the actual starting point coordinates
tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

# Reset the simulation time
env.my_hmmwv.GetSystem().SetChTime(0)

# # Time interval between two render frames
# env.render_step_size = 1.0 / 50  # FPS = 50 frame per second

# env.render_steps = math.ceil(env.render_step_size / step_size)
# env.step_number = 0

# vehicle_params = get_vehicle_parameters(my_hmmwv)

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

# control_step = planner_ekin_mpc.config.DTK/step_size  

# u_acc = []
# # u_steer_speed = [] #steering speed from MPC is not used. ox/oy are used instead
# t_controlperiod = [] # time list every control_period
# t_stepsize = [] # time list every step_size
# speed = []
# speed_ref = []
# speedPID_output = 1.0
# target_speed = 0
# target_acc = 0

# steering_output = 0
# target_steering_speed = 0

# driver_inputs = veh.DriverInputs()
# driver_inputs.m_throttle = throttle_value
# driver_inputs.m_braking = 0.0

# driver = veh.ChDriver(env.my_hmmwv.GetVehicle()) #This command does NOT work. Never use ChDriver!

while lap_counter < num_laps:
    # Render scene
    env.render()
    # if (env.step_number % (env.render_steps) == 0) :
    #     env.vis.BeginScene()
    #     env.vis.Render()
    #     env.vis.EndScene()

    env.step()

    

    # # Increment frame number
    # env.step_number += 1

    # # Driver inputs
    # time = env.my_hmmwv.GetSystem().GetChTime()
    # driver_inputs.m_steering = np.clip(steering_output, -1.0, +1.0)
    # speedPID_output = np.clip(speedPID_output, -1.0, +1.0)

    # if speedPID_output > 0:
    #     driver_inputs.m_throttle = speedPID_output
    #     driver_inputs.m_braking = 0.0
    # else:
    #     driver_inputs.m_throttle = 0.0
    #     driver_inputs.m_braking = -speedPID_output


    # # Update modules (process inputs from other modules)
    # env.terrain.Synchronize(time)
    # env.my_hmmwv.Synchronize(time, driver_inputs, env.terrain)
    # env.vis.Synchronize("", driver_inputs)
    
    # vehicle_state = get_vehicle_state(env)
    # # vehicle_state[2] = speedPID.GetCurrentSpeed() # vx from get_vehicle_state is a bit different from speedPID.GetCurrentSpeed()
    # t_stepsize.append(time)
    # speed.append(vehicle_state[2])
    # speed_ref.append(target_speed)
    # # print("speed", vehicle_state)
    
    # # Solve MPC every control_step
    # if (env.step_number % (control_step) == 0) : 
    #     # print("step number", step_number)
    #     u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_ekin_mpc.plan(
    #         vehicle_state)
    #     u[0] = u[0] / env.vehicle_params.MASS  # Force to acceleration
    #     target_acc = u[0]
    #     target_steering_speed = u[1]
    #     target_speed = vehicle_state[2] + target_acc*planner_ekin_mpc.config.DTK
    #     steering_output = driver_inputs.m_steering + u[1]*planner_ekin_mpc.config.DTK/env.config.MAX_STEER # Overshoots soooo much
    #     # steering_output = u[1]*planner_ekin_mpc.config.DTK/env.config.MAX_STEER # This one works better lol. It doesn't make sesnse
    #     u_acc.append(u[0])
    #     t_controlperiod.append(time)
    #     # print("vehicle_state.vx", vehicle_state[2],"speedPID.GetCurrentSpeed()", speedPID.GetCurrentSpeed())
    #     # print("mpc ref path x", mpc_ref_path_x) #list length of 16 (TK + 1)
    #     # print("mpc ref path y", mpc_ref_path_y)
    #     # print("mpc pred x", mpc_pred_x)
    #     # print("mpc pred y", mpc_pred_y)
    #     # print("mpc ox", mpc_ox)
    #     # print("mpc oy", mpc_oy)
    #     # print("target_speed", target_speed)
        
        # if mpc_ox is not None and not np.any(np.isnan(mpc_ox)):
        #     # Update mpc_path_asset with mpc_pred
        #     mpc_curve_points = [chrono.ChVectorD(mpc_ox[i], mpc_oy[i], 0.6) for i in range(env.config.TK + 1)]
        #     mpc_curve = chrono.ChBezierCurve(mpc_curve_points, False) # True = closed curve
        #     env.mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))

        # # Advance simulation for one timestep for all modules
        # speedPID_output = speedPID.Advance(env.my_hmmwv.GetVehicle(), target_speed, step_size)
        # # print('speed pid output', speedPID_output)
    

    # Check if the vehicle has crossed the starting point
    pos = env.my_hmmwv.GetVehicle().GetPos()
    distance_to_start = (pos - starting_point).Length()

    if distance_to_start < tolerance:
        lap_counter += 1
        print(f"Completed lap {lap_counter}")

    if env.time > t_end:
        print("env.time",env.time)
        break

fig, ax = plt.subplots(2,1)
ax[0].plot(env.t_stepsize, env.speed, label="speed")
ax[0].plot(env.t_stepsize, env.speed_ref, label="speed ref")
ax[0].set_title("longitudinal speed")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel(" longitudinal speed [m/s]")
ax[0].legend()
ax[1].plot(env.t_controlperiod, env.u_acc, label="acceleration")
ax[0].set_title("longitudinal acceleration")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("acceleration [m/s^2]")
ax[1].legend()
plt.savefig("longitudinal_speed_and_acceleration.png")
plt.show()

