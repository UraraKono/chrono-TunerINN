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
# HMMWV constant radius turn.
# This program uses explicitly a ChPathSteeringControler.
#
# The vehicle reference frame has Z up, X towards the front of the vehicle, and
# Y pointing to the left.
#
# Apply the MPC solution (acceleration and steering speed) to the vehicle
# The steering speed from MPC is fed into driver's steering input
# ChSpeedController is used to enforce the acceleration from MPC
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
import time
import yaml
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_single import GPSingleModel
from models.configs import *
from helpers.closest_point import *
from helpers.track import Track
import numpy as np
import json

# =============================================================================
step_size = 2e-3
throttle_value = 0.3
# =============================================================================

@dataclass
class MPCConfig:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 15  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.000001, 2.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000001, 2.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # final state error matrix, penalty  for the final state constraints

    # Calc ref parameters
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic

@dataclass
class VehicleParameters:
    LENGTH: float = 4.298  # Length of the vehicle [m]
    # WIDTH: float = 1.674  # Width of the vehicle [m]
    # LR: float = 1.50876
    # LF: float = 0.88392
    # WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    # MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    # MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MIN_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    # MASS: float = 1225.887  # Vehicle mass

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def init_vehicle():
    # Create the vehicle system
    my_hmmwv = veh.HMMWV_Full()
    my_hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    my_hmmwv.SetChassisFixed(False)
    my_hmmwv.SetInitPosition(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0.5),chrono.QUNIT))
    my_hmmwv.SetPowertrainType(veh.PowertrainModelType_SHAFTS)
    my_hmmwv.SetDriveType(veh.DrivelineTypeWV_RWD)
    my_hmmwv.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
    # my_hmmwv.SetSteeringType(veh.SteeringTypeWV_RACK_PINION)
    my_hmmwv.SetTireType(veh.TireModelType_TMEASY)
    my_hmmwv.SetTireStepSize(step_size)
    my_hmmwv.Initialize()

    my_hmmwv.SetChassisVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetSuspensionVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetSteeringVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetWheelVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetTireVisualizationType(veh.VisualizationType_MESH)
    
    return my_hmmwv

def init_terrain(friction, patch_coords, waypoints):
    rest_values = [0.01] * len(patch_coords)
    young_modulus_values = [2e7] * len(patch_coords)
    patch_mats = [chrono.ChMaterialSurfaceSMC() for _ in range(len(patch_coords))]
    for i, patch_mat in enumerate(patch_mats):
        patch_mat.SetFriction(friction[i])
        patch_mat.SetRestitution(rest_values[i])
        patch_mat.SetYoungModulus(young_modulus_values[i])

    terrain = veh.RigidTerrain(my_hmmwv.GetSystem())

    # Loop over the patch materials and coordinates to add each patch to the terrain
    patches = []
    for i, patch_mat in enumerate(patch_mats):
        coords = patch_coords[i]
        psi = waypoints[i, 3]
        if i == len(patch_mats) - 1:
            s = waypoints[i, 0] - waypoints[i-1,0]
        else:    
            s = waypoints[i+1, 0] - waypoints[i,0]
        r = chrono.ChQuaternionD()
        r.Q_from_AngZ(psi)
        patch = terrain.AddPatch(patch_mat, chrono.ChCoordsysD(chrono.ChVectorD(coords[0], coords[1], coords[2]), r), s, s)
        patches.append(patch)

    viz_patch = terrain.AddPatch(patch_mats[2], chrono.CSYSNORM, s, s)
    
    # Set color of patch based on friction value
    for i, patch in enumerate(patches):
        # print(friction[i])
        RGB_value = 1 - (friction[i] - 0.4)
        # print(RGB_value, friction[i])
        patch.SetColor(chrono.ChColor(RGB_value, RGB_value, RGB_value))

    # for patch in patches:
    #     patch.SetTexture(veh.GetDataFile("terrain/textures/concrete.jpg"), 10, 10)

    terrain.Initialize()
    return terrain, viz_patch

def init_irrlicht_vis(ego_vehicle):
    # Create the vehicle Irrlicht interface
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('Constant radius test')
    vis.SetWindowSize(1280, 1024)
    vis.SetHUDLocation(500, 20)
    vis.Initialize()
    vis.AddLogo()
    vis.AddLightDirectional()
    vis.SetChaseCamera(chrono.ChVectorD(0.0, 0.0, 1.75), 6.0, 4.5)
    vis.AddSkyBox()
    vis.AttachVehicle(ego_vehicle.GetVehicle())

    return vis

def get_vehicle_parameters(vehicle):
    params = VehicleParameters()
    params.MASS = vehicle.GetVehicle().GetMass()
    params.WB   = vehicle.GetVehicle().GetWheelbase()
    params.MIN_STEER = -vehicle.GetVehicle().GetMaxSteeringAngle()
    params.MAX_STEER = +vehicle.GetVehicle().GetMaxSteeringAngle()
    params.WIDTH = vehicle.GetVehicle().GetWheeltrack(0)
    chassisPos = vehicle.GetVehicle().GetChassis().GetPos()
    COMPos = vehicle.GetVehicle().GetChassis().GetCOMFrame().coord.pos
    absPosCOM = COMPos + chassisPos
    fw = vehicle.GetVehicle().GetAxle(0).GetWheels()[1].GetPos()
    tmp = fw - absPosCOM
    params.LF = np.linalg.norm(np.array([tmp.x, tmp.y, tmp.z]))
    
    params.LR = params.WB - params.LF 

    return params

def get_vehicle_state(vehicle):
    
    pos = vehicle.GetVehicle().GetPos()
    power = vehicle.GetVehicle().GetPowertrain().GetOutputTorque()
    # print("power train torque", power)
    # print("Vehicle position:", pos)
    x = pos.x
    y = pos.y
    rotation = vehicle.GetVehicle().GetRot()
    # print("Vehicle rotation:", rotation)

    # Get the angular velocities of the chassis in the local frame
    chassis_velocity = vehicle.GetVehicle().GetChassis().GetBody().GetWvel_loc()
    yaw_rate = chassis_velocity.z

    euler_angles = rotation.Q_to_Euler123()
    # print("Vehicle rotation (Euler angles):", euler_angles)
    roll_angle = euler_angles.x
    pitch_angle = euler_angles.y
    yaw_angle = euler_angles.z
    # print("Vehicle roll angle:", roll_angle)
    # print("Vehicle pitch angle:", pitch_angle)
    # print("Vehicle yaw angle:", yaw_angle)

    # Get the linear velocity of the chassis in the global frame
    chassis_velocity = vehicle.GetVehicle().GetChassis().GetBody().GetPos_dt()

    # Get the rotation matrix of the chassis
    chassis_rot = vehicle.GetVehicle().GetChassis().GetRot()
    rot_matrix = chrono.ChMatrix33D(chassis_rot)

    # Create an empty ChMatrix33D for the transpose
    transpose_rot_matrix = chrono.ChMatrix33D()

    # Manually set the transpose of the rotation matrix
    transpose_rot_matrix[0, 0] = rot_matrix[0, 0]
    transpose_rot_matrix[1, 0] = rot_matrix[0, 1]
    transpose_rot_matrix[2, 0] = rot_matrix[0, 2]
    transpose_rot_matrix[0, 1] = rot_matrix[1, 0]
    transpose_rot_matrix[1, 1] = rot_matrix[1, 1]
    transpose_rot_matrix[2, 1] = rot_matrix[1, 2]
    transpose_rot_matrix[0, 2] = rot_matrix[2, 0]
    transpose_rot_matrix[1, 2] = rot_matrix[2, 1]
    transpose_rot_matrix[2, 2] = rot_matrix[2, 2]

    # Transform the global frame velocity to the local frame
    chassis_velocity_local = transpose_rot_matrix * chassis_velocity

    # Extract the y-component of the velocity in the local frame
    velocity_y_local = chassis_velocity_local.y

    # Print the velocity in the y direction in the local frame
    # print("Vehicle velocity in y direction (local frame):", velocity_y_local)
    vy = velocity_y_local

    # Extract the y-component of the velocity in the local frame
    velocity_x_local = chassis_velocity_local.x

    # Print the velocity in the x direction in the local frame
    # print("Vehicle velocity in x direction (local frame):", velocity_x_local)
    vx = velocity_x_local

    # Extract the y-component of the velocity
    velocity_y = chassis_velocity.y

    # Print the velocity in the y direction
    # print("Vehicle velocity in y direction:", velocity_y)
    
    # rotation = list(rotation)
    # steering_angle = vehicle.GetVehicle()

    # acc = vehicle.GetVehicle().GetAcc()
    # print("acc", acc)

    max_steering_angle = vehicle.GetVehicle().GetMaxSteeringAngle()
    # print("max steering angle:", max_steering_angle)

    # print("steering angle", driver_inputs.m_steering)

    # get vehicle mass
    mass = vehicle.GetVehicle().GetMass()
    # print("Vehicle mass:", mass)

    # Get tire force
    tf_FL = vehicle.GetVehicle().GetTire(0, veh.LEFT).ReportTireForce(terrain)
    tf_FR = vehicle.GetVehicle().GetTire(0, veh.RIGHT).ReportTireForce(terrain)
    tf_RL = vehicle.GetVehicle().GetTire(1, veh.LEFT).ReportTireForce(terrain)
    tf_RR = vehicle.GetVehicle().GetTire(1, veh.RIGHT).ReportTireForce(terrain)
    # print("   Front left:  ", tf_FL.force.x, " ", tf_FL.force.y, " ", tf_FL.force.z)
    # print("   Front right: ", tf_FR.force.x, " ", tf_FR.force.y, " ", tf_FR.force.z)
    # print("   Rear left:   ", tf_RL.force.x, " ", tf_RL.force.y, " ", tf_RL.force.z)
    # print("   Rear right:  ", tf_RR.force.x, " ", tf_RR.force.y, " ", tf_RR.force.z)

    # get tractive force
    mu = 0.8

    Fx_left = tf_FL.force.z * mu
    Fx_right = tf_FR.force.z * mu
    Rx_left = tf_RL.force.z * mu
    Rx_right = tf_RR.force.z * mu
    # print("   Fx_left:  ", Fx_left)
    # print("   Fx_right: ", Fx_right)
    # print("   Rx_left:   ", Rx_left)
    # print("   Rx_right:  ", Rx_right)

    # my_vehicle = veh.ChWheeledVehicle("my_vehicle")
    my_driver = veh.ChDriver(vehicle.GetVehicle())
    throttle = my_driver.GetThrottle()
    steering = my_driver.GetSteering() # steering input [-1,+1]
    braking = my_driver.GetBraking()

    inputs = my_driver.GetInputs()

    driver_glob_location = vehicle.GetVehicle().GetDriverPos()  # global location of the driver
    # (Include the rest of the code inside the original loop)


    # vehicle state for single-track model
    vehicle_state = np.array([x,  # x
                              y,  # y
                              vx,  # vx
                              yaw_angle,  # yaw angle
                              vy,  # vy
                              yaw_rate,  # yaw rate
                              steering*max_steering_angle,  # steering angle
                            ])
    # print("vehicle state:", vehicle_state)

    return vehicle_state

if __name__ == '__main__':

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
    t_end = 40

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

        # print(centerline_descriptor)
        # print(centerline_descriptor.shape)

        track = Track(centerline_descriptor=centerline_descriptor, track_width=10.0, reference_speed=5.0)
        waypoints = track.get_reference_trajectory()
        print('waypoints\n',waypoints.shape)

        # Convert waypoints to ChBezierCurve
        curve_points = [chrono.ChVectorD(waypoint[1], waypoint[2], 0.6) for waypoint in waypoints]
        curve = chrono.ChBezierCurve(curve_points, True) # True = closed curve
        
    veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

    # Create the HMMWV vehicle
    my_hmmwv = init_vehicle()

    friction = [0.5 + i/waypoints.shape[0] for i 
                in range(waypoints.shape[0])]
    
    # Define the patch coordinates
    patch_coords = [[waypoint[1], waypoint[2], 0.0] for waypoint in waypoints]
    # print('patch coords')
    # print(patch_coords)
    
    # Create the terrain
    terrain, viz_patch = init_terrain(friction, patch_coords, waypoints)

    path = curve
    print("path\n", path)
 
    npoints = path.getNumPoints()

    path_asset = chrono.ChLineShape()
    path_asset.SetLineGeometry(chrono.ChLineBezier(path))
    path_asset.SetName("test path")
    path_asset.SetColor(chrono.ChColor(0.8, 0.0, 0.0))
    path_asset.SetNumRenderPoints(max(2 * npoints, 400))
    viz_patch.GetGroundBody().AddVisualShape(path_asset)

    MPC_params = MPCConfig()
    # Convert waypoints to ChBezierCurve
    mpc_curve_points = [chrono.ChVectorD(i/10 + 0.1, i/10 + 0.1, 0.6) for i in range(MPC_params.TK + 1)] #これはなにをやっているの？map情報からのwaypointガン無視してない？
    mpc_curve = chrono.ChBezierCurve(mpc_curve_points, True) # True = closed curve

    npoints = mpc_curve.getNumPoints()

    mpc_path_asset = chrono.ChLineShape()
    mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))
    mpc_path_asset.SetName("MPC path")
    mpc_path_asset.SetColor(chrono.ChColor(0.0, 0.0, 0.8))
    mpc_path_asset.SetNumRenderPoints(max(2 * npoints, 400))
    viz_patch.GetGroundBody().AddVisualShape(mpc_path_asset)

    # What's the difference between path_asset and mpc_path_asset?
    
    # Create the PID lateral controller
    # steeringPID = veh.ChPathSteeringController(mpc_curve)
    # steeringPID.SetLookAheadDistance(5) 
    # steeringPID.SetGains(0.8, 0, 0)
    # steeringPID.Reset(my_hmmwv.GetVehicle())

    speedPID = veh.ChSpeedController()
    Kp = 0.6
    Ki = 0.2
    Kd = 0.3
    speedPID.SetGains(Kp, Ki, Kd)
    speedPID.Reset(my_hmmwv.GetVehicle())

    # Create the vehicle Irrlicht application
    vis = init_irrlicht_vis(my_hmmwv)


    # # Visualization of controller points (target)
    # ballT = vis.GetSceneManager().addSphereSceneNode(0.1)
    # ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)

    # Visualization of controller points (sentinel & target)
    ballS = vis.GetSceneManager().addSphereSceneNode(0.1)
    ballT = vis.GetSceneManager().addSphereSceneNode(0.1)
    ballS.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 255, 0, 0)
    ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)


    # ---------------
    # Simulation loop
    # ---------------

    # steeringPID_output = 0

    my_hmmwv.GetVehicle().EnableRealtime(True)
    num_laps = 3  # Number of laps
    lap_counter = 0

    # Define the starting point and a tolerance distance
    starting_point = chrono.ChVectorD(-100, 0, 0.6)  # Replace with the actual starting point coordinates
    tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

    # Reset the simulation time
    my_hmmwv.GetSystem().SetChTime(0)

    # Time interval between two render frames
    render_step_size = 1.0 / 50  # FPS = 50 frame per second

    render_steps = math.ceil(render_step_size / step_size)
    step_number = 0
    render_frame = 0

    vehicle_params = get_vehicle_parameters(my_hmmwv)

    correct_config = MPCConfigEXT()
    correct_config.LENGTH = vehicle_params.LENGTH
    correct_config.WIDTH = vehicle_params.WIDTH
    correct_config.LR = vehicle_params.LR
    correct_config.LF = vehicle_params.LF
    correct_config.WB = vehicle_params.WB
    correct_config.MIN_STEER = vehicle_params.MIN_STEER
    correct_config.MAX_STEER = vehicle_params.MAX_STEER
    correct_config.MAX_STEER_V = vehicle_params.MAX_STEER_V
    correct_config.MAX_SPEED = vehicle_params.MAX_SPEED
    correct_config.MIN_SPEED = vehicle_params.MIN_SPEED
    correct_config.MAX_ACCEL = vehicle_params.MAX_ACCEL
    correct_config.MAX_DECEL = vehicle_params.MAX_DECEL
    correct_config.MASS = vehicle_params.MASS    

    planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=correct_config), 
                                    waypoints=waypoints,
                                    config=correct_config) #path_follow_mpc.py
    
    control_step = planner_ekin_mpc.config.DTK/step_size  # control step in sim steps
    
    u_acc = []
    # u_steer_speed = [] #steering speed from MPC is not used. ox/oy are used instead
    t_controlperiod = [] # time list every control_period
    t_stepsize = [] # time list every step_size
    speed = []
    speed_ref = []
    speedPID_output = 0
    target_speed = 0
    target_acc = 0
    steering_output = 0

    while lap_counter < num_laps:
        # target_speed += target_acc*step_size
        # Render scene
        if (step_number % (render_steps) == 0) :
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

        # Increment frame number
        step_number += 1

        # Driver inputs
        time = my_hmmwv.GetSystem().GetChTime()
        driver_inputs = veh.DriverInputs()
        driver_inputs.m_steering = np.clip(steering_output, -1.0, +1.0)
        driver_inputs.m_throttle = throttle_value
        driver_inputs.m_braking = 0.0

        # # Update sentinel and target location markers for the path-follower controller.
        # pT = steeringPID.GetTargetLocation()
        # ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))

        # Update sentinel and target location markers for the path-follower controller.
        # pS = steeringPID.GetSentinelLocation()
        # pT = steeringPID.GetTargetLocation()
        # ballS.setPosition(chronoirr.vector3df(pS.x, pS.y, pS.z))
        # ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))
    

        # Update modules (process inputs from other modules)
        terrain.Synchronize(time)
        my_hmmwv.Synchronize(time, driver_inputs, terrain)
        vis.Synchronize("", driver_inputs)
        
        vehicle_state = get_vehicle_state(my_hmmwv)
        t_stepsize.append(time)
        speed.append(speedPID.GetCurrentSpeed())
        speed_ref.append(target_speed)
        
        # Solve MPC every control_step
        if (step_number % (control_step) == 0) : 
            # print("step number", step_number)
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_ekin_mpc.plan(
                vehicle_state)
            u[0] = u[0] / vehicle_params.MASS  # Force to acceleration
            target_acc = u[0]
            target_speed = speedPID.GetCurrentSpeed() + target_acc*planner_ekin_mpc.config.DTK
            steering_output =veh.ChDriver(my_hmmwv.GetVehicle()).GetSteering() + u[1]*planner_ekin_mpc.config.DTK/correct_config.MAX_STEER
            # target_speed = speedPID.GetCurrentSpeed()
            u_acc.append(u[0])
            # u_steer_speed.append(u[1])
            t_controlperiod.append(time)
            # print("u", u)
            # print("mpc ref path x", mpc_ref_path_x) #list length of 16 (TK + 1)
            # print("mpc ref path y", mpc_ref_path_y)
            # print("mpc pred x", mpc_pred_x)
            # print("mpc pred y", mpc_pred_y)
            # print("mpc ox", mpc_ox)
            # print("mpc oy", mpc_oy)
            
            # Update mpc_path_asset with mpc_pred
            mpc_curve_points = [chrono.ChVectorD(mpc_ox[i], mpc_oy[i], 0.6) for i in range(MPC_params.TK + 1)]
            mpc_curve = chrono.ChBezierCurve(mpc_curve_points, False) # True = closed curve
            mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))

            # steeringPID = veh.ChPathSteeringController(mpc_curve) 
            # steeringPID.SetLookAheadDistance(5)
            # steeringPID.SetGains(0.8, 0, 0)
            # steeringPID.Reset(my_hmmwv.GetVehicle())

        # Advance simulation for one timestep for all modules
        speedPID_output = speedPID.Advance(my_hmmwv.GetVehicle(), target_speed, step_size)
        # steeringPID_output = steeringPID.Advance(my_hmmwv.GetVehicle(), step_size)
        terrain.Advance(step_size)
        my_hmmwv.Advance(step_size)
        vis.Advance(step_size)

        # Check if the vehicle has crossed the starting point
        pos = my_hmmwv.GetVehicle().GetPos()
        distance_to_start = (pos - starting_point).Length()

        if distance_to_start < tolerance:
            lap_counter += 1
            print(f"Completed lap {lap_counter}")

        if time > t_end:
            break

    # plt.figure()
    # plt.plot(t_stepsize, speed, label="speed")
    # plt.plot(t_stepsize, speed_ref, label="speed ref")
    # plt.xlabel("time [s]")
    # plt.ylabel(" longitudinal speed [m/s]")
    # plt.legend()
    # plt.savefig("longitudinal_speed.png")

    # plt.figure()
    # plt.plot(t_controlperiod, u_acc, label="acceleration") 
    # plt.xlabel("time [s]")
    # plt.ylabel("acceleration [m/s^2]")
    # plt.legend()
    # plt.savefig("acceleration.png")

    # plt.show()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(t_stepsize, speed, label="speed")
    ax[0].plot(t_stepsize, speed_ref, label="speed ref")
    ax[0].set_title("longitudinal speed")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel(" longitudinal speed [m/s]")
    ax[0].legend()
    ax[1].plot(t_controlperiod, u_acc, label="acceleration")
    ax[0].set_title("longitudinal acceleration")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("acceleration [m/s^2]")
    ax[1].legend()
    plt.savefig("longitudinal_speed_and_acceleration.png")
    plt.show()

    