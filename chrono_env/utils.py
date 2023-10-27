import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

def init_vehicle(self):
    # Create the vehicle system
    my_hmmwv = veh.HMMWV_Full()
    my_hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    my_hmmwv.SetChassisFixed(False)
    my_hmmwv.SetInitPosition(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0.5),chrono.QUNIT))
    my_hmmwv.SetPowertrainType(veh.PowertrainModelType_SHAFTS)
    my_hmmwv.SetDriveType(veh.DrivelineTypeWV_RWD)
    my_hmmwv.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
    my_hmmwv.SetTireType(veh.TireModelType_TMEASY)
    my_hmmwv.SetTireStepSize(self.step_size) # self.step_size
    my_hmmwv.Initialize()

    my_hmmwv.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetWheelVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetTireVisualizationType(veh.VisualizationType_PRIMITIVES)
    
    self.my_hmmwv = my_hmmwv
    return my_hmmwv

def init_terrain(self, friction, patch_coords, waypoints):
    rest_values = [0.01] * len(patch_coords)
    young_modulus_values = [2e7] * len(patch_coords)
    patch_mats = [chrono.ChMaterialSurfaceSMC() for _ in range(len(patch_coords))]
    for i, patch_mat in enumerate(patch_mats):
        patch_mat.SetFriction(friction[i])
        patch_mat.SetRestitution(rest_values[i])
        patch_mat.SetYoungModulus(young_modulus_values[i])

    terrain = veh.RigidTerrain(self.my_hmmwv.GetSystem()) # self.my_hmmwv

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

def get_vehicle_state(self):
    vehicle = self.my_hmmwv
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
    tf_FL = vehicle.GetVehicle().GetTire(0, veh.LEFT).ReportTireForce(self.terrain)
    tf_FR = vehicle.GetVehicle().GetTire(0, veh.RIGHT).ReportTireForce(self.terrain)
    tf_RL = vehicle.GetVehicle().GetTire(1, veh.LEFT).ReportTireForce(self.terrain)
    tf_RR = vehicle.GetVehicle().GetTire(1, veh.RIGHT).ReportTireForce(self.terrain)
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
    my_driver = veh.ChDriver(vehicle.GetVehicle()) #This command does NOT work. Never use ChDriver!
    # throttle = my_driver.GetThrottle()
    # steering = my_driver.GetSteering() # steering input [-1,+1]
    # braking = my_driver.GetBraking()

    steering = veh.DriverInputs().m_steering
    # inputs = my_driver.GetInputs()

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


class VehicleParameters:
    def __init__(self, vehicle):
        self.LENGTH: float = 4.298  # Length of the vehicle [m]
        # WIDTH: float = 1.674  # Width of the vehicle [m]
        # LR: float = 1.50876
        # LF: float = 0.88392
        # WB: float = 0.88392 + 1.50876  # Wheelbase [m]
        # MIN_STEER: float = -0.4189  # maximum steering angle [rad]
        # MAX_STEER: float = 0.4189  # maximum steering angle [rad]
        self.MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
        self.MIN_STEER_V: float = 3.2  # maximum steering speed [rad/s]
        self.MAX_SPEED: float = 45.0  # maximum speed [m/s]
        self.MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
        self.MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
        self.MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]
        self.MASS = vehicle.GetVehicle().GetMass()
        self.WB   = vehicle.GetVehicle().GetWheelbase()
        self.MIN_STEER = -vehicle.GetVehicle().GetMaxSteeringAngle()
        self.MAX_STEER = +vehicle.GetVehicle().GetMaxSteeringAngle()
        self.WIDTH = vehicle.GetVehicle().GetWheeltrack(0)
        chassisPos = vehicle.GetVehicle().GetChassis().GetPos()
        COMPos = vehicle.GetVehicle().GetChassis().GetCOMFrame().coord.pos
        absPosCOM = COMPos + chassisPos
        fw = vehicle.GetVehicle().GetAxle(0).GetWheels()[1].GetPos()
        tmp = fw - absPosCOM
        self.LF = np.linalg.norm(np.array([tmp.x, tmp.y, tmp.z]))
        self.LR = self.WB - self.LF


# def get_vehicle_parameters(vehicle):
#     params = VehicleParameters()
#     params.MASS = vehicle.GetVehicle().GetMass()
#     params.WB   = vehicle.GetVehicle().GetWheelbase()
#     params.MIN_STEER = -vehicle.GetVehicle().GetMaxSteeringAngle()
#     params.MAX_STEER = +vehicle.GetVehicle().GetMaxSteeringAngle()
#     params.WIDTH = vehicle.GetVehicle().GetWheeltrack(0)
#     chassisPos = vehicle.GetVehicle().GetChassis().GetPos()
#     COMPos = vehicle.GetVehicle().GetChassis().GetCOMFrame().coord.pos
#     absPosCOM = COMPos + chassisPos
#     fw = vehicle.GetVehicle().GetAxle(0).GetWheels()[1].GetPos()
#     tmp = fw - absPosCOM
#     params.LF = np.linalg.norm(np.array([tmp.x, tmp.y, tmp.z]))
    
#     params.LR = params.WB - params.LF 

#     return params       
