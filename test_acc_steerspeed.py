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
# Authors: Radu Serban
# =============================================================================
#
# Demonstration of a steering path-follower and cruise control PID controlers. 
#
# The vehicle reference frame has Z up, X towards the front of the vehicle, and
# Y pointing to the left.
# 
# Test pid controller by commanding acceleration and steering speed
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as irr
import math as m
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Interactive plot
plot_ion = False

# Assume these are the constant returns from the MPC solved just once and 
target_acc = 0.5 # [m/s**2]
target_steer_speed = 0.1 # [rad/s] 5.73 deg/s

def main():
    #print("Copyright (c) 2017 projectchrono.org\nChrono version: ", CHRONO_VERSION , "\n\n")

    #  Create the HMMWV vehicle, set parameters, and initialize
    # my_hmmwv = veh.HMMWV_Full()
    # my_hmmwv.SetContactMethod(contact_method)
    # my_hmmwv.SetChassisFixed(False) 
    # my_hmmwv.SetInitPosition(chrono.ChCoordsysD(initLoc, chrono.ChQuaternionD(1, 0, 0, 0)))
    # my_hmmwv.SetPowertrainType(powertrain_model)
    # my_hmmwv.SetDriveType(drive_type)
    # my_hmmwv.SetSteeringType(steering_type)
    # my_hmmwv.SetTireType(tire_model)
    # my_hmmwv.SetTireStepSize(tire_step_size)
    # my_hmmwv.Initialize()

    # my_hmmwv.SetChassisVisualizationType(chassis_vis_type)
    # my_hmmwv.SetSuspensionVisualizationType(suspension_vis_type)
    # my_hmmwv.SetSteeringVisualizationType(steering_vis_type)
    # my_hmmwv.SetWheelVisualizationType(wheel_vis_type)
    # my_hmmwv.SetTireVisualizationType(tire_vis_type)

    my_hmmwv = veh.HMMWV_Full()
    my_hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    my_hmmwv.SetChassisFixed(False)
    my_hmmwv.SetInitPosition(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0.5),chrono.QUNIT))
    my_hmmwv.SetPowertrainType(veh.PowertrainModelType_SHAFTS)
    my_hmmwv.SetDriveType(veh.DrivelineTypeWV_RWD)
    my_hmmwv.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
    my_hmmwv.SetTireType(veh.TireModelType_TMEASY)
    my_hmmwv.SetTireStepSize(step_size)
    my_hmmwv.Initialize()

    my_hmmwv.SetChassisVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetSuspensionVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetSteeringVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetWheelVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetTireVisualizationType(veh.VisualizationType_MESH)

    # my_vehicle = my_hmmwv.GetVehicle()
    # print(my_vehicle.GetMaxSteeringAngle())
    my_hmmwv_ChWheeledVehicle = my_hmmwv.GetVehicle()
    max_steer = my_hmmwv_ChWheeledVehicle.GetMaxSteeringAngle()
    print('Max Steering', max_steer) #0.5276130328778859 rad = about 30 degrees
    # print('GetSteerings',my_hmmwv_ChWheeledVehicle.GetSteerings())
    # print('GetSteering',my_hmmwv_ChWheeledVehicle.GetSteering(0).GetSteeringLink())

    # Create the terrain

    terrain = veh.RigidTerrain(my_hmmwv.GetSystem())
    if (contact_method == chrono.ChContactMethod_NSC):
        patch_mat = chrono.ChMaterialSurfaceNSC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
    elif (contact_method == chrono.ChContactMethod_SMC):
        patch_mat = chrono.ChMaterialSurfaceSMC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
        patch_mat.SetYoungModulus(2e7)
    patch = terrain.AddPatch(patch_mat, 
                             chrono.CSYSNORM, 
                             300, 50)
    patch.SetTexture(veh.GetDataFile("terrain/textures/tile4.jpg"), 200, 200)
    patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
    terrain.Initialize()

    # # Create the path-follower, cruise-control driver
    # # Use a parameterized ISO double lane change (to left)
    # path = veh.DoubleLaneChangePath(initLoc, 13.5, 4.0, 11.0, 50.0, True)
    # driver = veh.ChPathFollowerDriver(my_hmmwv.GetVehicle(), path, "my_path", target_speed)
    # driver.GetSteeringController().SetLookAheadDistance(5)
    # driver.GetSteeringController().SetGains(0.5, 0, 0) #0.8, 0, 0
    # driver.GetSpeedController().SetGains(0.4, 0, 0)
    # driver.Initialize()

    speedPID = veh.ChSpeedController()
    Kp = 0.6
    Ki = 0.2
    Kd = 0.3
    speedPID.SetGains(Kp, Ki, Kd)
    speedPID.Reset(my_hmmwv.GetVehicle())

    # Create the vehicle Irrlicht interface
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('HMMWV speed controller')
    vis.SetWindowSize(1280, 1024)
    vis.SetChaseCamera(chrono.ChVectorD(0.0, 0.0, 1.75), 6.0, 0.5)
    vis.Initialize()
    vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    vis.AddLightDirectional()
    vis.AddSkyBox()
    vis.AttachVehicle(my_hmmwv.GetVehicle())

   	# Visualization of controller points (sentinel & target)
    ballS = vis.GetSceneManager().addSphereSceneNode(0.1);
    ballT = vis.GetSceneManager().addSphereSceneNode(0.1);
    ballS.getMaterial(0).EmissiveColor = irr.SColor(0, 255, 0, 0);
    ballT.getMaterial(0).EmissiveColor = irr.SColor(0, 0, 255, 0);

    speedPID_output = 0
    steering_output = 0

    target_speed = 0

    # Simulation loop
    my_hmmwv.GetVehicle().EnableRealtime(True)

    # Plot the speed and steering with matplotlib
    if plot_ion:
        plt.ion()
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('Speed')
    # ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Speed [m/s]')
    ax[1].set_title('Steering')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Steering [degrees]')
    t = []
    speed = []
    speed_ref = []
    steering = []

    while vis.Run() :
        target_speed += target_acc*step_size
        steering_output += target_steer_speed/(max_steer)*step_size #open loop

        time = my_hmmwv.GetSystem().GetChTime()

        t.append(time)
        speed.append(speedPID.GetCurrentSpeed())
        speed_ref.append(target_speed)
        steering.append(steering_output*max_steer*180/np.pi) # [-1,1]*max_steer*180/pi

        if plot_ion:
            # Update the plot
            ax[0].plot(t, speed, 'r')
            ax[0].plot(t, speed_ref, 'k')
            ax[1].plot(t, steering, 'r')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.show(block=False) #plt.show(block=True) to stop the simulation


        # End simulation
        if (time >= t_end):
            break

        # # Update sentinel and target location markers for the path-follower controller.
        # pS = driver.GetSteeringController().GetSentinelLocation()
        # pT = driver.GetSteeringController().GetTargetLocation()
        # ballS.setPosition(irr.vector3df(pS.x, pS.y, pS.z))
        # ballT.setPosition(irr.vector3df(pT.x, pT.y, pT.z))

        # # Draw scene
        # vis.BeginScene()
        # vis.Render()
        # vis.EndScene()

        # Get driver inputs
        # driver_inputs = driver.GetInputs()
        # Driver inputs
        driver_inputs = veh.DriverInputs()
        driver_inputs.m_steering = np.clip(steering_output, -1.0, +1.0)
        driver_inputs.m_throttle = np.clip(speedPID_output, -1.0, +1.0)
        driver_inputs.m_braking = 0.0

        # Update modules (process inputs from other modules)
        # driver.Synchronize(time)
        terrain.Synchronize(time)
        my_hmmwv.Synchronize(time, driver_inputs, terrain)
        vis.Synchronize("", driver_inputs)

        # Advance simulation for one timestep for all modules
        # driver.Advance(step_size)
        speedPID_output = speedPID.Advance(my_hmmwv.GetVehicle(), target_speed, step_size)
        terrain.Advance(step_size)
        my_hmmwv.Advance(step_size)
        vis.Advance(step_size)

        # print('current speed', speedPID.GetCurrentSpeed(), 'speed command', target_speed)
        # print('', my_hmmwv.GetSte)

    if not plot_ion:
        ax[0].plot(t, speed, 'r')
        ax[0].plot(t, speed_ref, 'k')
        ax[1].plot(t, steering, 'r')
        plt.savefig('./PID_tuning/P'+str(Kp)+'I'+str(Ki)+'D'+str(Kd)+'.png')
        plt.show()

    return 0


# The path to the Chrono data directory containing various assets (meshes, textures, data files)
# is automatically set, relative to the default location of this demo.
# If running from a different directory, you must change the path to the data directory with: 
#chrono.SetChronoDataPath('path/to/data')
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

# Initial vehicle location
initLoc = chrono.ChVectorD(-50, 0, 0.7)

# # Vehicle target speed (cruise-control)
# target_speed = 12

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = veh.VisualizationType_NONE
suspension_vis_type =  veh.VisualizationType_PRIMITIVES
steering_vis_type = veh.VisualizationType_PRIMITIVES
wheel_vis_type = veh.VisualizationType_MESH
tire_vis_type = veh.VisualizationType_MESH 

# Type of powertrain model (SHAFTS, SIMPLE)
powertrain_model = veh.PowertrainModelType_SHAFTS

# Drive type (FWD, RWD, or AWD)
drive_type = veh.DrivelineTypeWV_AWD

# Steering type (PITMAN_ARM or PITMAN_ARM_SHAFTS)
steering_type = veh.SteeringTypeWV_PITMAN_ARM

# Type of tire model (RIGID, RIGID_MESH, FIALA, PAC89)
tire_model = veh.TireModelType_TMEASY

# Contact method
contact_method = chrono.ChContactMethod_SMC

# Simulation step sizes
step_size = 2e-3
tire_step_size = 1e-3

# Simulation end time
t_end = 15 #20


main()
