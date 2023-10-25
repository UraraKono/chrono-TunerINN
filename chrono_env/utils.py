import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import time
import matplotlib.pyplot as plt


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

    my_hmmwv.SetChassisVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetSuspensionVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetSteeringVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetWheelVisualizationType(veh.VisualizationType_MESH)
    my_hmmwv.SetTireVisualizationType(veh.VisualizationType_MESH)
    
    return my_hmmwv

def init_terrain(self, friction, patch_coords, waypoints):
    friction = self.friction
    patch_coords = self.patch_coords
    waypoints = self.waypoints
    
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
