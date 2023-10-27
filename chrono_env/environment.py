import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import time
import numpy as np
import json
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
import time
import yaml
from argparse import Namespace
from . import utils
# from regulators.pure_pursuit import *
# from regulators.path_follow_mpc import *
# from models.extended_kinematic import ExtendedKinematicModel
# from models.GP_model_single import GPSingleModel
# from models.configs import *
# from helpers.closest_point import *
# from helpers.track import Track

class ChronoEnv:
    def __init__(self, step_size, throttle_value) -> None:
        self.step_size = step_size
        self.my_hmmwv = None
        self.terrain, self.vis_patch = None, None
        self.vis = None
        self.vehicle_params = None
        self.config = None
        self.lap_counter = 0
        # # Time interval between two render frames
        # self.render_step_size = 1.0 / 50  # FPS = 50 frame per second
        # self.step_number = 0

        # self.u_acc = []
        # # u_steer_speed = [] #steering speed from MPC is not used. ox/oy are used instead
        # self.t_controlperiod = [] # time list every control_period
        # self.t_stepsize = [] # time list every step_size
        # self.speed = []
        # self.speed_ref = []
        # self.speedPID_output = 0
        # self.target_speed = 0
        # self.target_acc = 0
        # self.steering_output = 0
        # self.target_steering_speed = 0

        # self.driver_inputs = veh.DriverInputs()
        # self.driver_inputs.m_throttle = throttle_value
        # self.driver_inputs.m_braking = 0.0

    def make(self, config, friction, patch_coords, waypoints, curve) -> None:
        self.my_hmmwv = utils.init_vehicle(self)
        self.terrain, self.viz_patch = utils.init_terrain(self, friction, patch_coords, waypoints)
        self.vis = utils.init_irrlicht_vis(self.my_hmmwv)
        self.vehicle_params = utils.VehicleParameters(self.my_hmmwv)
        self.config = config

        path = curve
        # print("path\n", path)
        npoints = path.getNumPoints()
        path_asset = chrono.ChLineShape()
        path_asset.SetLineGeometry(chrono.ChLineBezier(path))
        path_asset.SetName("test path")
        path_asset.SetColor(chrono.ChColor(0.8, 0.0, 0.0))
        path_asset.SetNumRenderPoints(max(2 * npoints, 400))
        self.viz_patch.GetGroundBody().AddVisualShape(path_asset)

        mpc_curve_points = [chrono.ChVectorD(i/10 + 0.1, i/10 + 0.1, 0.6) for i in range(self.config.TK + 1)] #これはなにをやっているの？map情報からのwaypointガン無視してない？
        mpc_curve = chrono.ChBezierCurve(mpc_curve_points, True) # True = closed curve
        npoints = mpc_curve.getNumPoints()
        mpc_path_asset = chrono.ChLineShape()
        mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))
        mpc_path_asset.SetName("MPC path")
        mpc_path_asset.SetColor(chrono.ChColor(0.0, 0.0, 0.8))
        mpc_path_asset.SetNumRenderPoints(max(2 * npoints, 400))
        self.viz_patch.GetGroundBody().AddVisualShape(mpc_path_asset)
        self.mpc_path_asset = mpc_path_asset

        ballS = self.vis.GetSceneManager().addSphereSceneNode(0.1)
        ballT = self.vis.GetSceneManager().addSphereSceneNode(0.1)
        ballS.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 255, 0, 0)
        ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)

    # def reset(self) -> None:

    def step(self) -> None:
        # Increment frame number
        # self.step_number += 1

        # # Driver inputs
        # time = self.my_hmmwv.GetSystem().GetChTime()
        # self.driver_inputs.m_steering = np.clip(self.steering_output, -1.0, +1.0)

        # # Update modules (process inputs from other modules)
        # self.terrain.Synchronize(time)
        # self.my_hmmwv.Synchronize(time, self.driver_inputs, self.terrain)
        # self.vis.Synchronize("", self.driver_inputs)
        
        # vehicle_state = utils.get_vehicle_state(self)
        # vehicle_state[2] = speedPID.GetCurrentSpeed() # vx from get_vehicle_state is a bit different from speedPID.GetCurrentSpeed()
        # self.t_stepsize.append(time)
        # self.speed.append(self.speedPID.GetCurrentSpeed())
        # self.speed_ref.append(self.target_speed)


        self.terrain.Advance(self.step_size)
        self.my_hmmwv.Advance(self.step_size)
        self.vis.Advance(self.step_size)

    def render(self) -> None:
        pass
