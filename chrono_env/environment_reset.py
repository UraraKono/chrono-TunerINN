import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import numpy as np
# from .utils import *
from .utils import VehicleParameters, init_vehicle, init_terrain, init_irrlicht_vis, get_vehicle_state

class ChronoEnv:
    def __init__(self, step_size, throttle_value) -> None:
        self.step_size = step_size
        self.my_hmmwv = None
        self.terrain, self.vis_patch = None, None
        self.vis = None
        self.vehicle_params = None
        self.config = None

        # Time interval between two render frames
        self.render_step_size = 1.0 / 50  # FPS = 50 frame per second
        self.render_steps = math.ceil(self.render_step_size / self.step_size)

        self.step_number = 0
        self.time=0
        self.lap_counter = 0
        self.lap_flag = True
        self.tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

        self.u_acc = []
        # u_steer_speed = [] #steering speed from MPC is not used. ox/oy are used instead
        self.t_controlperiod = [] # time list every control_period
        self.t_stepsize = [] # time list every step_size
        self.speed = []
        self.speed_ref = []
        self.speedPID_output = 1.0
        self.target_speed = 0
        self.steering_output = 0
        self.target_steering_speed = 0

        self.driver_inputs = veh.DriverInputs()
        self.driver_inputs.m_throttle = throttle_value
        self.driver_inputs.m_braking = 0.0


    def make(self, config, friction, patch_coords, waypoints, curve, speedPID_Gain=[0.4,0,0], ini_pos = chrono.ChVectorD(0, 0, 0.5)) -> None:
        # self.my_hmmwv = utils.init_vehicle(self)
        # self.terrain, self.viz_patch = utils.init_terrain(self, friction, patch_coords, waypoints)
        # self.vis = utils.init_irrlicht_vis(self.my_hmmwv)
        # self.vehicle_params = utils.VehicleParameters(self.my_hmmwv)
        self.ini_pos = ini_pos
        self.my_hmmwv = init_vehicle(self)
        self.terrain, self.viz_patch = init_terrain(self, friction, patch_coords, waypoints)
        self.vis = init_irrlicht_vis(self.my_hmmwv)
        self.vehicle_params = VehicleParameters(self.my_hmmwv)
        self.config = config
        self.control_step = self.config.DTK / self.step_size # control step for MPC in sim steps

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

        # Set up the longitudinal speed PID controller
        self.Kp = speedPID_Gain[0]
        self.Ki = speedPID_Gain[1]
        self.Kd = speedPID_Gain[2]
        print("speedPID_Gain",speedPID_Gain)
        self.speedPID = veh.ChSpeedController()
        self.speedPID.SetGains(self.Kp, self.Ki, self.Kd)
        self.speedPID.Reset(self.my_hmmwv.GetVehicle())

    def reset(self, initial_state=np.zeros(7), throttle=0, brake=0) -> None:
        # vehicle_state = np.array([x,  # x
        #                       y,  # y
        #                       vx,  # vx
        #                       yaw_angle,  # yaw angle
        #                       vy,  # vy
        #                       yaw_rate,  # yaw rate
        #                       steering*max_steering_angle,  # steering angle
        #                     ])
        self.my_hmmwv.SetInitPosition(chrono.ChCoordsysD(
            chrono.ChVectorD(initial_state[0], initial_state[1], 0.5),chrono.QUNIT))
        self.driver_inputs.m_throttle = throttle
        self.driver_inputs.m_braking = brake
        self.driver_inputs.m_steering = initial_state[-1]/self.vehicle_params.MAX_STEER
        if throttle > 0:
            self.speedPID_output = throttle
        else:
            self.speedPID_output = -brake

        X = chrono.ChState()
        V = chrono.ChStateDelta()
        A = chrono.ChStateDelta()
        time = 0.0

        self.my_hmmwv.GetSystem()
        self.my_hmmwv.GetSystem().StateSetup(X, V, A)
        self.my_hmmwv.GetSystem().StateGather(X, V, np.double(time))

        self.my_hmmwv.GetSystem().StateScatter(X, V, veh.GetSystem().GetChTime(), False)
        self.my_hmmwv.GetSystem().Update(False)



    def step(self) -> None:
        # Increment frame number
        self.step_number += 1

        # Driver inputs
        self.time = self.my_hmmwv.GetSystem().GetChTime()
        self.driver_inputs.m_steering = np.clip(self.steering_output, -1.0, +1.0)
        self.speedPID_output = np.clip(self.speedPID_output, -1.0, +1.0)

        if self.speedPID_output > 0:
            self.driver_inputs.m_throttle = self.speedPID_output
            self.driver_inputs.m_braking = 0.0
        else:
            self.driver_inputs.m_throttle = 0.0
            self.driver_inputs.m_braking = -self.speedPID_output

        # Update modules (process inputs from other modules)
        self.terrain.Synchronize(self.time)
        self.my_hmmwv.Synchronize(self.time, self.driver_inputs, self.terrain)
        self.vis.Synchronize("", self.driver_inputs)
        
        vehicle_state = get_vehicle_state(self)
        # print("vehicle_state", vehicle_state)
        # vehicle_state = utils.get_vehicle_state(self)
        # vehicle_state[2] = speedPID.GetCurrentSpeed() # vx from get_vehicle_state is a bit different from speedPID.GetCurrentSpeed()
        self.t_stepsize.append(self.time)
        self.speed.append(vehicle_state[2])
        self.speed_ref.append(self.target_speed)

        # lap counter: Check if the vehicle has crossed the starting point
        pos = self.my_hmmwv.GetVehicle().GetPos()
        distance_to_start = (pos - self.ini_pos).Length()

        if distance_to_start < self.tolerance and self.lap_flag is False:
            self.lap_counter += 1
            self.lap_flag = True
            print(f"Completed lap {self.lap_counter} at time {self.time}")
        elif self.lap_flag is True and distance_to_start > self.tolerance:
            self.lap_flag = False


        
        # Solve MPC every control_step
        if (self.step_number % (self.control_step) == 0) : 
            # print("step number", self.step_number)
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = self.planner_ekin_mpc.plan(
                vehicle_state)
            u[0] = u[0] / self.vehicle_params.MASS  # Force to acceleration
            # print("u", u)
            self.target_speed = vehicle_state[2] + u[0]*self.planner_ekin_mpc.config.DTK
            self.steering_output = self.driver_inputs.m_steering + u[1]*self.planner_ekin_mpc.config.DTK/self.config.MAX_STEER # Overshoots soooo much
            # steering_output = u[1]*planner_ekin_mpc.config.DTK/self.config.MAX_STEER # This one works better lol. It doesn't make sesnse
            self.speedPID_output = self.speedPID.Advance(self.my_hmmwv.GetVehicle(), self.target_speed, self.step_size)
            # print('speed pid output', self.speedPID_output)
            self.u_acc.append(u[0])
            self.t_controlperiod.append(self.time)
            # print("vehicle_state.vx", vehicle_state[2],"speedPID.GetCurrentSpeed()", speedPID.GetCurrentSpeed())
            # print("mpc ref path x", mpc_ref_path_x) #list length of 16 (TK + 1)
            # print("mpc ref path y", mpc_ref_path_y)
            # print("mpc pred x", mpc_pred_x)
            # print("mpc pred y", mpc_pred_y)
            # print("mpc ox", mpc_ox)
            # print("mpc oy", mpc_oy)
            # print("target_speed", target_speed)
            
            if mpc_ox is not None and not np.any(np.isnan(mpc_ox)):
                # Update mpc_path_asset with mpc_pred
                mpc_curve_points = [chrono.ChVectorD(mpc_ox[i], mpc_oy[i], 0.6) for i in range(self.config.TK + 1)]
                mpc_curve = chrono.ChBezierCurve(mpc_curve_points, False) # True = closed curve
                self.mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))

        
        # Advance simulation for one timestep for all modules
        # These three lines should be outside of the if statement for MPC!!!
        self.terrain.Advance(self.step_size) 
        self.my_hmmwv.Advance(self.step_size)
        self.vis.Advance(self.step_size)

    def render(self) -> None:
        if (self.step_number % (self.render_steps) == 0) :
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        else:
            pass
