import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import numpy as np
from .utils import *
# from .utils import VehicleParameters, init_vehicle, init_terrain, init_irrlicht_vis, get_vehicle_state, get_steering, LongitudinalSpeedPIDController, SteeringAnglePIDController

class ChronoEnv:
    def __init__(self, step_size, throttle_value) -> None:
        self.step_size = step_size
        self.my_hmmwv = None
        self.terrain, self.viz_patch = None, None
        self.vis = None
        self.vehicle_params = None
        self.config = None
        self.mpc_ox = None
        self.reduced_rate = 1

        # Time interval between two render frames
        self.render_step_size = 1.0 / 50  # FPS = 50 frame per second
        self.render_steps = math.ceil(self.render_step_size / self.step_size)

        self.step_number = 0
        self.time=0
        self.lap_counter = 0
        self.lap_flag = True
        self.tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

        # self.u_acc = []
        # u_steer_speed = [] #steering speed from MPC is not used. ox/oy are used instead
        self.x_trajectory = []
        self.y_trajectory = []
        self.t_controlperiod = [] # time list every control_period
        self.t_stepsize = [] # time list every step_size
        self.speed = []
        self.speed_ref = []
        self.speedPID_output = 1.0
        self.steeringPID_output = 0
        
        self.toein_FL = []
        self.toein_FR = []
        self.toein_RL = []
        self.toein_RR = []
        self.steering_driver = []

        self.driver_inputs = veh.DriverInputs()
        self.driver_inputs.m_throttle = throttle_value
        self.driver_inputs.m_braking = 0.0


    def make(self, config, friction, reduced_waypoints, curve, speedPID_Gain=[1,0,0], steeringPID_Gain=[1,0,0], x0=0, y0=0, w0=0) -> None:
        self.x0 = x0 # initial x position
        self.y0 = y0 # initial y position
        self.w0 = w0 # initial yaw angle
        # self.my_hmmwv = init_vehicle(self)
        init_vehicle(self)
        self.my_hmmwv.state = get_vehicle_state(self)
        init_terrain(self, friction, reduced_waypoints)
        self.vis = init_irrlicht_vis(self.my_hmmwv)
        self.vehicle_params = VehicleParameters(self.my_hmmwv)
        self.config = config
        self.control_step = self.config.DTK / self.step_size # control step for MPC in sim steps

        path = curve
        print("path\n", path)
        npoints = path.getNumPoints()
        print("npoints", npoints)
        self.path_asset = chrono.ChLineShape()
        self.path_asset.SetLineGeometry(chrono.ChLineBezier(path))
        self.path_asset.SetName("test path")
        self.path_asset.SetColor(chrono.ChColor(1, 0.0, 0.0))
        self.path_asset.SetNumRenderPoints(max(2 * npoints, 400))
        print("self.viz_patch",self.viz_patch)
        self.viz_patch.GetGroundBody().AddVisualShape(self.path_asset)
        # Why doesn't this path_asset show up in the visualization?
        

        mpc_curve_points = [chrono.ChVectorD(i/10 + 0.1, i/10 + 0.1, 0.6) for i in range(self.config.TK)] #これはなにをやっているの？map情報からのwaypointガン無視してない？
        mpc_curve = chrono.ChBezierCurve(mpc_curve_points, True) # True = closed curve
        npoints = mpc_curve.getNumPoints()
        self.mpc_path_asset = chrono.ChLineShape()
        self.mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))
        self.mpc_path_asset.SetName("MPC path")
        self.mpc_path_asset.SetColor(chrono.ChColor(0.0, 0.0, 0.8))
        self.mpc_path_asset = chrono.ChLineShape()
        self.mpc_path_asset.SetNumRenderPoints(max(2 * npoints, 400))
        self.viz_patch.GetGroundBody().AddVisualShape(self.mpc_path_asset)

        # ballS = self.vis.GetSceneManager().addSphereSceneNode(0.1)
        self.ballT = self.vis.GetSceneManager().addSphereSceneNode(0.1)
        # ballS.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 255, 0, 0)
        self.ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)

        # Set up the longitudinal speed PID controller
        self.speedPID = LongitudinalSpeedPIDController(self.my_hmmwv)
        self.speedPID.SetGains(speedPID_Gain[0], speedPID_Gain[1], speedPID_Gain[2])

        # Set up the steering controller
        self.steeringPID = SteeringAnglePIDController(self.my_hmmwv)
        self.steeringPID.SetGains(steeringPID_Gain[0], steeringPID_Gain[1], steeringPID_Gain[2])


    # def reset(self) -> None:

    def step(self, target_speed, target_steering) -> None:
        # Driver inputs
        self.time = self.my_hmmwv.GetSystem().GetChTime()
        self.driver_inputs.m_steering = np.clip(self.steeringPID_output, -1.0, +1.0)
        # self.driver_inputs.m_steering = np.clip(target_steering/self.config.MAX_STEER, -1.0, +1.0)
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
        
        self.my_hmmwv.state = get_vehicle_state(self)
        # print("vehicle_state", self.my_hmmwv.state)
        self.t_stepsize.append(self.time)
        self.speed.append(self.my_hmmwv.state[2])
        self.speed_ref.append(target_speed)
        self.x_trajectory.append(self.my_hmmwv.state[0])
        self.y_trajectory.append(self.my_hmmwv.state[1])

        # lap counter: Check if the vehicle has crossed the starting point
        pos = self.my_hmmwv.GetVehicle().GetPos()
        distance_to_start = (pos - self.ini_pos).Length()

        if distance_to_start < self.tolerance and self.lap_flag is False:
            self.lap_counter += 1
            self.lap_flag = True
            print(f"Completed lap {self.lap_counter} at time {self.time}")
        elif self.lap_flag is True and distance_to_start > self.tolerance:
            self.lap_flag = False

        #Advance the PID controller every simulation time step
        self.speedPID_output = self.speedPID.Advance(target_speed, self.step_size)
        self.steeringPID_output = self.steeringPID.Advance(target_steering, self.step_size)

        # every control_step
        if (self.step_number % (self.control_step) == 0) : 
            # self.speedPID_output = self.speedPID.Advance(target_speed, self.step_size)
            # self.steeringPID_output = self.steeringPID.Advance(target_steering, self.step_size)
            # self.t_controlperiod.append(self.time)
            
            if self.mpc_ox is not None and not np.any(np.isnan(self.mpc_ox)):
                # print("Update self.mpc_path_asset")
                mpc_curve_points = [chrono.ChVectorD(self.mpc_ox[i], self.mpc_oy[i], 0.6) for i in range(len(self.mpc_ox))]
                mpc_curve = chrono.ChBezierCurve(mpc_curve_points, False) # True = closed curve
                self.mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))

                pT = chrono.ChVectorD(self.mpc_ox[-1], self.mpc_oy[-1], 0.6)
                self.ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))
            elif self.mpc_ox is not None and np.any(np.isnan(self.mpc_ox)):
                print("No update self.mpc_path_asset")
            else:
                print("self.mpc_ox is None")
        
        # Advance simulation for one timestep for all modules
        # These three lines should be outside of the if statement for MPC!!!
        self.terrain.Advance(self.step_size) 
        self.my_hmmwv.Advance(self.step_size)
        self.vis.Advance(self.step_size)

        # Increment frame number
        self.step_number += 1

        steering_FL = get_steering(self,0,0)
        steering_FR = get_steering(self,0,1)
        steering_RL = get_steering(self,1,0)
        steering_RR = get_steering(self,1,1)

        self.toein_FL.append(steering_FL*180/np.pi)
        self.toein_FR.append(steering_FR*180/np.pi)
        self.toein_RL.append(steering_RL*180/np.pi)
        self.toein_RR.append(steering_RR*180/np.pi)
        self.steering_driver.append(self.driver_inputs.m_steering*self.config.MAX_STEER*180/np.pi)

        
    def render(self) -> None:
        if (self.step_number % (self.render_steps) == 0) :
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        else:
            pass