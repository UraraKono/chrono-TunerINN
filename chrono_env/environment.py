import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import warnings
from .utils import *
# from .utils import VehicleParameters, init_vehicle, init_terrain, init_irrlicht_vis, get_vehicle_state, get_steering, LongitudinalSpeedPIDController, SteeringAnglePIDController

class ChronoEnv:
    """
    OpenAI gym-like environment for PyChrono.
    Env should be initialized by calling ChronoEnv.make(**kwargs).
    Args:
        kwargs:
            step_size: simulation step size
            render_step_size: time interval between two render frames, FPS
            tolerance: tolerance distance (in meters) to consider the vehicle has crossed the starting point
            throttle_value: throttle value for the vehicle to start. This shouldn't be set zero; otherwise it doesn't start
            control_period: control period for your own controller like MPC
            waypoints: waypoints for the vehicle to follow. 
            sample_rate_waypoints: sampling rate of waypoints for the visualization, patch_waypoints. 
                            If it's too dense, there will be too many patches, heavy to visualize.
            friction: friction coefficient for each patch of the terrain.
                    The length of friction should be the same as that of reduced_waypoints.
            speedPID_Gain: [Kp, Ki, Kd] for the longitudinal speed PID controller
            steeringPID_Gain: [Kp, Ki, Kd] for the steering angle PID controller
            x0: initial x position of the vehicle
            y0: initial y position of the vehicle
            w0: initial yaw angle of the vehicle
    """
    def __init__(self,**kwargs) -> None:
        try:
            # simulation step size
            self.step_size = kwargs['step_size']
        except:
            self.step_size = 2e-3

        self.my_hmmwv = None
        self.terrain, self.viz_patch = None, None
        self.vis = None
        self.vehicle_params = None
        self.mpc_ox = None

        # Time interval between two render frames
        try:
            self.render_step_size = kwargs['render_step_size']
        except:
            self.render_step_size = 1.0 / 50  # FPS = 50 frame per second
        self.render_steps = math.ceil(self.render_step_size / self.step_size)

        self.step_number = 0
        self.time=0
        self.lap_counter = 0
        self.lap_flag = True
        try:
            self.tolerance = kwargs['tolerance']
        except:
            self.tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point

        self.x_trajectory, self.y_trajectory = [], []
        self.t_controlperiod = [] # time list every control_period
        self.t_stepsize = [] # time list every step_size
        self.speed, self.speed_ref = [], []
        self.speedPID_output = 1.0
        self.steeringPID_output = 0

        self.toein_FL, self.toein_FR, self.toein_RL, self.toein_RR = [], [], [], []
        self.steering_driver = []

        self.driver_inputs = veh.DriverInputs()
        self.driver_inputs.m_braking = 0.0
        try:
            # This shouldn't be set zero; otherwise it doesn't start
            self.driver_inputs.m_throttle = kwargs['throttle_value']
        except:
            self.driver_inputs.m_throttle = 0.3 


    def make(self, **kwargs) -> None:
        try:
            # simulation step size
            self.step_size = kwargs['timestep']
        except:
            self.step_size = 2e-3
        try:
            self.render_step_size = kwargs['render_step_size']
        except:
            self.render_step_size = 1.0 / 50  # FPS = 50 frame per second
        try:
            self.tolerance = kwargs['tolerance']
        except:
            self.tolerance = 5  # Tolerance distance (in meters) to consider the vehicle has crossed the starting point
        try:
            # This shouldn't be set zero; otherwise it doesn't start
            self.driver_inputs.m_throttle = kwargs['throttle_value']
        except:
            self.driver_inputs.m_throttle = 0
        try:
            self.control_period = kwargs['control_period']
        except:
            self.control_period = 0.1 
        try:
            waypoints = kwargs['waypoints']
        except:
            waypoints = None
        try:
            self.sample_rate_waypoints = kwargs['sample_rate_waypoints']
        except:
            self.sample_rate_waypoints = 1
        try:
            friction = kwargs['friction']
        except:
            friction = None
        try:
            speedPID_Gain = kwargs['speedPID_Gain']
        except:
            speedPID_Gain = [5,0,0]
        try:
            steeringPID_Gain = kwargs['steeringPID_Gain']
        except:
            steeringPID_Gain = [0.5,0,0]
        try:
            self.x0 = kwargs['x0']
        except:
            self.x0 = waypoints[0,1]
        try:
            self.y0 = kwargs['y0']
        except:
            self.y0 = waypoints[0,2]
        try:
            self.w0 = kwargs['w0']
        except:
            self.w0 = waypoints[0,3]-np.pi
        try:
            self.patch_scale = kwargs['patch_scale']
        except:
            self.patch_scale = 5
        try:
            self.constant_friction = kwargs['constant_friction']
        except:
            self.constant_friction = False
        try:
            self.visualize = kwargs['visualize']
        except:
            self.visualize = True
        try:
            self.model = kwargs['model']
        except:
            self.model = 'MB'
        # try:
        #     self.config = kwargs['config']
        # except:
        #     self.config = None
        if 'config' in kwargs:
            self.config = kwargs['config']
        if 'map' in kwargs:
            warnings.warn("'map' is not supported. Use 'waypoints' instead.", stacklevel=2)
            # What stacklevel should be used? 
        if 'map_ext' in kwargs:
            warnings.warn("'map_ext' is not supported. Use 'waypoints' instead.", stacklevel=2)
        if 'num_agents' in kwargs:
            warnings.warn("Only one agent is supported. Ignoring 'num_agents'.", stacklevel=2)
        if 'model' in kwargs:
            warnings.warn("'model' argument is not supported and ignored.", stacklevel=2)
        if 'drive_control_mode' in kwargs:
            warnings.warn("'drive_control_mode' is not supported and ignored.", stacklevel=2)
        if 'steering_control_mode' in kwargs:
            warnings.warn("'steering_control_mode' is not supported and ignored.", stacklevel=2)
        
# nv = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                            # num_agents=1, timestep=0.01, model='MB', drive_control_mode='acc',
                            # steering_control_mode='vel')
        if not self.constant_friction:
            patch_waypoints = waypoints[::self.sample_rate_waypoints, :]
            print("patch_waypoints", patch_waypoints.shape)
            print("waypoints", waypoints.shape)
        else:
            patch_waypoints = None
        veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')
        init_vehicle(self)
        self.vehicle_params = VehicleParameters(self.my_hmmwv)
        self.my_hmmwv.state = get_vehicle_state(self)
        init_terrain(self, friction, patch_waypoints)
        if self.visualize:
            self.vis = init_irrlicht_vis(self.my_hmmwv)

        self.control_step = self.control_period / self.step_size # control step for MPC in sim steps

        if (self.constant_friction == False) and (self.visualize == True):
            curve_points = [chrono.ChVectorD(waypoint[1], waypoint[2], 0.8) for waypoint in waypoints]
            curve = chrono.ChBezierCurve(curve_points, True) # True = closed curve

            path = curve
            # print("path\n", path)
            npoints = path.getNumPoints()
            # print("npoints", npoints)
            self.path_asset = chrono.ChLineShape()
            self.path_asset.SetLineGeometry(chrono.ChLineBezier(path))
            self.path_asset.SetName("test path")
            self.path_asset.SetColor(chrono.ChColor(1, 0.0, 0.0))
            self.path_asset.SetNumRenderPoints(max(2 * npoints, 400))
            # print("self.viz_patch",self.viz_patch)
            self.viz_patch.GetGroundBody().AddVisualShape(self.path_asset)
            # Why doesn't this path_asset show up in the visualization?
            
            mpc_curve_points = [chrono.ChVectorD(i/10 + 0.1, i/10 + 0.1, 0.6) for i in range(30)] #これはなにをやっているの？map情報からのwaypointガン無視してない？
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

        self.my_hmmwv.GetVehicle().EnableRealtime(True)
        # Reset the simulation time
        self.my_hmmwv.GetSystem().SetChTime(0) 

        return self

    # def reset(self) -> None:

    def step(self, target_steering, target_speed) -> None:
        # Driver inputs
        self.time = self.my_hmmwv.GetSystem().GetChTime()
        self.speedPID_output = np.clip(self.speedPID_output, -1.0, +1.0)
        # self.driver_inputs.m_steering = np.clip(self.steeringPID_output, -1.0, +1.0)
        self.driver_inputs.m_steering = np.clip(target_steering/self.vehicle_params.MAX_STEER, -1.0, +1.0)

        if self.speedPID_output > 0:
            self.driver_inputs.m_throttle = self.speedPID_output
            self.driver_inputs.m_braking = 0.0
        else:
            self.driver_inputs.m_throttle = 0.0
            self.driver_inputs.m_braking = -self.speedPID_output

        # Update modules (process inputs from other modules)
        self.terrain.Synchronize(self.time)
        self.my_hmmwv.Synchronize(self.time, self.driver_inputs, self.terrain)
        if self.visualize:
            self.vis.Synchronize("", self.driver_inputs)
        
        self.my_hmmwv.state = get_vehicle_state(self)
        # print("self.my_hmmwv.state[-1]", self.my_hmmwv.state[-1])
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

        #Advance the PID controller every control_step
        # if (self.step_number % (self.control_step) == 0) : 
        #     self.speedPID_output = self.speedPID.Advance(target_speed, self.control_period)
        #     self.steeringPID_output = self.steeringPID.Advance(target_steering, self.control_period)
        #     # self.t_controlperiod.append(self.time)
            
        #     if self.mpc_ox is not None and not np.any(np.isnan(self.mpc_ox)):
        #         # print("Update self.mpc_path_asset")
        #         mpc_curve_points = [chrono.ChVectorD(self.mpc_ox[i], self.mpc_oy[i], 0.6) for i in range(len(self.mpc_ox))]
        #         mpc_curve = chrono.ChBezierCurve(mpc_curve_points, False) # True = closed curve
        #         self.mpc_path_asset.SetLineGeometry(chrono.ChLineBezier(mpc_curve))

        #         pT = chrono.ChVectorD(self.mpc_ox[-1], self.mpc_oy[-1], 0.6)
        #         self.ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))
        #     elif self.mpc_ox is not None and np.any(np.isnan(self.mpc_ox)):
        #         print("No update self.mpc_path_asset")
            # else:
            #     print("self.mpc_ox is None")
        
        # Advance simulation for one timestep for all modules
        # These three lines should be outside of the if statement for MPC!!!
        self.terrain.Advance(self.step_size) 
        self.my_hmmwv.Advance(self.step_size)
        if self.visualize:
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
        self.steering_driver.append(self.driver_inputs.m_steering*self.vehicle_params.MAX_STEER*180/np.pi)

        if self.model == 'ST':
            observation = {'poses_x':self.my_hmmwv.state[0], 'poses_y':self.my_hmmwv.state[1],
                        'steering':self.my_hmmwv.state[2], 'v':self.my_hmmwv.state[3],
                        'yaw_angle':self.my_hmmwv.state[4], 'yaw_rate':self.my_hmmwv.state[5], 
                        'beta':self.my_hmmwv.state[6], 'state':self.my_hmmwv.state}
        else:
            observation = {'poses_x': self.my_hmmwv.state[0],'poses_y': self.my_hmmwv.state[1],
                        'vx':self.my_hmmwv.state[2], 'poses_theta': self.my_hmmwv.state[3],
                        'vy':self.my_hmmwv.state[4],'yaw_rate':self.my_hmmwv.state[5],
                        'steering':self.my_hmmwv.state[6], 'state':self.my_hmmwv.state}
        reward = 0
        done = False
        info = {}

            # vehicle_state = np.array([x,  # x
            #                   y,  # y
            #                   vx,  # vx
            #                   yaw_angle,  # yaw angle
            #                   vy,  # vy
            #                   yaw_rate,  # yaw rate
            #                   steering  # steering angle
            #                 ])

        return observation, reward, done, info
        
    def render(self) -> None:
        if (self.step_number % (self.render_steps) == 0) :
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        else:
            pass