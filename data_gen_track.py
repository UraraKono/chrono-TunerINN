# import time
# import yaml
# # import gym
# import numpy as np
# from argparse import Namespace
import os, sys
# from planner import PurePursuitPlanner, get_render_callback, pid
# from utils.mb_model_params import param1
# import matplotlib.pyplot as plt
# from utils.frenet_utils import cartesian_to_frenet, frenet_to_cartesian, centerline_to_frenet

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from argparse import Namespace
import warnings
from EGP.regulators.pure_pursuit import *
from EGP.regulators.path_follow_mpc import *
from EGP.models.extended_kinematic import ExtendedKinematicModel
from EGP.models.configs import *
from EGP.helpers.closest_point import *
from EGP.helpers.track import Track
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *
from utilities import load_map, friction_func, centerline_to_frenet, cartesian_to_frenet, frenet_to_cartesian

SEGMENT_LENGTH = 10
RENDER = True
SAVE_DIR = '/home/lucerna/Documents/DATA/tuner_inn/track39/'
MAP_DIR = './f1tenth_racetracks/'
ACC_VS_CONTROL = False
VEL_SAMPLE_UP = 0.3
SAVE_STEP = 210

# --------------
step_size = 2e-3 #simulation step size
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
SAVE_MODEL = True
MAP_DIR = './f1tenth-racetrack/'
SAVE_DIR = './data/'
t_end = 2000
map_ind = 17 # 39 Zandvoort_raceline
map_scale = 10
map_reverse = False
patch_scale = 1.5 # map_ind 16: 1.5
# ref_vx = 8.0
control_model = "pure_pursuit" # options: ext_kinematic, pure_pursuit
num_laps = 1  # Number of laps
# --------------


if len(sys.argv) > 1:
    start_vel = float(sys.argv[1])
    # vels = [vel]
    vels = np.arange(start_vel, start_vel + 0.3, 0.1)
else:
    print('Please provide a starting velocity. start_vel = 8.0 automatically used.')
    start_vel = 8.0
    vels = np.arange(start_vel, start_vel + 0.3, 0.1)

# friction_func changes the friction just in s direction
# def friction_func(pose_frenet, waypoints):
#     s_max = np.max(waypoints[:, 0]) # Handles the case when waypoints is flipped
#     s = pose_frenet[0]
#     ey = pose_frenet[1]
#     ey_max = 10 # Maximum lateral error - Track width
#     if abs(ey) > ey_max:
#         return np.nan
#     if s < 0.5 * s_max:
#         # Linear change from 1.1 abs(ey) = 0 to 0.5 abs(ey) >= ey_max
#         ey = min(abs(ey), ey_max)
#         return 1.1 - 0.6 * ey / ey_max
#     else:
#         ey = min(abs(ey), ey_max)
#         return 0.5 - 0.3 * ey / ey_max

def test_friction_func(map_ind):
    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=7, reverse=False)

    # Check if the waypoints are of the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
    if waypoints.shape[1] == 4:
        waypoints = centerline_to_frenet(waypoints)
        
    # Sample 10,000 points over s, each with 100 points over ey
    s = np.linspace(0, np.max(waypoints[:, 0]), 1000)
    ey = np.linspace(-10, 10, 20)
    mu = np.zeros((s.shape[0], ey.shape[0]))
    for i in range(s.shape[0]):
        for j in range(ey.shape[0]):
            mu[i, j] = friction_func(np.array([s[i], ey[j], 0]), waypoints)
            
    # convert to cartesian
    x = np.zeros(mu.shape)
    y = np.zeros(mu.shape)
    for i in range(s.shape[0]):
        for j in range(ey.shape[0]):
            x[i, j], y[i, j], _ = frenet_to_cartesian(np.array([s[i], ey[j], 0]), waypoints)
    
    # Plot colorbar with x,y vs mu and the waypoints
    plt.figure()
    plt.pcolor(x, y, mu)
    plt.show()

def main():
    """
    main entry point
    """
    map_ind = 39

    # # Visualize friction function
    # test_friction_func(map_ind)

    with open('EGP/maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    # friction_funcs = [friction_func]
    # vels = np.arange(8, 9, 1)
    vels = np.arange(15, 16, 1)

    if control_model == "ext_kinematic":
        Kp = 5
        Ki = 0.01
        Kd = 0
    elif control_model == "pure_pursuit":
        Kp = 1
        Ki = 0.01
        Kd = 0

    # for map_ind in range(7, 40):
    
    # for friction_func_ in friction_funcs:
    total_states = []
    total_controls = []
    for vel in vels:
        for reverse in range(2):
            step_reward = 0
            map_info = np.genfromtxt('map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
            print(map_ind, map_info[0], 'reverse', reverse)
            waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=map_scale, reverse=reverse)
            
            # waypoints_frenet = waypoints.copy()
            # Check if the waypoints are of the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
            if waypoints.shape[1] == 4:
                print("centerline to frenet")
                waypoints = centerline_to_frenet(waypoints)
                # reduced_rate = 1
                # map_ind = 16
                reduced_rate = 2
                w0=waypoints[0,3]-np.pi
                if map_reverse:
                    w0 = waypoints[0,3]
            else: # raceline
                print("reduced_rate is 5")
                reduced_rate = 5
                w0=waypoints[0,3]
                if map_reverse:
                    w0 = waypoints[0,3]-np.pi
            
            waypoints[:, -2] = vel
                
            print('vel', vel)
            # print('friction', friction_func_.__name__)

            # sample every env.reduced_rate 10 waypoints for patch in visualization
            reduced_waypoints = waypoints[::reduced_rate, :] 
            s_max = np.max(reduced_waypoints[:, 0])
            friction = [friction_func(i,s_max) for i in range(reduced_waypoints.shape[0])]


            # work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 0.950338203837889}
            # planner = PurePursuitPlanner(conf, 0.805975 + 1.50876)
            # planner.waypoints = waypoints
            if ACC_VS_CONTROL:
                warnings.warn('ACC_VS_CONTROL is not supported')
            else:
                env = ChronoEnv().make(timestep=step_size, control_period=0.1, waypoints=waypoints, patch_scale=patch_scale,
                                        sample_rate_waypoints=reduced_rate, friction=friction, speedPID_Gain=[Kp, Ki, Kd],
                                        steeringPID_Gain=[0.5,0,0], x0=reduced_waypoints[0,1], 
                                        y0=reduced_waypoints[0,2], w0=w0)
                
                obs = {'poses_x': env.my_hmmwv.state[0],'poses_y': env.my_hmmwv.state[1],
                'vx':env.my_hmmwv.state[2], 'poses_theta': env.my_hmmwv.state[3],
                'vy':env.my_hmmwv.state[4],'yaw_rate':env.my_hmmwv.state[5],'steering':env.my_hmmwv.state[6]}
  

            if control_model == "ext_kinematic":
                planner_ekin_mpc_config = MPCConfigEXT()
                planner_ekin_mpc_config.dlk = np.sqrt((waypoints[1, 1] - waypoints[0, 1]) ** 2 + (waypoints[1, 2] - waypoints[0, 2]) ** 2)
                reset_config(planner_ekin_mpc_config, env.vehicle_params)
                planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=planner_ekin_mpc_config), 
                                                waypoints=waypoints.copy(),
                                                config=planner_ekin_mpc_config) #path_follow_mpc.py
                planner_ekin_mpc.config.DTK = env.step_size * SEGMENT_LENGTH
                # if planner_ekin_mpc.config.DTK != env.control_period:
                #     warnings.warn("planner_ekin_mpc.config.DTK != env.control_period. Setting DTK to be equal to env.control_period.")
                #     planner_ekin_mpc.config.DTK = env.control_period
                #     print("planner_ekin_mpc.config.DTK", planner_ekin_mpc.config.DTK)
            elif control_model == "pure_pursuit":
                # Init Pure-Pursuit regulator
                work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance
                # work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 15, 'vgain': 1.0} # tlad: look ahead distance
                conf.wpt_path = MAP_DIR + conf.wpt_path
                conf.wpt_xind = 1
                conf.wpt_yind = 2
                conf.wpt_vind = -2
                planner_pp = PurePursuitPlanner(conf, env.vehicle_params.WB)
                planner_pp.waypoints = waypoints.copy()

                ballT = env.vis.GetSceneManager().addSphereSceneNode(0.1)
                ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)

            if RENDER: env.render()

            # # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
            # obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
            #                                                     waypoints[0, conf.wpt_yind], 
            #                                                     init_theta, 0.0, 0.0, 0.0, 0.0]]))

            laptime = 0.0
            start = time.time()            
            controls = []
            states = []
            cnt = 0
            while env.lap_counter < num_laps:
                if cnt % 42 == 0:
                    target_vel = vel + np.random.uniform(-VEL_SAMPLE_UP, VEL_SAMPLE_UP)
                
                if control_model == "ext_kinematic":
                    warnings.warn('ext_kinematic is not supported')
                elif control_model == "pure_pursuit":
                    waypoints[:, -2] = target_vel
                    planner_pp.waypoints = waypoints.copy()
                    speed, steering = planner_pp.plan(obs['poses_x'], obs['poses_y'],
                                                    obs['poses_theta'],work['tlad'], work['vgain'])
                    # print("pure pursuit input speed", speed, "steering angle ratio [-1,1]", steering/env.vehicle_params.MAX_STEER)  
                    u = np.array([speed, steering])
                    # Visualize the lookahead point of pure-pursuit
                    if planner_pp.lookahead_point is not None:
                        pT = chrono.ChVectorD(planner_pp.lookahead_point[0], planner_pp.lookahead_point[1], 0.0)
                        ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))
                    else:
                        print("No lookahead point found in main_map_info.py!!!")
                # speed, steer, ind = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
                #                             work['vgain'], target_vel)
                
                # pose_frenet = cartesian_to_frenet(np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]), waypoints_frenet)
                # env.params['tire_p_dy1'] = friction_func_(pose_frenet, waypoints_frenet)  # mu_y  # mu_y
                # env.params['tire_p_dx1'] = friction_func_(pose_frenet, waypoints_frenet)  # mu_y  # mu_x
                
                if ACC_VS_CONTROL:
                    # steering angle velocity input to steering velocity acceleration input
                    # accl, sv = pid(target_vel, steer, obs['x4'][0], obs['x3'][0], param1['sv_max'], param1['a_max'],
                    #             param1['v_max'], param1['v_min'])
                    # control = np.array([sv, accl])
                    warnings.warn('ACC_VS_CONTROL is not supported')
                else:
                    control = np.array([steering, speed])
                
                    
                for i in range(SEGMENT_LENGTH):
                    obs, rew, done, info = env.step(steering, speed)
                    step_reward += rew

                # state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                # ## x3 = steering angle of front wheels
                # ## x4 = velocity in x-direction
                # ## x6 = yaw rate
                # ## x11 = velocity in y-direction
                state = np.array([obs['steering'], obs['vx'], obs['yaw_rate'], obs['vy']])
                control = np.array([steering, env.speedPID_output])
                
                cnt += 1
                states.append(state)
                controls.append(control)
                
                if cnt % SAVE_STEP == 0:
                    total_states.append(np.stack(states))
                    total_controls.append(np.stack(controls))
                    controls = []
                    states = []

                laptime += step_reward
                if RENDER: 
                    # env.render(mode='human_fast')
                    # print('target_vel', target_vel, np.sqrt(obs['x4'][0] ** 2 + obs['x11'][0] ** 2))
                    # print(ind, f'x {obs["x1"][0]:.2f}, y {obs["x2"][0]:.2f}, yaw {obs["x5"][0]:.2f}, yawrate {obs["x6"][0]:.2f}' + \
                    #     f', vx {obs["x4"][0]:.2f}, vy {obs["x11"][0]:.2f}, steer {obs["x3"][0]:.2f}')
                    env.render()
                    # print('steering', steering, 'target_vel', target_vel, 'speed', speed, 'speedPID_output(acc)', env.speedPID_output)
            
                if env.lap_counter >= num_laps:
                    done = True
                    print("Completed num_laps. env.time:",env.time)
                    break

            print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    np.save(SAVE_DIR + 'states_f{}_v{}.npy'.format("friction_func", int(np.rint(vels[0]*100))), total_states)
    np.save(SAVE_DIR + 'controls_f{}_v{}.npy'.format("friction_func", int(np.rint(vels[0]*100))), total_controls)

if __name__ == '__main__':
    main()

