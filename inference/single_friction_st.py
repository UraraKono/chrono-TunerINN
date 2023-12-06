import time
import yaml
# import gym
import numpy as np
from argparse import Namespace
import os, sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp

# os.environ['F110GYM_PLOT_SCALE'] = str(10.)
# from planner import PurePursuitPlanner, pid
# from inference.dynamics_models.mb_model_params import param1
# from additional_renderers import *
from jax_mpc.mppi import MPPI
sys.path.append("../")
import vehicle_data_gen_utils.jax_utils as jax_utils
import vehicle_data_gen_utils.utils as utils
import vehicle_data_gen_utils.frenet_utils as frenet_utils
from infer_env import InferEnv, JaxInfer
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *
from EGP.regulators.pure_pursuit import *


# SIM_TIME_STEP = 0.01
SIM_TIME_STEP = 2e-3
RENDER = True
MAP_DIR = '/home/tomas/Documents/vehicle_data_gen/f1tenth_racetracks/'
ACC_VS_CONTROL = False
# NOISE = [1e-2, 1e-2, 1e-2] # control_vel, control_steering, state 
NOISE = [0, 0, 0] # control_vel, control_steering, state 
MB_MPPI = False
friction_c = 0.5
map_ind = 17 # sanpaulo
# map_ind = 41 # duo
# map_ind = 44 # round
reduced_rate = 5
patch_scale = 1.5

n_steps = 10
# n_samples = 128
n_samples = 1
control_sample_trunc = jnp.array([1.0, 1.0])
state_predictor = 'st'
scale = 1
DT = 0.1
# SEGMENT_LENGTH = 10
SEGMENT_LENGTH = int(np.rint(DT/SIM_TIME_STEP))
# Speed PID gains
Kp = 5
Ki = 0.01
Kd = 0

def load_map(MAP_DIR, map_info, conf, scale=1, reverse=False):
    """
    loads waypoints
    """
    conf.wpt_path = map_info[0]
    conf.wpt_delim = map_info[1]
    conf.wpt_rowskip = int(map_info[2])
    conf.wpt_xind = int(map_info[3])
    conf.wpt_yind = int(map_info[4])
    conf.wpt_thind = int(map_info[5])
    conf.wpt_vind = int(map_info[6])
    
    print('loading map', MAP_DIR + conf.wpt_path)
    waypoints = np.loadtxt(MAP_DIR + conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    if reverse: # NOTE: reverse map
        waypoints = waypoints[::-1]
        if map_ind == 41: waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + 3.14
    waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + np.pi / 2
    waypoints[:, conf.wpt_yind] = waypoints[:, conf.wpt_yind] * scale
    waypoints[:, conf.wpt_xind] = waypoints[:, conf.wpt_xind] * scale # NOTE: map scales
    
    # NOTE: initialized states for forward
    if conf.wpt_thind == -1:
        init_theta = np.arctan2(waypoints[1, conf.wpt_yind] - waypoints[0, conf.wpt_yind], 
                                waypoints[1, conf.wpt_xind] - waypoints[0, conf.wpt_xind])
    else:
        init_theta = waypoints[0, conf.wpt_thind]
    
    return waypoints, conf, init_theta


if len(sys.argv) > 1:
    vel = float(sys.argv[1])
else:
    vel = 5

def main():
    """
    main entry point
    """
    with open('../EGP/maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    config = utils.ConfigJSON()
    jrng = jax_utils.oneLineJaxRNG(0)
    colorpal = utils.colorPalette()
    timer = utils.Timer()
    plt_utils = utils.pltUtils()
    config.load_file('maps/config.json')
    normalization_param = np.array(config.d['normalization_param']).T
    
    map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    print(map_ind, map_info[0], 'reverse', 0, 'vel', vel, 'friction', friction_c)
    waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=scale, reverse=0)
    waypoints[:, -2] = vel
    reduced_waypoints = waypoints[::reduced_rate, :] 
    friction = [friction_c for i in range(reduced_waypoints.shape[0])]

    if ACC_VS_CONTROL:
        warnings.warn('ACC_VS_CONTROL is not supported')
    else:   
        # env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
        #             num_agents=1, timestep=SIM_TIME_STEP, model='dynamic_ST', drive_control_mode='vel',
        #             steering_control_mode='angle')
        env = ChronoEnv().make(timestep=SIM_TIME_STEP, control_period=DT, waypoints=waypoints, 
                                friction=friction, sample_rate_waypoints=reduced_rate, patch_scale=patch_scale,
                                speedPID_Gain=[Kp, Ki, Kd],steeringPID_Gain=[0.5,0,0], model='ST',
                                x0=waypoints[0,1],y0=waypoints[0,2],w0=waypoints[0,3],visualize=RENDER)

    # initializing the MPPIs
    infer_env_st = InferEnv(waypoints, n_steps, mode=state_predictor, DT=DT) # state predictor that MPPI uses. It has vehicle equations of motion.
    print('infer_env_st', infer_env_st)
    mppi_st = MPPI(config, n_iterations=1, n_steps=n_steps, n_samples=n_samples, 
                a_noise=control_sample_trunc, scan=False, mode=state_predictor)
    
    mppi_state_st = mppi_st.init_state(infer_env_st.a_shape, jrng.new_key())
    a_opt_original = mppi_state_st[0].copy()
    # print('a_opt_original', a_opt_original) # all zeros

    
    if RENDER: 
        map_waypoint_renderer = MapWaypointRenderer(waypoints)
        steer_renderer = SteerRenderer(np.zeros((3,)), np.ones((1,)), colorpal.rgb('r'))
        acce_renderer = AcceRenderer(np.zeros((3,)), np.ones((1,)))
        reference_traj_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('br'), mode='quad')
        sampled_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('g'), point_size=2, mode='quad')
        sampled_renderer_mb = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('p'), point_size=2, mode='quad')
        sampled_renderer_st = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('y'), point_size=2, mode='quad')
        opt_traj_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('o'), point_size=5, mode='quad')
        # renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, sampled_renderer_mb, opt_traj_renderer]
        renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, steer_renderer, acce_renderer, sampled_renderer_mb, sampled_renderer_st, opt_traj_renderer]
        # renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, steer_renderer, acce_renderer]
        env.add_render_callback(get_render_callback(renderers))

    laptime = 0
    start = time.time()            
    test_laptime = 200
    
    tracking_errors = []
    prediction_errors = []
    ref_speeds = []
    speed_profiles = []
    controls = []
    
    np.set_printoptions(suppress=True, precision=10)
    # while True:

    tracking_error = []
    prediction_error = []
    speed_profile = []
    ref_speed = []
    control_list = []
    
    ## init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    # obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
    #                                                 waypoints[0, conf.wpt_yind], 
    #                                                 init_theta, 0.0, waypoints[0, conf.wpt_vind], 0.0, 0.0]]))
    if env.model == 'ST':
        obs = {'poses_x':env.my_hmmwv.state[0], 'poses_y':env.my_hmmwv.state[1],
               'steering':env.my_hmmwv.state[2], 'v':env.my_hmmwv.state[3],
               'poses_theta':env.my_hmmwv.state[4], 'yaw_rate':env.my_hmmwv.state[5], 
               'beta':env.my_hmmwv.state[6], 'state':env.my_hmmwv.state}
    else:
        obs = {'poses_x':env.my_hmmwv.state[0], 'poses_y':env.my_hmmwv.state[1],
                'vx':env.my_hmmwv.state[2], 'poses_theta': env.my_hmmwv.state[3],
                'vy':env.my_hmmwv.state[4],'yaw_rate':env.my_hmmwv.state[5],
                'steering':env.my_hmmwv.state[6]}
        
    ### Warm up the vehicle with pure pursuit###
    # Init Pure-Pursuit regulator
    work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance
    # work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 15, 'vgain': 1.0} # tlad: look ahead distance
    conf_purepursuit = conf
    conf_purepursuit.wpt_path = MAP_DIR + conf_purepursuit.wpt_path
    conf_purepursuit.wpt_xind = 1
    conf_purepursuit.wpt_yind = 2
    conf_purepursuit.wpt_vind = -2
    planner_pp = PurePursuitPlanner(conf_purepursuit, env.vehicle_params.WB)
    planner_pp.waypoints = waypoints.copy()

    while obs['v'] < 2:
        # Render scene
        env.render()
        planner_pp.waypoints = waypoints.copy()
        speed, steering = planner_pp.plan(obs['poses_x'], obs['poses_y'], obs['poses_theta'],
                                        work['tlad'], work['vgain'])
        print("pure pursuit input speed", speed, "steering angle ratio [-1,1]", steering/env.vehicle_params.MAX_STEER)  
        obs, _, _, _ = env.step(steering, speed)

    ### MPPI ###
    a_opt = a_opt_original.copy()
    for laptime in range(test_laptime):
        # target_vel = vel + np.random.uniform(-3, 3)
        # target_vel = vel 
        target_vel = None
    
        # state_st_0 = obs['state'][0]
        state_st_0 = env.my_hmmwv.state
        
        a_opt = jnp.concatenate([a_opt[1:, :],
                    jnp.expand_dims(jnp.zeros((2,)),
                                    axis=0)])  # [n_steps, dim_a]#previous action steering speed and acceleration
        da = jax.random.truncated_normal(
            jrng.new_key(),
            -jnp.ones_like(a_opt) - a_opt,
            jnp.ones_like(a_opt) - a_opt,
            shape=(n_samples, n_steps, 2)
        )  # [n_samples, n_steps, dim_a] #sample noise added to future action
        
        # print('state_st_0', state_st_0)
        state_st_0 = jnp.array(state_st_0)
        print("state_st_0.copy()", type(state_st_0.copy()), state_st_0.copy())
        # print("infer_env_st BEFORE", type(infer_env_st), infer_env_st)
        # infer_env_st = jnp.array(infer_env_st)
        # print("infer_env_st AFTER", type(infer_env_st), infer_env_st)
        reference_traj, ind = infer_env_st.get_refernece_traj(state_st_0.copy(), state_st_0[3], vind=conf.wpt_vind, speed_factor=1.0)
        print('reference_traj', reference_traj)
        print('ind', ind)
        ref_speed.append(waypoints[ind, conf.wpt_vind])
        mppi_state_st, predicted_states_st, s_opt_st, _, _, _ = mppi_st.update(mppi_state_st, infer_env_st, state_st_0, jrng.new_key(), da)
        a_opt = mppi_state_st[0]
        print('a_opt', a_opt)
        control = mppi_state_st[0][0]
        predicted_states = jax.device_get(predicted_states_st[0])
        print('predicted_states', predicted_states)
        print('s_opt_st', s_opt_st)
        state_opt = jax.device_get(s_opt_st)
        # if RENDER: sampled_renderer_st.update(np.concatenate(predicted_states[:, :, :2]))

        
        if RENDER: 
            # acce_renderer.update(state_st_0, control)
            # reference_traj_renderer.update(reference_traj)
            # opt_traj_renderer.update(jax.device_get(s_opt_st))
            # map_waypoint_renderer.update(state_st_0[:2])
            # steer_renderer.update(state_st_0, control)
            # env.render(mode='human_fast')
            env.render()


        for i in range(SEGMENT_LENGTH):
            control0 = control[0] + np.random.normal(scale=NOISE[0]) # steering velocity
            control1 = control[1] + np.random.normal(scale=NOISE[1]) # acceleration
            print('control', [control0, control1])
            control_denom = np.array([control0, control1]) * normalization_param[0, 7:9]/2 # denormalize
            print('control denom', control_denom)
            
            steering = env.driver_inputs.m_steering*env.vehicle_params.MAX_STEER + control_denom[0]*env.control_period # [-max steer,max steer]
            speed = env.my_hmmwv.state[2] + control_denom[1]*env.control_period
            print('steering', steering, 'speed', speed)
            # obs, _, _, _ = env.step(np.array([[control0, control1]]) * normalization_param[0, 7:9]/2)
            obs, _, _, _ = env.step(steering, speed)

        laptime += 1
        # state_st_1 = obs['state'][0]
        state_st_1 = obs['state']
        print('state_st_1', state_st_1)
        print('state_opt', state_opt)
        

        # frenet_pose = frenet_utils.cartesian_to_frenet(np.array([obs['x1'][0], obs['x2'][0], obs['x4'][0]]), waypoints)
        frenet_pose = frenet_utils.cartesian_to_frenet(np.array([obs['poses_x'], obs['poses_y'], obs['poses_theta']]), waypoints)
        tracking_error.append(np.abs(frenet_pose[1]))
        error = state_st_1 - state_opt[0]
        error[4] = np.abs(np.sin(state_st_1[4]) - np.sin(state_opt[0][4])) + np.abs(np.cos(state_st_1[4]) - np.cos(state_opt[0][4]))
        prediction_error.append(np.abs(error))
        # speed_profile.append(np.sqrt(state_nf_1[3] ** 2 + state_nf_1[6] ** 2))
        speed_profile.append(state_st_0[3])
        control_list.append(control)
        
    tracking_errors.append(np.asarray(tracking_error))
    prediction_errors.append(np.asarray(prediction_error))
    speed_profiles.append(np.asarray(speed_profile))
    ref_speeds.append(np.asarray(ref_speed))
    controls.append(np.asarray(control_list))

    print('mean tracking error:', np.mean(tracking_errors[0]))
    print('mean speed:', np.mean(speed_profiles[0]))
    
    axs = plt_utils.get_fig([2, 1])
    axs[0].plot(np.arange(test_laptime), speed_profiles[0])
    axs[0].plot(np.arange(test_laptime), ref_speeds[0], markersize=1)
    axs[0].set_title('speed')
    axs[0].legend(['st', 'ref'])
    
    axs[1].plot(np.arange(test_laptime), tracking_errors[0])
    axs[1].set_title('tracking error')
    axs[1].legend(['st'])
    plt_utils.show()
    
    axs = plt_utils.get_fig([4, 2])
    titles = ['x', 'y', 'steer', 'vx', 'yaw', 'yawrate', 'vy']
    for ind in range(7):
        axs[ind].plot(np.arange(test_laptime), prediction_errors[0][:, ind])
        axs[ind].set_title(titles[ind])
    plt_utils.show()

if __name__ == '__main__':
    main()

