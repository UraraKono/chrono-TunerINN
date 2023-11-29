import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import os, sys
sys.path.append("../")
os.environ['F110GYM_PLOT_SCALE'] = str(10.)
from planner import PurePursuitPlanner, pid
from inference.dynamics_models.mb_model_params import param1
from additional_renderers import *
from jax_mpc.mppi import MPPI
import utils.jax_utils as jax_utils
import jax
import utils.utils as utils
import jax.numpy as jnp
import utils.frenet_utils as frenet_utils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


SIM_TIME_STEP = 0.01
RENDER = True
SAVE_DIR = '/home/lucerna/Documents/'
MAP_DIR = '/home/lucerna/MEGA/Reasearch/tuner_inn/vehicle_data_gen/f1tenth_racetracks/'
# MAP_DIR = '/home/lucerna/Documents/vehicle_data_gen/f1tenth_racetracks/'
ACC_VS_CONTROL = True
# NOISE = [1e-2, 1e-2, 1e-2] # control_vel, control_steering, state 
NOISE = [0, 0, 0] # control_vel, control_steering, state 
MB_MPPI = False

friction = 1.1
# map_ind = 17 # sanpaulo
map_ind = 41 # duo
# map_ind = 44 # round
n_steps = 10
n_samples = 128
control_sample_trunc = jnp.array([1.0, 1.0])
state_predictor = 'st'
scale = 1
DT = 0.1
SEGMENT_LENGTH = 10
# SEGMENT_LENGTH = int(np.rint(DT/SIM_TIME_STEP))

from infer_env import InferEnv, JaxInfer

    
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
    with open('maps/config_example_map.yaml') as file:
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
    print(map_ind, map_info[0], 'reverse', 0, 'vel', vel, 'friction', friction)
    waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=scale, reverse=0)
    if ACC_VS_CONTROL:
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                num_agents=1, timestep=SIM_TIME_STEP, model='dynamic_ST', drive_control_mode='acc',
                steering_control_mode='vel')
    else:
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                    num_agents=1, timestep=SIM_TIME_STEP, model='dynamic_ST', drive_control_mode='vel',
                    steering_control_mode='angle')
        
    # initializing the MPPIs
    infer_env_st = InferEnv(waypoints, n_steps, mode=state_predictor, DT=DT)
    mppi_st = MPPI(config, n_iterations=1, n_steps=n_steps, n_samples=n_samples, 
                a_noise=control_sample_trunc, scan=False, mode=state_predictor)
    
    mppi_state_st = mppi_st.init_state(infer_env_st.a_shape, jrng.new_key())
    a_opt_original = mppi_state_st[0].copy()

    
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
    test_laptime = 100
    
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
    obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                    waypoints[0, conf.wpt_yind], 
                                                    init_theta, 0.0, waypoints[0, conf.wpt_vind], 0.0, 0.0]]))
    
    a_opt = a_opt_original.copy()
    for laptime in range(test_laptime):
        # target_vel = vel + np.random.uniform(-3, 3)
        # target_vel = vel 
        target_vel = None
        
    
        state_st_0 = obs['state'][0]
        
        a_opt = jnp.concatenate([a_opt[1:, :],
                    jnp.expand_dims(jnp.zeros((2,)),
                                    axis=0)])  # [n_steps, dim_a]
        da = jax.random.truncated_normal(
            jrng.new_key(),
            -jnp.ones_like(a_opt) - a_opt,
            jnp.ones_like(a_opt) - a_opt,
            shape=(n_samples, n_steps, 2)
        )  # [n_samples, n_steps, dim_a]
        

        state_st_0 = np.asarray(state_st_0)
        reference_traj, ind = infer_env_st.get_refernece_traj(state_st_0.copy(), target_vel, vind=conf.wpt_vind, speed_factor=1.0)
        ref_speed.append(waypoints[ind, conf.wpt_vind])
        mppi_state_st, predicted_states_st, s_opt_st, _, _, _ = mppi_st.update(mppi_state_st, infer_env_st, state_st_0, jrng.new_key(), da)
        a_opt = mppi_state_st[0]
        control = mppi_state_st[0][0]
        predicted_states = jax.device_get(predicted_states_st[0])
        state_opt = jax.device_get(s_opt_st)
        if RENDER: sampled_renderer_st.update(np.concatenate(predicted_states[:, :, :2]))

        
        
        if RENDER: 
            acce_renderer.update(state_st_0, control)
            reference_traj_renderer.update(reference_traj)
            opt_traj_renderer.update(jax.device_get(s_opt_st))
            map_waypoint_renderer.update(state_st_0[:2])
            steer_renderer.update(state_st_0, control)
            env.render(mode='human_fast')

        
        # env.params['tire_p_dy1'] = friction  # mu_y
        # env.params['tire_p_dx1'] = friction  # mu_x
        env.params['mu'] = friction  # mu_y
        env.params['mu'] = friction  # mu_x
        for i in range(SEGMENT_LENGTH):
            control0 = control[0] + np.random.normal(scale=NOISE[0])
            control1 = control[1] + np.random.normal(scale=NOISE[1])
            # print('control', [control0, control1])
            obs, _, _, _ = env.step(np.array([[control0, control1]]) * normalization_param[0, 7:9]/2)

        laptime += 1
        state_st_1 = obs['state'][0]
        print('state_st_1', state_st_1)
        print('state_opt', state_opt)
        

        frenet_pose = frenet_utils.cartesian_to_frenet(np.array([obs['x1'][0], obs['x2'][0], obs['x4'][0]]), waypoints)
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

    
    # axs = plt_utils.get_fig([2, 1])
    # for ind in range(2):
    #     axs[ind].plot(np.arange(test_laptime), controls[0][:, ind])
    #     axs[ind].legend(['st'])
    # plt_utils.show()
    
    
    
                
                

if __name__ == '__main__':
    main()


# maps = os.listdir(MAP_DIR)[:-1]
# del maps[3]
# print(maps)
# row = '# wpt_path|wpt_delim|wpt_rowskip|wpt_xind|wpt_yind|wpt_thind|wpt_vind'
# file1 = open("map_info.txt", "w")
# file1.write(row + '\n')
# for ind in range(len(maps)):
#     file1.write(str(ind*2) + '|' + maps[ind] + '/' + maps[ind] + '_centerline.csv|,|1|0|1|-1|-1' + '\n')
#     file1.write(str(ind*2+1) + '|' + maps[ind] + '/' + maps[ind] + '_raceline.csv|;|3|1|2|3|5' + '\n')

# exit()