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


SEGMENT_LENGTH = 10
SIM_TIME_STEP = 0.01
RENDER = True
SAVE_DIR = '/home/lucerna/Documents/DATA/tuner_inn/track39/'
MAP_DIR = '/home/lucerna/MEGA/Reasearch/tuner_inn/vehicle_data_gen/f1tenth_racetracks/'
# MAP_DIR = '/home/lucerna/Documents/vehicle_data_gen/f1tenth_racetracks/'
ACC_VS_CONTROL = True
SAVE_STEP = 210
NOISE = [1e-2, 1e-2, 1e-2] # control_vel, control_steering, state 
# NOISE = [0, 0, 0] # control_vel, control_steering, state 
MB_MPPI = False

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
        waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + 3.14
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

def main():
    """
    main entry point
    """
    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    config = utils.ConfigJSON()
    # config.load_file('/home/lucerna/MEGA/Reasearch/tuner_inn/tuner_inn/inference/dynamics_models/config.json')
    
    
    jrng = jax_utils.oneLineJaxRNG(0)
    colorpal = utils.colorPalette()
    timer = utils.Timer()
    plt_utils = utils.pltUtils()
    
    friction = 0.5
    # map_ind = 17 # sanpaulo
    map_ind = 41 # duo
    # map_ind = 44 # round
    n_steps = 8
    n_samples = 64
    control_sample_trunc = jnp.array([1.0, 1.0])
    state_predictor = 'nf'
    scale = 1
    DT = 0.2
    
    
    map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    print(map_ind, map_info[0], 'reverse', 0, 'vel', vel, 'friction', friction)
    waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=scale, reverse=0)
    if ACC_VS_CONTROL:
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                num_agents=1, timestep=SIM_TIME_STEP, model='MB', drive_control_mode='acc',
                steering_control_mode='vel')
    else:
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                    num_agents=1, timestep=SIM_TIME_STEP, model='MB', drive_control_mode='vel',
                    steering_control_mode='angle')
        
    # initializing the MPPIs
    infer_env = InferEnv(waypoints, n_steps, mode=state_predictor, DT=DT)
    infer_env_st = InferEnv(waypoints, n_steps, mode='st', DT=DT)
    if MB_MPPI: infer_env_mb = InferEnv(waypoints, n_steps, mode='mb', DT=DT)
    
    mppi = MPPI(config, n_iterations=1, n_steps=n_steps, n_samples=n_samples, 
                a_noise=control_sample_trunc, scan=False, mode=state_predictor)
    mppi_st = MPPI(config, n_iterations=1, n_steps=n_steps, n_samples=n_samples, 
                a_noise=control_sample_trunc, scan=False, mode='st')
    if MB_MPPI: mppi_mb = MPPI(config, n_iterations=1, n_steps=n_steps, n_samples=n_samples, 
                a_noise=control_sample_trunc, scan=False, mode='mb')
    
    mppi_state = mppi.init_state(infer_env.a_shape, jrng.new_key())
    mppi_state_st = mppi_st.init_state(infer_env.a_shape, jrng.new_key())
    if MB_MPPI: mppi_state_mb = mppi_mb.init_state(infer_env.a_shape, jrng.new_key())

    
    if RENDER: 
        map_waypoint_renderer = MapWaypointRenderer(waypoints)
        steer_renderer = SteerRenderer(np.zeros((3,)), np.ones((1,)), colorpal.rgb('r'))
        acce_renderer = AcceRenderer(np.zeros((3,)), np.ones((1,)))
        reference_traj_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('br'), mode='quad')
        sampled_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('g'), point_size=2, mode='quad')
        sampled_renderer_mb = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('p'), point_size=2, mode='quad')
        sampled_renderer3 = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('y'), point_size=2, mode='quad')
        opt_traj_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb('o'), point_size=5, mode='quad')
        # renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, sampled_renderer_mb, opt_traj_renderer]
        renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, steer_renderer, acce_renderer, sampled_renderer_mb, sampled_renderer3]
        # renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, steer_renderer, acce_renderer]
        env.add_render_callback(get_render_callback(renderers))

    # # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                        waypoints[0, conf.wpt_yind], 
                                                        init_theta, 0.0, vel, 0.0, 0.0]]))

    laptime = 0
    laptime_record = []
    start = time.time()            
    controls = []
    states = []
    cnt = 0
    test_laptime = 2000
    
    tracking_errors = []
    prediction_errors = []
    ref_speeds = []
    speed_profiles = []
    
    np.set_printoptions(suppress=True, precision=10)
    # while True:
    test = 3 if MB_MPPI else 2
    for test in range(test):
        tracking_error = []
        prediction_error = []
        ref_speed = []
        speed_profile = []
        
        try:
            # target_vel = vel + np.random.uniform(-3, 3)
            # target_vel = vel 
            target_vel = None
            
            
            state_nf_0 = np.array([obs['x1'][0], obs['x2'][0], obs['x3'][0], 
                                obs['x4'][0], obs['x5'][0], obs['x6'][0], obs['x11'][0]])
            state_nf_0 = state_nf_0 + np.random.normal(scale=NOISE[2], size=state_nf_0.shape)

            state_st_0 = np.asarray(infer_env.state_nf2st(state_nf_0))
            state_mb_0 = np.asarray(infer_env.state_nf2mb(obs['state'][0], state_nf_0))
             
            # timer.tic()
            reference_traj, ind = infer_env.get_refernece_traj(state_st_0.copy(), target_speed=target_vel, vind=conf.wpt_vind, speed_factor=1.0)
            _ = infer_env_st.get_refernece_traj(state_st_0.copy(), target_vel, vind=conf.wpt_vind, speed_factor=1.0)
            ref_speed.append(waypoints[ind, conf.wpt_vind])
            

            a_opt = mppi_state[0]
            a_opt = jnp.concatenate([a_opt[1:, :],
                        jnp.expand_dims(jnp.zeros((2,)),
                                        axis=0)])  # [n_steps, dim_a]
            da = jax.random.truncated_normal(
                jrng.new_key(),
                -jnp.ones_like(a_opt) - a_opt,
                jnp.ones_like(a_opt) - a_opt,
                shape=(n_samples, n_steps, 2)
            )  # [n_samples, n_steps, dim_a]
            
            # if laptime >= test_laptime:
            mppi_state_st, predicted_states_st, s_opt_st, _, _, _ = mppi_st.update(mppi_state_st, infer_env_st, state_st_0.copy(), jrng.new_key(), da)
            mppi_state, predicted_states, s_opt, r_sample, a, nf_dyna = mppi.update(mppi_state, infer_env, state_nf_0, jrng.new_key(), da)
            a_opt = mppi_state[0]
            # mb_dyna = jax.device_get(mb_dyna)
            # nf_dyna = jax.device_get(nf_dyna)
            
            predicted_states = jax.device_get(predicted_states[0])
            # predicted_states_st = jax.device_get(predicted_states_st[0])
            
            
            
            control = mppi_state[0][0]
            control_st = mppi_state_st[0][0]
            if laptime >= test_laptime:
                control = control_st
            
            controls.append(control)
            # controls_st.append(control_st)
            # if spread < 0.5:
            #     print('here')
            #     control = jnp.zeros(2)
            
            if RENDER: 
                acce_renderer.update(state_st_0, control)
                sampled_renderer.update(np.concatenate(predicted_states[:, :, :2]))
                reference_traj_renderer.update(reference_traj)
                opt_traj_renderer.update(s_opt)
                map_waypoint_renderer.update(state_st_0[:2])
                steer_renderer.update(state_st_0, control)
                env.render(mode='human_fast')

            

                if MB_MPPI: _ = infer_env_mb.get_refernece_traj(state_st_0.copy(), target_vel)
                if MB_MPPI: mppi_state_mb, predicted_states_mb, s_opt_mb, _, _, mb_dyna = mppi_mb.update(mppi_state, infer_env_mb, obs['state'][0], jrng.new_key(), da)
                if MB_MPPI: predicted_states_mb = jax.device_get(predicted_states_mb[0])
                if MB_MPPI: sampled_renderer_mb.update(np.concatenate(predicted_states_mb[:, :, :2]))
            
            env.params['tire_p_dy1'] = friction  # mu_y
            env.params['tire_p_dx1'] = friction  # mu_x
            for i in range(SEGMENT_LENGTH):
                obs, rew, done, info = env.step(np.array([[control[0] + np.random.normal(scale=NOISE[0]),
                                                        control[1] + np.random.normal(scale=NOISE[1])]]))
            
            laptime += 1
            state_st_1 = np.array([obs['x1'][0], obs['x2'][0], obs['x3'][0], 
                                    np.sqrt(obs['x11'][0] ** 2 + obs['x4'][0] ** 2), 
                                    obs['x5'][0], obs['x6'][0],
                                    np.arctan2(obs['x11'][0], obs['x4'][0])])
            
            if state_st_1[3] < 5:
                raise ZeroDivisionError
            state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
            ## x3 = steering angle of front wheels
            ## x4 = velocity in x-direction
            ## x6 = yaw rate
            ## x11 = velocity in y-direction
            
            
            
            
            cnt += 1
            states.append(state)
            
            # if cnt == 2:
            #     exit()
            
            frenet_pose = frenet_utils.cartesian_to_frenet(np.array([obs['x1'][0], obs['x2'][0], obs['x4'][0]]), waypoints)
            tracking_error.append(np.abs(frenet_pose[1]))
            
                
            #     # fig = plt.figure(dpi=200)
            #     # plt.plot(np.stack(controls)[:, 0], np.stack(controls)[:, 1], '.', markersize=2)
            #     # plt.plot(np.stack(controls_st)[:, 0], np.stack(controls_st)[:, 1], '.', markersize=2)
            #     # plt.show()
                
                
            #     plt.plot(np.arange(100), np.stack(controls)[-100:, 0], '.', markersize=2)
            #     plt.plot(np.arange(100), np.stack(controls_st)[-100:, 0], '.', markersize=2)
            #     plt.show()
                
            #     fig = plt.figure(dpi=200)
            #     plt.plot(np.arange(100), np.stack(controls)[-100:, 1], '.', markersize=2)
            #     plt.plot(np.arange(100), np.stack(controls_st)[-100:, 1], '.', markersize=2)
            #     plt.show()
                
            if laptime == test_laptime*2:
                axs = plt_utils.get_fig([2, 1], figsize=[10, 5])
                axs[0].plot(np.arange(test_laptime), np.stack(states)[:test_laptime, 1])
                axs[0].plot(np.arange(test_laptime), np.stack(states)[test_laptime:, 1])
                axs[0].plot(np.arange(test_laptime), np.stack(ref_speed)[test_laptime:])
                axs[0].set_title('speed')
                axs[0].legend(['nf', 'st', 'ref'])
                
                axs[1].plot(np.arange(test_laptime), np.stack(tracking_error)[:test_laptime])
                axs[1].plot(np.arange(test_laptime), np.stack(tracking_error)[test_laptime:])
                axs[1].set_title('tracking error')
                axs[1].legend(['nf', 'st'])
                plt_utils.show()
                
                # total_states.append(np.stack(states))
                # total_controls.append(np.stack(controls))
                # controls = []
                # states = []


                # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
                # print('target_vel', target_vel, np.sqrt(obs['x4'][0] ** 2 + obs['x11'][0] ** 2))
                # print(ind, f'x {obs["x1"][0]:.2f}, y {obs["x2"][0]:.2f}, yaw {obs["x5"][0]:.2f}, yawrate {obs["x6"][0]:.2f}' + \
                #     f', vx {obs["x4"][0]:.2f}, vy {obs["x11"][0]:.2f}, steer {obs["x3"][0]:.2f}')
            
            
            
                
            if laptime == test_laptime:
                obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                        waypoints[0, conf.wpt_yind], 
                                                        init_theta, 0.0, vel, 0.0, 0.0]]))
            
            if laptime > 10000:
                print('laptime_record', laptime_record)
                print('laptime_record_len', len(laptime_record))
                print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
                exit()
            if int(laptime) % 100 == 0:
                print(laptime, 'tracking_error', np.sum(tracking_error)/laptime)
        except ZeroDivisionError:
            print('ZeroDivisionError', laptime)
            laptime_record.append(laptime)
                            
            obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                        waypoints[0, conf.wpt_yind], 
                                                        init_theta, 0.0, vel, 0.0, 0.0]]))
                
            

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