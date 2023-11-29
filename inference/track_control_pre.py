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


SEGMENT_LENGTH = 10
SIM_TIME_STEP = 0.01
RENDER = True
SAVE_DIR = '/home/lucerna/Documents/DATA/tuner_inn/track39/'
MAP_DIR = '/home/lucerna/MEGA/Reasearch/tuner_inn/vehicle_data_gen/f1tenth_racetracks/'
# MAP_DIR = '/home/lucerna/Documents/vehicle_data_gen/f1tenth_racetracks/'
ACC_VS_CONTROL = True
SAVE_STEP = 210
NOISE = [1e-2, 1e-2, 1e-2] # control_vel, control_steering, state 


from infer_env import InferEnv, JaxInfer



def warm_up(env, vel, warm_up_steps):
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    
    obs, step_reward, done, info = env.reset(
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    step_count = 0
    while (step_count < warm_up_steps) and (np.abs(obs['x3'][0] - vel) > 0.01):
        try:
            obs, step_reward, done, info = env.step(np.array([[0.0, vel]]))
            if RENDER: 
                env.render(mode='human_fast')
                # print(f'x {obs["x1"][0]:.2f}, y {obs["x2"][0]:.2f}, yaw {obs["x5"][0]:.2f}, yawrate {obs["x6"][0]:.2f}' + \
                #         f', vx {obs["x4"][0]:.2f}, vy {obs["x11"][0]:.2f}, steer {obs["x3"][0]:.2f}')
            step_count += 1
        except ZeroDivisionError:
            print('error warmup: ', step_count)

    
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
    start_vel = float(sys.argv[1])
    # vels = [vel]
    vels = np.arange(start_vel, start_vel + 0.3, 0.1)

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
    
    frictions = [0.5]
    map_ind = 17
    n_steps = 8
    n_samples = 64
    control_sample_noise = jnp.array([1.0, 0.5])
    state_predictor = 'nf'
    

    
    for friction in frictions:
        total_states = []
        total_controls = []
        
        for vel in vels:
            prediction_errors = []
            map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
            print(map_ind, map_info[0], 'reverse', 0, 'vel', vel, 'friction', friction)
            waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=9, reverse=0)
            infer_env = InferEnv(waypoints, n_steps, mode=state_predictor)
            mppi = MPPI(config, n_iterations=1, n_steps=n_steps, n_samples=n_samples, 
                        a_noise=control_sample_noise, scan=False, mode=state_predictor)
            mppi_state = mppi.init_state(infer_env.a_shape, jrng.new_key())

            if ACC_VS_CONTROL:
                env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                        num_agents=1, timestep=SIM_TIME_STEP, model='MB', drive_control_mode='acc',
                        steering_control_mode='vel')
            else:
                env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                            num_agents=1, timestep=SIM_TIME_STEP, model='MB', drive_control_mode='vel',
                            steering_control_mode='angle')
                
            map_waypoint_renderer = MapWaypointRenderer(waypoints)
            reference_traj_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb(7))
            sampled_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb(2), point_size=2)
            opt_traj_renderer = WaypointRenderer(np.zeros((10, 2)), colorpal.rgb(4), point_size=5, mode='quad')
            renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer, opt_traj_renderer]
            # renderers = [map_waypoint_renderer, reference_traj_renderer, sampled_renderer]
            if RENDER: env.add_render_callback(get_render_callback(renderers))
            

            # # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
            obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                                waypoints[0, conf.wpt_yind], 
                                                                init_theta, 0.0, vel, 0.0, 0.0]]))

            laptime = 0.0
            start = time.time()            
            controls = []
            states = []
            cnt = 0
            while True:
                target_vel = vel

                state_st_0 = np.array([obs['x1'][0], obs['x2'][0], obs['x3'][0], 
                                    np.sqrt(obs['x11'][0] ** 2 + obs['x4'][0] ** 2), 
                                    obs['x5'][0], obs['x6'][0],
                                    np.arctan2(obs['x11'][0], obs['x4'][0])])
                state_st_0 = state_st_0 + np.random.normal(scale=NOISE[2], size=state_st_0.shape)
                
                # timer.tic()
                reference_traj = infer_env.get_refernece_traj(state_st_0.copy(), target_vel)
                mppi_state, predicted_states, s_opt = mppi.update(mppi_state, infer_env, state_st_0.copy(), jrng.new_key())
                control = mppi_state[0][0]
                # timer.toc('mppi')

                
                predicted_states = jax.device_get(predicted_states[0, :, :, :2])
                sampled_renderer.update(np.concatenate(predicted_states))
                reference_traj_renderer.update(reference_traj)
                opt_traj_renderer.update(s_opt)
                map_waypoint_renderer.update(state_st_0[:2])

                
                env.params['tire_p_dy1'] = friction  # mu_y
                env.params['tire_p_dx1'] = friction  # mu_x
                for i in range(SEGMENT_LENGTH):
                    obs, rew, done, info = env.step(np.array([[control[0] + np.random.normal(scale=NOISE[0]),
                                                               control[1] + np.random.normal(scale=NOISE[1])]]))
                    step_reward += rew
                state_st_1 = np.array([obs['x1'][0], obs['x2'][0], obs['x3'][0], 
                                        np.sqrt(obs['x11'][0] ** 2 + obs['x4'][0] ** 2), 
                                        obs['x5'][0], obs['x6'][0],
                                        np.arctan2(obs['x11'][0], obs['x4'][0])])
                state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                ## x3 = steering angle of front wheels
                ## x4 = velocity in x-direction
                ## x6 = yaw rate
                ## x11 = velocity in y-direction
                
                np.set_printoptions(suppress=True, precision=10)
                
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
                    env.render(mode='human_fast')
                    # print('target_vel', target_vel, np.sqrt(obs['x4'][0] ** 2 + obs['x11'][0] ** 2))
                    # print(ind, f'x {obs["x1"][0]:.2f}, y {obs["x2"][0]:.2f}, yaw {obs["x5"][0]:.2f}, yawrate {obs["x6"][0]:.2f}' + \
                    #     f', vx {obs["x4"][0]:.2f}, vy {obs["x11"][0]:.2f}, steer {obs["x3"][0]:.2f}')

            print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

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