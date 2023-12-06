import os
import sys
sys.path.append("../")
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from vehicle_data_gen_utils.utils import DataProcessor, ConfigJSON, Logger
import vehicle_data_gen_utils.utils as utils
import vehicle_data_gen_utils.jax_utils as jax_utils
import flax.training.train_state as flax_TrainState
import jax
import optax
from flax import linen as nn
import distrax
from models.nsf import NeuralSplineFlow
import time
from functools import partial
from dynamics_models.dynamics_models import vehicle_dynamics_st, vehicle_dynamics_mb, reset_mb
from numba import njit


CUDANUM = 0
# EXP_NAME = 'map41_f5'
EXP_NAME = 'st_mppi_f11_v5'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDANUM)

class Config(ConfigJSON):
    exp_name = EXP_NAME
    # savedir = '/home/lucerna/MEGA/Reasearch/tuner_inn/tuner_inn/results/' + EXP_NAME + '/'
    # datadir = '/home/lucerna/Documents/DATA/tuner_inn/fric3_rand/'
    # test_datadir = '/home/lucerna/Documents/DATA/tuner_inn/fric3_rand_test/'
    savedir = '/home/tomas/Documents/chrono-TunerINN/data/' + EXP_NAME + '/'
    datadir = '/home/tomas/Documents/chrono-TunerINN/data/' + EXP_NAME + '/'
    test_datadir = '/home/tomas/Documents/chrono-TunerINN/data/' + EXP_NAME + '/'
    train_segment = 2
    latent_size = 1
    pe_level = 3
    batchsize = 2000
    lr = 5e-4
    test_period = 10
    n_dim = 3 * 1 * 2
    n_context = 6 * 3 * 2
    n_sample = 20
    pre_mode = False
    
    hidden_dims = [128, 128]
    n_transforms = 4
    n_bins = 3
            

class JaxInfer():

    def __init__(self) -> None:         
        config = Config()
        self.pe = jax_utils.PositionalEncoding_jax(config.pe_level)
        self.pe1 = jax_utils.PositionalEncoding_jax(1)
        self.dp = DataProcessor()
        config.load_file(config.savedir + 'config.json')
        self.normalization_param = jnp.array(config.d['normalization_param']).T
        
        self.jrng = jax_utils.oneLineJaxRNG(0)
        self.model = NeuralSplineFlow(n_dim=config.n_dim, n_context=config.n_context, 
                                hidden_dims=config.hidden_dims, 
                                n_transforms=config.n_transforms, activation="relu", 
                                n_bins=config.n_bins)
        x_init = jnp.zeros((config.batchsize, config.n_dim))
        x_context = jnp.zeros((config.batchsize, config.n_context))
        self.sample_dist = distrax.MultivariateNormalDiag(jnp.zeros(6), jnp.ones(6)/5)
        self.params = self.model.init(self.jrng.new_key(), x_init, x_context)
        
        flax_train_state = flax_TrainState.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=optax.chain(optax.clip_by_global_norm(8), 
                        optax.adam(learning_rate=config.lr)),
        )
        
        epoch_info = jnp.zeros(5)
        load_filename = config.savedir + config.exp_name + '_model_' + 'last'
        flax_train_state, epoch_info = jax_utils.load_state(flax_train_state, epoch_info, load_filename)
        print(epoch_info)
        self.params = flax_train_state.params
        self.config = config
        self.pre_context = self.pe.batch_encode(jnp.concatenate([np.zeros((4,)), np.ones((2,)) * 0.5])[None, :])
        
    @partial(jax.jit, static_argnums=(0,))
    def inference(self, state, control):
        
        def test_step(rng_key, context):
            context_batch = context[None, :, :].repeat(self.config.n_sample, 0).reshape(-1, context.shape[-1])
            z = self.sample_dist.sample(seed=rng_key, sample_shape=(context_batch.shape[0],))
            samples = self.model.apply(self.params, z, context_batch, method=self.model.sample)
            samples = self.pe.batch_decode2(samples)
            samples = samples.reshape(self.config.n_sample, -1, samples.shape[-1])
            samples_mean = samples.mean(axis=0)
            return samples_mean, samples
        
        state_normal = self.dp.runtime_normalize(state.copy(), self.normalization_param[:, :4])
        control_normal = self.dp.runtime_normalize(control.copy(), self.normalization_param[:, 7:9])
        # print('state_normal', state_normal, 'control_normal', control_normal)
        test_context = self.pe.batch_encode(jnp.concatenate([state_normal, control_normal])[None, :])
        
        # print('state_normal', state_normal.shape)
        # print('control_normal', control_normal.shape)
        # print(test_context.shape)
        if self.config.pre_mode:
            context_input = jnp.concatenate([self.pre_context, test_context], axis=1)
        else:
            context_input = test_context
        samples_mean, samples = test_step(self.jrng.new_key(), context_input)
        denormalized_sample_mean = self.dp.de_normalize(samples_mean, self.normalization_param[:, 4:7])
        if self.config.pre_mode:
            self.pre_context = test_context.copy()
        return denormalized_sample_mean, jnp.sum(jnp.var(samples, axis=0))



class InferEnv():
    def __init__(self, waypoints, n_steps, mode='st', DT=0.2) -> None:
        self.a_shape = 2
        self.waypoints = np.array(waypoints)
        self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.n_steps = n_steps
        self.reference = None
        self.DT = DT
        config = Config()
        config.load_file(config.savedir + 'config.json')
        self.normalization_param = jnp.array(config.d['normalization_param']).T
        self.mode = mode
        if mode == 'st':
            def update_fn(x, u):
                x1 = x.copy()
                def step_fn(i, x0):
                    return x0 + vehicle_dynamics_st(x0, u) * 0.02
                x1 = jax.lax.fori_loop(0, int(self.DT/0.02), step_fn, x1)
                # for _ in range(int(self.DT/0.01)):
                #     dynamics = vehicle_dynamics_st(x1, u)
                #     print('before')
                #     x1 = step_fn(0, x1)
                #     print('x1', x1)
                return (x1, 0, x1-x)
            self.update_fn = update_fn
        if mode == 'mb':
            def update_fn(x, u):
                x1 = x.copy()
                def step_fn(i, x0):
                    return x0 + vehicle_dynamics_mb(x0, u) * 0.001
                x1 = jax.lax.fori_loop(0, int(self.DT/0.001), step_fn, x1)
                return (x1, 0, x1-x)
            self.update_fn = update_fn
        elif mode == 'nf':
            self.jax_env = JaxInfer()
            self.update_fn = lambda x, u: self.infer2nf_update(x, u, self.jax_env.inference(self.state_nf2infer(x), u))
    
    
    def state_nf2infer(self, mb_state):
        return jnp.array([mb_state[2], mb_state[3], mb_state[5], mb_state[6]])
    
    
    @partial(jax.jit, static_argnums=(0,))    
    def infer2nf_update(self, nf_state, u, mb_dyna_ret):
        mb_dyna = mb_dyna_ret[0]
        mb_dyna_var = mb_dyna_ret[1]
        vx = nf_state[3]
        yawrate = nf_state[5]
        vy = nf_state[6]
        
        vx_new = vx + mb_dyna[0, 0] * self.DT
        yawrate_new = yawrate + mb_dyna[0, 1] * self.DT
        vy_new = vy + mb_dyna[0, 2] * self.DT
        
        beta = jnp.arctan(nf_state[6] / nf_state[3])
        vel = jnp.sqrt(nf_state[3] ** 2 + nf_state[6] ** 2)
        
        beta_new = jnp.arctan(vy_new / vx_new)
        vel_new = jnp.sqrt(vx_new ** 2 + vy_new ** 2)
        
        

        # new_state = jnp.array([nf_state[0] + jnp.cos(beta + nf_state[4]) * vel * self.DT, 
        #                        nf_state[1] + jnp.sin(beta + nf_state[4]) * vel * self.DT, 
        #                        nf_state[2] + u[0] * self.DT,
        #                        vx_new,
        #                        nf_state[4] + yawrate * self.DT,
        #                        yawrate_new, 
        #                        vy_new])
        
        # yaw_new = nf_state[4] + yawrate * self.DT
        # new_state = jnp.array([nf_state[0] + jnp.cos(beta + nf_state[4]) * vel * self.DT + \
        #                             jnp.cos(beta_new + yaw_new - (beta + nf_state[4])) * jnp.sqrt((mb_dyna[0, 0] * self.DT) ** 2 + (mb_dyna[0, 2] * self.DT) ** 2) * self.DT / 2, 
        #                         nf_state[1] + jnp.sin(beta + nf_state[4]) * vel * self.DT + \
        #                             jnp.sin(beta_new + yaw_new - (beta + nf_state[4])) * jnp.sqrt((mb_dyna[0, 0] * self.DT) ** 2 + (mb_dyna[0, 2] * self.DT) ** 2) * self.DT / 2, 
        #                         nf_state[2] + u[0] * self.DT,
        #                         vx_new,
        #                         yaw_new,
        #                         nf_state[4] + yawrate * self.DT, 
        #                         vy_new])
        
        yaw_new = nf_state[4] + yawrate * self.DT + (mb_dyna[0, 1] * self.DT) * self.DT / 2
        new_state = jnp.array([nf_state[0] + jnp.cos(beta + nf_state[4]) * vel * self.DT / 2 + \
                                    jnp.cos(beta_new + yaw_new) * jnp.sqrt(vx_new ** 2 + vy_new ** 2) * self.DT / 2, 
                                nf_state[1] + jnp.sin(beta + nf_state[4]) * vel * self.DT / 2 + \
                                    jnp.sin(beta_new + yaw_new) * jnp.sqrt(vx_new ** 2 + vy_new ** 2) * self.DT / 2, 
                                nf_state[2] + u[0] * self.DT,
                                vx_new,
                                yaw_new,
                                yawrate_new, 
                                vy_new])
        return new_state, mb_dyna_var, mb_dyna  

            
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, x, u):
        return self.update_fn(x, u * self.normalization_param[0, 7:9]/2)
        # return self.update_fn(x, u)
    
    
    # @partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, s, reference):
        xy_cost = -jnp.linalg.norm(reference[1:, :2] - s[:, :2], axis=1)
        # # xy_cost = xy_cost.at[0].set(0)
        # # if self.mode == 'nf':
        # #     slip_cost = -jnp.abs(jnp.arctan(s[:, 6] / s[:, 3]))
        # # else:
        # #     slip_cost = 0
            
        # # xy_cost = jnp.sqrt((reference[1:, 0] - s[:, 0]) ** 2 + (reference[1:, 1] - s[:, 1]) ** 2)
        # vel_cost = -jnp.linalg.norm(reference[1:, 2] - s[:, 3])
        # yaw_cost = -jnp.abs(jnp.sin(self.reference[1:, 3]) - jnp.sin(s[:, 4])) - \
        #     jnp.abs(jnp.cos(self.reference[1:, 3]) - jnp.cos(s[:, 4]))
            
        # return xy_cost + yaw_cost
        # yaw_cost = yaw_cost.at[0].set(0)
        # print('xy_cost', xy_cost)
        # print('yaw_cost', yaw_cost)
        return xy_cost
        # state = s[:, [0,1,3,4]]
        # cost = -jnp.linalg.norm(state - reference[1:, :4], axis=1)**2
        # print('cost', cost.shape)
        # return cost
    
    
    # @partial(jax.jit, static_argnums=(0,))
    def reward(self, x):
        return 0
    
    
    def get_refernece_traj(self, state, target_speed=None, vind=5, speed_factor=1.0):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            speed = self.waypoints[ind, vind] * speed_factor
            # speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        speeds = np.ones(self.n_steps) * speed
        
        reference = get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(self.n_steps),
                                            self.waypoints_distances.copy(), DT=self.DT)
        orientation = state[4]
        reference[3, :][reference[3, :] - orientation > 5] = np.abs(
            reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = np.abs(
            reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind

    
    def state_st2infer(self, st_state):
        return jnp.array([st_state[2], 
                        st_state[3] * jnp.cos(st_state[6]),
                        st_state[5],
                        st_state[3] * jnp.sin(st_state[6])])
        
    
    def state_st2nf(self, st_state):
        return np.array([st_state[0], st_state[1], st_state[2],
                        st_state[3] * np.cos(st_state[6]),
                        st_state[4], st_state[5],
                        st_state[3] * np.sin(st_state[6])])
        
    
    def state_nf2st(self, nf_state):
        return np.array([nf_state[0], nf_state[1], nf_state[2],
                        np.sqrt(nf_state[3] ** 2 + nf_state[6] ** 2),
                        nf_state[4], nf_state[5],
                        np.arctan2(nf_state[6], nf_state[3])])
        
    def state_mb2st(self, mb_state):
        return np.array([mb_state[0], mb_state[1], mb_state[2],
                        np.sqrt(mb_state[3] ** 2 + mb_state[10] ** 2),
                        mb_state[4], mb_state[5],
                        np.arctan2(mb_state[10], mb_state[3])])
        
        
    def state_mb2nf(self, mb_state):
        return np.array([mb_state[0], mb_state[1], mb_state[2],
                        mb_state[3], mb_state[4], mb_state[5],
                        mb_state[10]])
        
    
    def state_nf2mb(self, mb_state, nf_state):
        mb_state[0:6] = nf_state[0:6]
        mb_state[10] = nf_state[6]
        return mb_state
        
    
    
    # @partial(jax.jit, static_argnums=(0,))    
    def mb2st_update(self, st_state, u, mb_dyna_ret):
        mb_dyna = mb_dyna_ret[0]
        mb_dyna_var = mb_dyna_ret[1]
        mb_state = self.state_st2infer(st_state)
        # steer = mb_state[0]
        vx = mb_state[1]
        yawrate = mb_state[2]
        vy = mb_state[3]
        vx_new = vx + mb_dyna[0, 0] * self.DT
        yawrate_new = yawrate + mb_dyna[0, 1] * self.DT
        vy_new = vy + mb_dyna[0, 2] * self.DT

        new_state = jnp.array([st_state[0] + st_state[3] * jnp.cos(st_state[6] + st_state[4]) * self.DT, 
                               st_state[1] + st_state[3] * jnp.sin(st_state[6] + st_state[4]) * self.DT, 
                               st_state[2] + u[0] * self.DT, 
                               jnp.sqrt(vx_new ** 2 + vy_new ** 2),
                               st_state[4] + yawrate * self.DT,
                               yawrate_new, 
                               jnp.arctan(vy_new/vx_new)])
                            #    jnp.arctan2(vy_new, vx_new)])
        return new_state, mb_dyna_var, mb_dyna
        
        
            
        
    

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


# @njit(cache=True)
def get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    s_relative = np.zeros((n_steps + 1,))
    s_relative[0] = dist_from_segment_start
    s_relative[1:] = predicted_speeds * DT
    s_relative = np.cumsum(s_relative)

    waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

    index_relative = np.int_(np.ones((n_steps + 1,)))
    for i in range(n_steps + 1):
        index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
    index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

    segment_part = s_relative - (
            waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

    t = (segment_part / waypoints_distances[index_absolute])
    # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

    position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                        waypoints[index_absolute][:, (1, 2)])
    orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                            waypoints[index_absolute][:, 3])
    speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                    waypoints[index_absolute][:, 5])

    interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
    interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
    interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
    interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
    
    reference = np.array([
        # Sort reference trajectory so the order of reference match the order of the states
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speeds,
        interpolated_orientations,
        # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
        np.zeros(len(interpolated_speeds)),
        np.zeros(len(interpolated_speeds)),
        np.zeros(len(interpolated_speeds))
    ])
    return reference
    

