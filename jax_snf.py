import os
import matplotlib.pyplot as plt
import numpy as np
from vehicle_data_gen_utils.utils import DataProcessor, ConfigJSON, Logger
import vehicle_data_gen_utils.jax_utils as jax_utils
from trainer_jax import Trainer
import flax.training.train_state as flax_TrainState
import jax
import optax
from flax import linen as nn
from jax import random
import jax.numpy as jnp
from functools import partial
from models.nsf import NeuralSplineFlow
import distrax


CUDANUM = 2
EXP_NAME = '23_' + 'jax_snf_f5_v18_t02_4layer_moresteer'
EXP_NAME = 'fric3_rand_f8'
CONTINUE_TRAINING = 0
class Config():
    exp_name = EXP_NAME
    # savedir = '/home/lucerna/Documents/DATA/results/' + EXP_NAME + '/'
    # datadir = '/home/lucerna/Documents/DATA/tuner_inn/fric3_rand/'
    # test_datadir = '/home/lucerna/Documents/DATA/tuner_inn/fric3_rand_test/'
    # savedir = '/workspace/data/tuner/results2/' + EXP_NAME + '/'
    # datadir = '/workspace/data/tuner/fric3_rand_acc2_t02/'
    # test_datadir = '/workspace/data/tuner/fric3_rand_t02/'
    savedir = './data/'+ EXP_NAME + '/'
    datadir = './data/' + EXP_NAME + '/'
    train_segment = 2
    latent_size = 1
    pe_level = 3
    batchsize = 1000
    # test_batchsize = 2000
    test_batchsize = 1000
    lr = 2e-4
    test_period = 10
    test_perm = 20
    n_dim = 3 * 1 * 2
    n_context = 6 * pe_level * 2
    n_sample = 10
    max_epoch = 3000

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDANUM)

config = Config()

def main():
    from torch.utils.tensorboard import SummaryWriter
    tensorboard = SummaryWriter(config.savedir + 'tensorboard/')
    logger = Logger(config.savedir, EXP_NAME)
    logger.write_file(__file__)

    data = np.load(config.datadir + 'train_data_f5.npz') 
    
    train_states = np.asarray(data['train_states'])[0, :, 0, :]
    arr = np.arange(train_states.shape[0])
    np.random.shuffle(arr)
    arr_train = arr[:int(np.rint(arr.shape[0] * 0.95))]
    arr_test = arr[int(np.rint(arr.shape[0] * 0.95)):]
    train_states = np.asarray(data['train_states'])[0, arr_train, 0, :] #なぜ０？
    train_controls = np.asarray(data['train_controls'])[0, arr_train, 1, :] # np.asarray(data['train_controls'])は(1,66560,2,2)　なぜ１？
    train_dynamics = np.asarray(data['train_dynamics'])[0, arr_train, 0, :]
    print('train_states', train_states.shape) # (63232, 4)

    # data = np.load(config.test_datadir + 'train_data.npz')
    test_states = np.asarray(data['train_states'])[0, arr_test, 0, :]
    test_controls = np.asarray(data['train_controls'])[0, arr_test, 1, :]
    test_dynamics = np.asarray(data['train_dynamics'])[0, arr_test, 0, :]
    print('test_states', test_states.shape) # (3328, 4)

    pe = jax_utils.PositionalEncoding_jax(config.pe_level)
    pe1 = jax_utils.PositionalEncoding_jax(1)

    data = jnp.concatenate([train_states, train_controls, train_dynamics], axis=1) # (63232, 4+2+3=9)
    context = pe.batch_encode(data[:, :6])
    dyna = pe1.batch_encode(data[:, 6:9])
    test_data = jnp.concatenate([test_states, test_controls, test_dynamics], axis=1) # (3328, 4+2+3=9)
    test_context = pe.batch_encode(test_data[:, :6])
    test_dyna = pe1.batch_encode(test_data[:, 6:9])
    print(test_context.shape) # (3328, 36)
    print(test_dyna.shape) # (3328, 6)
    print(context.shape) # (63232, 36)
    print(dyna.shape) # (63232, 6)

    dp = DataProcessor()
    c = ConfigJSON()
    c.load_file(config.datadir + 'config_f5.json')
    c.save_file(config.savedir + 'config.json')
    normalization_param = np.array(c.d['normalization_param']).T
    dyna_normalization_param = normalization_param[:, 4:7]
    
    jrng = jax_utils.oneLineJaxRNG(0)
    train_perms = jax_utils.generate_perms(jrng.new_key(), train_states.shape[0], config.batchsize)
    test_perms = jax_utils.generate_perms(jrng.new_key(), test_states.shape[0], config.test_batchsize)
    

    model = NeuralSplineFlow(n_dim=config.n_dim, n_context=config.n_context, 
                         hidden_dims=[128, 128], n_transforms=4, activation="relu", n_bins=3)
    x_init = jnp.zeros((config.batchsize, config.n_dim))
    x_context = jnp.zeros((config.batchsize, config.n_context))
    params = model.init(jrng.new_key(), x_init, x_context)
    # opt_state = optimizer.init(params)
    dist = distrax.MultivariateNormalDiag(jnp.zeros(6), jnp.ones(6)/5)
    
    flax_train_state = flax_TrainState.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(8), 
                       optax.adam(learning_rate=config.lr)),
    )
    
    @jax.jit
    def train_step(state, x, context):
        def loss_fn(params):
            log_prob = model.apply(
                params, x, context
            )
            loss = -log_prob.mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    @jax.jit
    def test_step(state, rng_key, context):
        context_batch = context[None, :, :].repeat(config.n_sample, 0).reshape(-1, context.shape[-1])
        z = dist.sample(seed=rng_key, sample_shape=(context_batch.shape[0],))
        samples = model.apply(state.params, z, context_batch, method=model.sample)
        samples = pe.batch_decode2(samples)
        samples = samples.reshape(config.n_sample, -1, samples.shape[-1])
        samples_mean = samples.mean(axis=0)
        return samples_mean

    def model_merit_fn(info):
        return info[1]
    trainer = Trainer(EXP_NAME, config.savedir,
                    max_epoch=config.max_epoch, best_fn=model_merit_fn,
                    info_template=np.zeros(8), initial_lr=config.lr)
    np.set_printoptions(suppress=True, precision=6)
    epoch_info = np.zeros(8)
    
    test_perms = jax_utils.generate_perms(jrng.new_key(), test_states.shape[0], config.test_batchsize)
    while(not trainer.is_done()):
        epoch, epoch_info[0] = trainer.epoch, trainer.epoch
        train_perms = jax_utils.generate_perms(jrng.new_key(), train_states.shape[0], config.batchsize)
        
        
        for perm in train_perms:
            flax_train_state, loss = train_step(flax_train_state, dyna[perm], context[perm])
            epoch_info[1] += jax.device_get(loss)
        epoch_info[1] /= len(train_perms)
        print(epoch, epoch_info[1])
        logger.log_line(str(epoch) + ' ' + str(epoch_info[1]))
        
        params = flax_train_state.params
        
        if epoch % config.test_period == 0 and epoch > 0:
            flax_train_state, epoch_info = trainer.load_state(flax_train_state, epoch_info, save_name='last')
            params = flax_train_state.params
            cnt = 0
            for perm in test_perms:
                samples_mean = test_step(flax_train_state, jrng.new_key(), test_context[perm, :])
                # samples = samples.reshape(config.n_sample, -1, samples.shape[-1])
                # atan2_value = jax.device_get(atan2_value).copy()
                # sub_zero_inds = np.where(atan2_value < 0)
                # atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
                # samples_mean = samples.mean(axis=0)
                
                denormalized_samples = dp.de_normalize(samples_mean, dyna_normalization_param)
                denormalized_test_dyna = dp.de_normalize(test_dynamics[perm], dyna_normalization_param)
                print(denormalized_samples[0], denormalized_test_dyna[0])
                error = jnp.abs(denormalized_samples - denormalized_test_dyna).mean(axis=0)
                print('error', error)
                
                # error = jnp.abs(samples_mean - test_dyna[perm]).mean(axis=0)
                epoch_info[2:5] += jax.device_get(error)
                cnt += 1
                # if cnt == config.test_perm:
                #     break
            epoch_info[2:8] /= config.test_perm #これって前のステップでのエラーも残ってない？？epoch_info = np.zeros(8)をwhileの中で最初にやるべきでは？
            logger.log_line(str(epoch) + ' ' + str(epoch_info[2:5]) + ' ' + 'test')
            print(epoch, epoch_info[2:5], 'test')
            
        return_text = trainer.step(flax_train_state, epoch_info)
        # flax_train_state, epoch_info = trainer.load_state(flax_train_state, epoch_info, save_name='best')
        
if __name__ == '__main__':
    main()

