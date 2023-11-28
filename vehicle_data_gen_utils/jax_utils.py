import flax
import jax
from jax import random
import numpy as np
import jax.numpy as jnp
from flax.training import orbax_utils
import orbax
from functools import partial
import warnings

def load_state(state, info, path=""):
    print(path)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {'state': state, 'info': info}
    load_filename = path
    state_restored = orbax_checkpointer.restore(load_filename, item=ckpt)
    print('Load from model: ', load_filename)
    return state_restored['state'], state_restored['info'].copy()

class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
        
    def new_key(self):
        self.rng, key = random.split(self.rng)
        return key
        
        
def generate_perms(rng_key, data_length, batch_size):
    perms = jax.random.permutation(rng_key, data_length)
    steps_per_epoch = data_length//batch_size
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    return perms


class PositionalEncoding_jax():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = jnp.array(self.val_list)

    def encode(self, x):
        return jnp.sin(self.val_list * jnp.pi * x), jnp.cos(self.val_list * jnp.pi * x)

    def encode_even(self, x):
        return jnp.sin(self.val_list * jnp.pi * 2 * x), jnp.cos(self.val_list * jnp.pi * 2 * x)
    
    @partial(jax.jit, static_argnums=(0,2))
    def batch_encode(self, batch, loop_ind=1):
        batch_encoded_list = []
        for ind in range(batch.shape[loop_ind]):
            encoded_ = self.encode(batch[:, ind, None])
            batch_encoded_list.append(encoded_[0])
            batch_encoded_list.append(encoded_[1])
        batch_encoded = jnp.stack(batch_encoded_list)
        batch_encoded = batch_encoded.transpose(1, 2, 0).reshape((batch_encoded.shape[1], 
                                                                  self.L * batch_encoded.shape[0]))
        return batch_encoded
    
    def decode(self, sin_value, cos_value):
        atan2_value = jnp.arctan2(sin_value, cos_value) / (jnp.pi)
        if jnp.isscalar(atan2_value) == 1:
            if atan2_value > 0:
                return atan2_value
            else:
                return 1 + atan2_value
        else:
            atan2_value[jnp.where(atan2_value < 0)] = atan2_value[jnp.where(atan2_value < 0)] + 1
            return atan2_value
        
    def decode_even(self, sin_value, cos_value):
        atan2_value = jnp.arctan2(sin_value, cos_value) / jnp.pi/2
        if jnp.isscalar(atan2_value) == 1:
            if atan2_value < 0:
                atan2_value = 1 + atan2_value
            if jnp.abs(atan2_value - 1) < 0.001:
                atan2_value = 0
        else:
            atan2_value[jnp.where(atan2_value < 0)] = atan2_value[jnp.where(atan2_value < 0)] + 1
            atan2_value[jnp.where(jnp.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value

    def batch_decode(self, sin_value, cos_value):
        atan2_value = jnp.arctan2(sin_value, cos_value) / (jnp.pi)
        sub_zero_inds = jnp.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        return atan2_value
    
    # @partial(jax.jit, static_argnums=(0))
    def batch_decode2(self, list_data):
        dim = int(list_data.shape[1]/2)
        list_data = list_data.reshape(list_data.shape[0], dim, 2)
        atan2_value = jnp.arctan2(list_data[:, :, 0], list_data[:, :, 1]) / (np.pi)
        # sub_zero_inds = jnp.where(atan2_value < 0)
        # atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        return atan2_value


