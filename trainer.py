import jax
import jax.numpy as jnp
import optax

from flax.training import checkpoints, train_state


import numpy as np
import functools


import yaml

from nerf import (
    get_model,
    get_loss_func,
)

from utils import get_config
from datasets import dataset_factory



config_path = 'configs/lego.yaml'
config = get_config(config_path)


devices = jax.local_devices()
print(f'Num devices: {len(devices)}')

print(f'Using config:\n{config}')


dataset = dataset_factory(config)

model, params = get_model(config.L_position, config.L_direction)

# create train state
tx = optax.adam(config.learning_rate)
state = train_state.TrainState.create(apply_fn=model, params=params, tx=tx) 
state = jax.device_put_replicated(state, devices)

# create loss function
loss_func = get_loss_func(
    model=model,
    near=config.near,
    far=config.far,
    L_position=config.L_position,
    L_direction=config.L_direction,
    num_samples_coarse=config.num_samples_coarse,
    num_samples_fine=config.num_samples_fine,
    use_hvs=config.use_hvs,
    use_direction=True,
    use_random_noise=True,
)


@functools.partial(jax.pmap, axis_name='batch', in_axes=(None, 0, 0, 0, 0)) 
def train_step( 
    state,
    key,
    origins,
    directions,
    rgbs,
    ):
    '''
    Train step for a single batch of data.
    '''

    def _loss(params):
        return loss_func(params, key, origins, directions, rgbs)

    # compute loss and grads
    (loss, rgbs_pred, weights, ts), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

    # combine grads and loss from all devices
    grads = jax.lax.pmean(grads, 'batch')
    loss = jax.lax.pmean(loss, 'batch')

    # apply updates
    state = state.apply_gradients(grads=grads)

    return state, loss, rgbs_pred, weights, ts 

key = jax.random.PRNGKey(0)
    
for idx, (img, origins, directions) in enumerate(dataset['train']):
    print(f'img shape: {img.shape}')

    key_train = jax.random.split(key, img.shape[0])

    print(f'key_train shape: {key_train.shape}')
    # train step
    state, loss, rgb_pred, weights, ts = train_step(
        state,
        key_train,
        origins,
        directions,
        img,
    )

    print(f'loss: {loss}')


    key = jax.random.split(key, 1)[0]
    

    


