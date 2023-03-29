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

dataset = dataset_factory(config)

model, params = get_model(config.L_position, config.L_direction)

# create train state
tx = optax.adam(config.learning_rate)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx)

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


@functools.partial(jax.pmap, axis_name='batch')
def train_step( 
    params, 
    model,
    optimizer, 
    batch):
    pass

