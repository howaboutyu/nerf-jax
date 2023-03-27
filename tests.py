import pytest

from nerf import get_model, hvs, render


import jax
import jax.numpy as jnp


class Config:
    L_position = 10
    L_direction = 4
    batch_size = 32 
    num_samples = 64


def test_model():

    model, params = get_model(10, 4)

        
    position = jnp.ones((Config.batch_size, Config.L_position * 6 + 3))
    direction = jnp.ones((Config.batch_size, Config.L_direction * 6 + 3))

    rgb, density = model.apply(params, position, direction)
    
    assert rgb.shape == (Config.batch_size, 3)
    assert density.shape == (Config.batch_size, 1)




def test_hvs():
    
    weights = jnp.ones((Config.batch_size, Config.num_samples, 1))
    ts = jnp.ones((Config.batch_size, Config.num_samples, 1))
    num_samples = Config.num_samples

    key, _ = jax.random.PRNGKey(0)

    weights_hvs, ts_hvs = hvs(weights, ts, num_samples, key)
