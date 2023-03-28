import pytest

from nerf import (
    get_model, 
    hvs, 
    render, 
    get_points,
    encode_points_nd_directions,
)


import jax
import jax.numpy as jnp


class Config:
    L_position = 10
    L_direction = 4
    batch_size = 32 
    num_samples_coarse = 64
    num_samples_fine = 128

    near = 0.0
    far = 1.0


def test_model():

    model, params = get_model(10, 4)

        
    position = jnp.ones((Config.batch_size, Config.L_position * 6 + 3))
    direction = jnp.ones((Config.batch_size, Config.L_direction * 6 + 3))

    rgb, density = model.apply(params, position, direction)
    
    assert rgb.shape == (Config.batch_size, 3)
    assert density.shape == (Config.batch_size, 1)


def test_hvs():

    weights = jnp.ones((Config.batch_size, Config.num_samples_coarse))
    ts = jnp.ones((Config.batch_size, Config.num_samples_coarse))

    key = jax.random.PRNGKey(0)

    weights_hvs = hvs(weights, ts, Config.num_samples_fine, key)
    
    assert weights_hvs.shape == (Config.batch_size, Config.num_samples_fine + Config.num_samples_coarse)
    
def test_get_points():

    origin = jnp.ones((Config.batch_size, 3))
    direction = jnp.ones((Config.batch_size, 3))
    weights = jnp.ones((Config.batch_size, Config.num_samples_coarse, 1))

    key = jax.random.PRNGKey(0)

    points = get_points(
        key,
        origin,
        direction,
        Config.near,
        Config.far,
        Config.num_samples_coarse,
        Config.num_samples_fine,
        random_sample=False,
        use_hvs=False,
        weights=weights,)

    sample_nd_hvs_points = get_points(
        key,
        origin,
        direction,
        Config.near,
        Config.far,
        Config.num_samples_coarse,
        Config.num_samples_fine,
        random_sample=False,
        use_hvs=True,
        weights=weights,)


    assert points.shape == (Config.batch_size, Config.num_samples_coarse, 3)
    assert sample_nd_hvs_points.shape == (Config.batch_size, Config.num_samples_fine + Config.num_samples_coarse, 3)

def test_encode_points_nd_directions():

    points = jnp.ones((Config.batch_size, Config.num_samples_coarse, 3))
    direction = jnp.ones((Config.batch_size, Config.num_samples_coarse, 3)) 

    encoded_points, encoded_directions = encode_points_nd_directions(points, direction, Config.L_position, Config.L_direction)

    assert encoded_points.shape == (Config.batch_size, Config.num_samples_coarse, Config.L_position * 6 + 3)
    assert encoded_directions.shape == (Config.batch_size, Config.num_samples_coarse, Config.L_direction * 6 + 3)





if __name__ == '__main__':
    pytest.main()


