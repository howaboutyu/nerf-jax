import pytest

from nerf import (
    get_model, 
    hvs, 
    render_fn, 
    get_points,
    encode_points_nd_directions,
    loss_func,
    get_loss_func 
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

@pytest.fixture
def dummy_data():
    #origins = jnp.ones((Config.batch_size, Config.num_samples_coarse, 3))
    #directions = jnp.ones((Config.batch_size, Config.num_samples_coarse, 3))

    origins = jnp.ones((Config.batch_size, 3))
    directions = jnp.ones((Config.batch_size, 3))


    points = jnp.ones((Config.batch_size, Config.num_samples_coarse, 3))
    t = jnp.ones((Config.batch_size, Config.num_samples_coarse))

    weights = jnp.ones((Config.batch_size, Config.num_samples_coarse))



    model, params = get_model(10, 4)


    return model, params, origins, directions, points, t, weights



def test_hvs():

    weights = jnp.ones((Config.batch_size, Config.num_samples_coarse))
    ts = jnp.linspace(0.0, 1.0, Config.num_samples_coarse) 
    ts = jnp.broadcast_to(ts, (Config.batch_size, Config.num_samples_coarse))

    ts_to_sample = jnp.linspace(0.0, 1.0, Config.num_samples_fine )
    ts_to_sample = jnp.broadcast_to(ts_to_sample, (Config.batch_size, Config.num_samples_fine))

    key = jax.random.PRNGKey(0)

    weights_hvs = hvs(weights, ts, ts_to_sample, key)
    
    assert weights_hvs.shape == (Config.batch_size, Config.num_samples_fine + Config.num_samples_coarse)
    
def test_get_points(dummy_data):

    model, params, origins, directions, _, _, weights = dummy_data


    key = jax.random.PRNGKey(0)

    points, t, _, _ = get_points(
        key,
        origins,
        directions,
        Config.near,
        Config.far,
        Config.num_samples_coarse,
        Config.num_samples_fine,
        random_sample=False,
        use_hvs=False,
        weights=weights,)
    


    sample_nd_hvs_points, t_hvs, _, _ = get_points(
        key,
        origins,
        directions,
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



def test_render_fn(dummy_data):
    
    key = jax.random.PRNGKey(0) 
    
    model, params, origins, directions, _, _, weights = dummy_data

    points, t, _, _ = get_points(
        key,
        origins,
        directions,
        Config.near,
        Config.far,
        Config.num_samples_coarse,
        Config.num_samples_fine,
        random_sample=True,
        use_hvs=False,
        weights=weights,)

    
    directions = jnp.broadcast_to(directions[..., jnp.newaxis, :], (Config.batch_size, Config.num_samples_coarse, 3))
    encoded_points, encoded_directions = encode_points_nd_directions(points, directions, Config.L_position, Config.L_direction)


    # use_direction = True, use_random_noise = True 
    rendered, _ = render_fn(
        key=key,
        model_func=model,
        params=params,
        t=t,
        encoded_points=encoded_points,
        encoded_directions=encoded_directions,
        weights=weights,
        use_direction=True,
        use_random_noise=True,
    )
    assert rendered.shape == (Config.batch_size, 3)

    # use_direction = False, use_random_noise = False
    model, params = get_model(10, None)

    rendered, _ = render_fn(
        key=key,
        model_func=model,
        params=params,
        t=t,
        encoded_points=encoded_points,
        encoded_directions=encoded_directions,
        weights=weights,
        use_direction=False,
        use_random_noise=False,
    )
    assert rendered.shape == (Config.batch_size, 3)


def test_loss_fn(dummy_data):

    key = jax.random.PRNGKey(0)
    
    model, params, origins, directions, _, _, _ = dummy_data


    origins = jnp.ones((Config.batch_size, 3))
    directions = jnp.ones((Config.batch_size, 3))

    #loss, (rendered, weights, t) = loss_func(
    #    params=params,
    #    model_func=model,
    #    key=key,
    #    origins=origins,
    #    directions=directions,
    #    near=Config.near,
    #    far=Config.far,
    #    L_position=Config.L_position,
    #    L_direction=Config.L_direction,
    #    num_samples_coarse=Config.num_samples_coarse,
    #    num_samples_fine=Config.num_samples_fine,
    #    expected_rgb=jnp.ones((Config.batch_size, 3)),
    #    use_hvs=True)

    loss_func_got = get_loss_func(
        model,
        Config.near,
        Config.far,
        Config.L_position,
        Config.L_direction,
        Config.num_samples_coarse,
        Config.num_samples_fine,
        use_hvs=True,
        use_direction=True,
        use_random_noise=True,
    )

    loss, (rendered, weights, t) = loss_func_got(
        params=params,
        key=key,
        origins=origins,
        directions=directions,
        expected_rgb=jnp.ones((Config.batch_size, 3)),
    )
        

    assert loss.shape == ()
    assert rendered.shape == (Config.batch_size, 3)
    assert weights.shape == (Config.batch_size, Config.num_samples_fine + Config.num_samples_coarse, 1)

        

if __name__ == '__main__':
    pytest.main()


