import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import checkpoints, train_state
import optax
import functools

from typing import Any


def encoding_func(x, L):
    """Encoding function for the input
    Inputs:
      x : input to be encoded
      L : number of frequencies to encode
    Outputs:
      encoded_array : encoded array
    """

    encoded_array = [x]
    for i in range(L):
        encoded_array.extend(
            [jnp.sin(2.0**i * jnp.pi * x), jnp.cos(2.0**i * jnp.pi * x)]
        )
    return jnp.concatenate(encoded_array, -1)


def hvs(weights, t, t_to_sample, key):
    """Hierarchical volume sampling
    Inputs:
      weights : weights of the distribution to sample from, shape (batch_size, num_samples)
      t : points to sample from the distribution, shape (batch_size, N_c )
      t_to_sample: points to sample from the distribution, shape (batch_size, N_f)
      key : random key
    Outputs:
      sampled_t : sampled points from the distribution
    """

    # find cdf
    weights = jnp.squeeze(weights) + 1e-10
    norm = jnp.sum(weights, -1)

    pdf = weights / norm[..., jnp.newaxis]

    cdf = jnp.cumsum(pdf, -1)

    def inverse_sample(Z, cdf, t_to_sample):
        # Sample from the inverse CDF using inverse transform sampling method
        # Inputs:
        #   Z : numbers from a uniform distribution U(0,1)
        #   cdf : CDF of the distribution to sample from
        #   t_to_sample : points to sample from the distribution
        # Outputs:
        #   sampled_t : sampled points from the distribution

        abs_diff = jnp.abs(cdf[..., jnp.newaxis, :] - Z[..., jnp.newaxis])

        argmin = jnp.argmin(abs_diff, 1)

        # expand argmin to match the shape of t_to_sample

        sampled_t = jnp.take_along_axis(t_to_sample, argmin, 1)

        return sampled_t

    # t with hvs
    t_hvs = [t]

    # we want argmin to have ids ranging from 0 to num_to_sample
    num_to_sample = t_to_sample.shape[1]
    num_loops = int(num_to_sample / t.shape[1])

    # loop over num_loops until we have num_to_sample samples
    for i in range(num_loops):
        Z = jax.random.uniform(key, (weights.shape[0], num_to_sample))

        key, _ = jax.random.split(key)

        sampled_t = inverse_sample(Z, cdf, t_to_sample)
        t_hvs.append(sampled_t)

    t_hvs = jnp.concatenate(t_hvs, -1)

    # sort the t_hvs
    t_hvs = jnp.sort(t_hvs, -1)

    return t_hvs


def get_points(
    key,
    origin,
    direction,
    near,
    far,
    num_coarse_samples,
    num_fine_samples,
    random_sample,
    use_hvs,
    weights,
):
    """
    Get points to render
    Inputs:
        key : random key
        origin : origin of the rays, [batch_size, 3]
        direction : direction of the rays, [batch_size, 3]
        near : near plane
        far : far plane
        num_coarse_samples : number of coarse samples
        num_fine_samples : number of fine samples
        random_sample: sample randomly
        use_hvs : use hierarchical volume sampling
        weights : weights of the distribution to sample from
    Outputs:
        points : points to render, [batch_size, num_samples, 3]
        t : t values of the points, [batch_size, num_samples]
        origins : origins of the rays, [batch_size, num_samples, 3]
        directions : directions of the rays, [batch_size, num_samples, 3]
    """

    batch_size = origin.shape[0]

    origin = origin[..., jnp.newaxis, :]
    direction = direction[..., jnp.newaxis, :]

    t = jnp.linspace(near, far, num_coarse_samples)

    origins = jnp.broadcast_to(origin, (batch_size, num_coarse_samples, 3))
    directions = jnp.broadcast_to(direction, (batch_size, num_coarse_samples, 3))
    t = jnp.broadcast_to(t, (batch_size, num_coarse_samples))

    if random_sample:
        random_shift = (
            jax.random.uniform(key, (batch_size, num_coarse_samples))
            * (far - near)
            / num_coarse_samples
        )
        t = t + random_shift

    elif use_hvs:
        t_to_sample = jnp.linspace(near, far, num_fine_samples)
        t_to_sample = jnp.broadcast_to(t_to_sample, (batch_size, num_fine_samples))

        t = hvs(weights, t, t_to_sample, key)
        t = jax.lax.stop_gradient(t)

    # t has shape [batch_size, num_samples (either N_c or N_c + N_f)]
    # direction and origin has shapes [batch_size, 3]

    # points = origin[..., jnp.newaxis, :] + t[..., jnp.newaxis] * direction[..., jnp.newaxis, :]
    points = origin + t[..., jnp.newaxis] * direction

    if use_hvs:
        points = jax.lax.stop_gradient(points)

        origins = jnp.broadcast_to(
            origin, (batch_size, num_fine_samples + num_coarse_samples, 3)
        )
        directions = jnp.broadcast_to(
            direction, (batch_size, num_fine_samples + num_coarse_samples, 3)
        )
        t = jnp.broadcast_to(t, (batch_size, num_fine_samples + num_coarse_samples))

    return points, t, origins, directions


def encode_points_nd_directions(points, directions, L_position, L_direction):
    """
    Encode points and directions
    Inputs:
        points : points to encode, [batch_size, num_samples, 3]
        directions : directions to encode, [batch_size, num_samples, 3]
        L_position : number of frequencies to encode for points
        L_direction : number of frequencies to encode for directions
    Outputs:
        encoded_points : encoded points, [batch_size, num_samples, 3*(L_position+1)]
        encoded_directions: encoded directions, [batch_size, num_samples, 3*(L_direction+1)]
    """

    encoded_points = encoding_func(points, L_position)
    encoded_directions = encoding_func(directions, L_direction)

    return encoded_points, encoded_directions


def render_fn(
    key,
    model_func,
    params,
    t,
    encoded_points,
    encoded_directions,
    use_direction,
    use_random_noise,
):
    """
    Render function
    Inputs:
        key : random key
        model_func : model function
        params : model parameters
        t : points to render, [batch_size, num_samples]
        encoded_points : encoded points, [batch_size, num_samples, 3*(L_position+1)]
        encoded_directions : encoded directions, [batch_size, num_samples, 3*(L_direction+1)]
        weights : weights of the distribution to sample from
        use_direction : use directions
        use_random_noise : use random noise
    Outputs:
        rgb : rgb values, [batch_size, num_samples, 3]
        density : density values, [batch_size, num_samples]
    """

    if use_direction:
        rgb, density = model_func.apply(params, encoded_points, encoded_directions)
    else:
        rgb, density = model_func.apply(params, encoded_points)

    if use_random_noise:
        density = density + jax.random.normal(key, density.shape, dtype=density.dtype)

    rgb = jax.nn.sigmoid(rgb)
    density = jax.nn.relu(density)

    t_delta = t[..., 1:] - t[..., :-1]

    # t_detla shape : [batch_size, N-1]
    # so we need to add a column of 1e10 to the right to make it [batch_size, N]

    # create a column of 1e10 with shape [batch_size, 1]
    zero_column = jnp.broadcast_to(jnp.array([1e10]), [t.shape[0], 1])
    t_delta = jnp.concatenate([t_delta, zero_column], -1)

    # volume rendering
    T_i = jnp.cumsum(jnp.squeeze(density) * t_delta, -1)
    T_i = jnp.insert(T_i, 0, jnp.zeros_like(T_i[..., 0]), -1)
    T_i = jnp.exp(-T_i)[..., :-1]

    # weights = T_i * a_i
    a_i = 1.0 - jnp.exp(-density * t_delta[..., jnp.newaxis])
    weights = T_i[..., jnp.newaxis] * a_i

    c_array = weights * rgb
    c_sum = jnp.sum(c_array, -2)

    return c_sum, weights


def get_model(L_position, L_direction):
    """This function returns a model with the given number of frequencies for the position and direction encodings
    Inputs:
        L_position : number of frequencies to encode for points
        L_direction : number of frequencies to encode for directions
    Outputs:
        model : flax model
        params : model parameters
    """

    class Model(nn.Module):
        @nn.compact
        def __call__(self, position, direction=None):
            x = position
            for i in range(7):
                x = nn.Dense(256, name=f"fc{i + 1}")(x)
                x = nn.relu(x)

                # Concat x with original input
                if i == 4:
                    x = jnp.concatenate([x, position], -1)

            x = nn.Dense(256, name=f"fc{8}_linear")(x)

            vol_density = nn.Dense(1, name=f"fc{8}_sigmoid")(x)

            # Concat direction information after the volume density
            if L_direction:
                x = jnp.concatenate([x, direction], -1)

            x = nn.Dense(128, name="fc_128")(x)
            x = nn.relu(x)
            x = nn.Dense(3, name="fc_f")(x)

            # rgb color is between 0 and 1
            rgb = x
            return rgb, vol_density

    model = Model()
    if not L_direction:
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, L_position * 6 + 3)))
    else:
        params = model.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, L_position * 6 + 3)),
            jnp.ones((1, L_direction * 6 + 3)),
        )
        # params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1, L_position * 6 + 3)), jnp.ones((1, 1, L_direction * 6 + 3)))

    return model, params


def nerf(
    params,
    key,
    origins,
    directions,
    model_func,
    near,
    far,
    L_position,
    L_direction,
    num_samples_coarse,
    num_samples_fine,
    use_hvs,
    use_direction=True,
    use_random_noise=True,
):
    """This function implements neural radiance fields (NeRF) algorithm for rendering.

    Inputs:
        params : model parameters
        key : random key
        origins : origin of rays, shape (batch_size, 3)
        directions : direction of rays, shape (batch_size, 3)
        model_func : function that maps encoded points and directions to rgb and density
        near : near plane of the camera
        far : far plane of the camera
        L_position : number of frequencies to encode for points
        L_direction : number of frequencies to encode for directions
        num_samples_coarse : number of coarse samples for rendering
        num_samples_fine : number of fine samples for rendering if use_hvs is True
        use_hvs : whether to use hierarchical volume sampling for rendering
        use_direction : whether to use directions for encoding and rendering
        use_random_noise : whether to add random noise to density for rendering
    Outputs:
        If use_hvs is True, returns a tuple of 2 tensors:
        - rendered: first element is from coarse rendering, second element is from fine rendering
        - weights_hvs: weights of the distribution at coarse and fine samples
        - t_hvs: t-values at coarse and fine samples

        If use_hvs is False, returns a tuple of 2 tensors:
        - rendered: both elements are from coarse rendering
        - weights_coarse: weights of the distribution at coarse samples
        - t: t-values at coarse samples
    """

    # get points for coarse
    points, t, origins_ray, directions_ray = get_points(
        key,
        origins,
        directions,
        near,
        far,
        num_samples_coarse,
        num_samples_fine,
        random_sample=True,
        weights=None,
        use_hvs=False,
    )

    # encode poitns and directions
    encoded_points, encoded_directions = encode_points_nd_directions(
        points, directions_ray, L_position, L_direction
    )

    # render coarse
    rendered, weights_coarse = render_fn(
        key,
        model_func,
        params,
        t,
        encoded_points,
        encoded_directions,
        use_direction=use_direction,
        use_random_noise=use_random_noise,
    )

    if use_hvs:
        # get points using hvs
        # which has num_samples_coarse + num_samples_fine points
        points_hvs, t_hvs, origins_hvs_ray, directions_hvs_ray = get_points(
            key,
            origins,
            directions,
            near,
            far,
            num_samples_coarse,
            num_samples_fine,
            random_sample=False,
            use_hvs=True,
            weights=weights_coarse,
        )

        # encode poitns and directions
        encoded_points_hvs, encoded_directions_hvs = encode_points_nd_directions(
            points_hvs, directions_hvs_ray, L_position, L_direction
        )

        rendered_hvs, weights_hvs = render_fn(
            key,
            model_func,
            params,
            t_hvs,
            encoded_points_hvs,
            encoded_directions_hvs,
            use_direction=use_direction,
            use_random_noise=use_random_noise,
        )

        return (rendered, rendered_hvs), weights_hvs, t_hvs
    else:
        return (rendered, rendered), weights_coarse, t


def get_nerf(
    near,
    far,
    L_position,
    L_direction,
    num_samples_coarse,
    num_samples_fine,
    use_hvs,
    use_direction=True,
    use_random_noise=True,
):
    """
    Returns a partial function `nerf_specific` that renders a scene using a NeRF model
    with the given parameters.

    Inputs:
        near (float): The distance to the near clipping plane.
        far (float): The distance to the far clipping plane.
        L_position (int): The number of frequencies to encode for each position coordinate.
        L_direction (int): The number of frequencies to encode for each direction coordinate.
        num_samples_coarse (int): The number of coarse samples to use for rendering.
        num_samples_fine (int): The number of fine samples to use for rendering, if hierarchical
            volume sampling (HVS) is enabled.
        use_hvs (bool): Whether to use hierarchical volume sampling (HVS) during rendering.
        use_direction (bool, optional): Whether to use the direction information when rendering.
            Defaults to True.
        use_random_noise (bool, optional): Whether to add random noise to the density when rendering.
            Defaults to True.

    Outputs:
        A partial function `nerf_specific` that renders a scene using a NeRF model with the given
        parameters when called with the following arguments:
        - params: The parameters of the NeRF model.
        - key: A random key used for generating random numbers during rendering.
        - origins: The origin points of the rays to be cast.
        - directions: The direction vectors of the rays to be cast.
    """

    nerf_specific = functools.partial(
        nerf,
        near=near,
        far=far,
        L_position=L_position,
        L_direction=L_direction,
        num_samples_coarse=num_samples_coarse,
        num_samples_fine=num_samples_fine,
        use_hvs=use_hvs,
        use_direction=use_direction,
        use_random_noise=use_random_noise,
    )

    return nerf_specific


