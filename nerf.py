import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import checkpoints, train_state
import optax
from typing import Any



def encoding_func(x, L):
    encoded_array = [x]
    for i in range(L):
        encoded_array.extend([jnp.sin(2. ** i * jnp.pi * x), jnp.cos(2. ** i * jnp.pi * x)])
    return jnp.concatenate(encoded_array, -1)


def hvs(weights, t, t_to_sample, key):
    weights = jnp.squeeze(weights) + 1e-10 

    # normalize
    norm = jnp.sum(weights, -1)
    weights = weights/norm[..., jnp.newaxis]

    cdf = jnp.cumsum(weights, -1)

#    cdf = jnp.concatenate([
#        jnp.zeros_like(cdf[...,0][..., jnp.newaxis]), 
#        cdf, 
#        jnp.ones_like(cdf[..., 1][..., jnp.newaxis])], -1)
 


    def inverse_sample(Z):
    
        abs_diff = jnp.abs(cdf[...,  jnp.newaxis, :] - Z[..., jnp.newaxis])
    
        argmin = jnp.argmin(abs_diff, -2)
    
        sampled_t = jnp.take_along_axis(t_to_sample, argmin, -1)
        return sampled_t

    
    # TODO: currently it is assuming that len(t)//2 = len(t_to_sample), like in the paper
    Z = jax.random.uniform(key, t_to_sample.shape)

    key, _ = jax.random.split(key)
    sampled_t1 = inverse_sample(Z)

    Z = jax.random.uniform(key, t_to_sample.shape)
    sampled_t2 = inverse_sample(Z)

    new_t = jnp.concatenate([sampled_t1, sampled_t2, t], -1)
    new_t = jnp.sort(new_t, -1)
    return new_t 

def render(model_func, params, origin, direction, key, near, far, num_samples, L_position, L_direction, rand, use_hvs, weights):
    t = jnp.linspace(near, far, num_samples) 

    if rand: 
        random_shift = jax.random.uniform(key, (origin.shape[0], num_samples)) * (far-near)/num_samples  
        t = t+ random_shift 

    elif use_hvs:

        t_to_sample = jnp.broadcast_to(t, (origin.shape[0], num_samples))

        t = jnp.linspace(near, far, weights.shape[-2]) 
        t = jnp.broadcast_to(t, (origin.shape[0], weights.shape[-2]))

        t = hvs(weights, t, t_to_sample, key)
        t = jax.lax.stop_gradient(t) 

    else:
        t = jnp.broadcast_to(t, (origin.shape[0], num_samples))


    
    points = origin[..., jnp.newaxis, :] + t[..., jnp.newaxis] * direction[..., jnp.newaxis, :]
    encoded_x = encoding_func(points, L_position)

    if L_direction:
        direction = jnp.broadcast_to(direction[..., jnp.newaxis, :], points.shape)
        encoded_dir = encoding_func(direction, L_direction)
        rgb, opacity = model_func.apply(params, encoded_x, encoded_dir) 
    else: 
        rgb, opacity = model_func.apply(params, encoded_x) 

    rgb = jax.nn.sigmoid(rgb)
    
    if rand:
        opacity = opacity + jax.random.normal(key, opacity.shape, dtype=opacity.dtype) 
   
    opacity = jax.nn.relu(opacity) 

    t_delta = t[...,1:] - t[...,:-1]
    t_delta = jnp.concatenate([t_delta, jnp.broadcast_to(jnp.array([1e10]),   [points.shape[0], 1])], 1)

    
    T_i = jnp.cumsum(jnp.squeeze(opacity) * t_delta + 1e-10, -1)   
    T_i = jnp.insert(T_i, 0, jnp.zeros_like(T_i[...,0]),-1)
    T_i = jnp.exp(-T_i)[..., :-1]
    

    weights = T_i[..., jnp.newaxis]*(1.-jnp.exp(-opacity*t_delta[..., jnp.newaxis])) 
    c_array = weights * rgb 
    c_sum =jnp.sum(c_array, -2)

    return c_sum, weights, t


def get_model(L_position, L_direction):
    class Model(nn.Module):

      @nn.compact
      def __call__(self, z, direction):
        input = z
    
        for i in range(9):
            z = nn.Dense(256, name=f'fc{i}')(z)
            z = nn.relu(z)
            if i == 4:
                z = jnp.concatenate([z, input], -1) 
    
            if i == 7: 
                d = nn.Dense(1, name='fcd2')(z)

        if L_direction: z = jnp.concatenate([z, direction], -1)

        z = nn.Dense(128, name='fc_128')(z)
        z = nn.relu(z) 
        z = nn.Dense(3, name='fc_f')(z)
        return z, d 
    
    model = Model()

    if not L_direction:
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, L_position * 6 + 3)))
    else:
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, L_position * 6 + 3)), jnp.ones((1, L_direction * 6 + 3)))

    return model, params

def get_grad(params, data, render, render_hvs, use_hvs):
    origins, directions, y_target, key = data
    def loss_func(params):
        image_pred, weights, ts = render(params, origins, directions, key)

        if not use_hvs:
            return jnp.mean((image_pred -  y_target) ** 2), (image_pred, weights, ts)

        image_pred_hvs, weights_hvs, ts  = render_hvs(params, origins, directions, key, weights)

        loss_hvs =  jnp.mean((image_pred -  y_target) ** 2 + (image_pred_hvs -  y_target) ** 2)
        return loss_hvs, (image_pred_hvs, weights, ts)

    (loss_val, (image_pred, weights, ts)), grads = jax.value_and_grad(loss_func, has_aux=True)(params)
    return loss_val, grads, image_pred, weights, ts



def get_nerf_componets(config):

    near = config['near']
    far = config['far']
    
    num_samples = config['num_samples'] 
    use_hvs = config['use_hvs']
    hvs_num_samples = config['hvs_num_samples'] 
    L_position = config['L_position']
    L_direction = config['L_direction'] if 'L_direction' in config else None

 
    model, params = get_model(config['L_position'], L_direction)

    # render function for training with random sampling
    render_concrete = lambda params, origin, direction, key: \
        render(model, params, origin, direction, key, near, far, num_samples, L_position, L_direction, True, False, None)

    # render function for training with Hierarchical volume sampling (hvs) 
    render_concrete_hvs = lambda params, origin, direction, key, weights: \
        render(model, params, origin, direction, key, near, far, hvs_num_samples, L_position, L_direction, False, True, weights)

    render_concrete = jax.jit(render_concrete) 
    render_concrete_hvs = jax.jit(render_concrete_hvs) 

    grad_fn = lambda params, data: get_grad(params, data, render_concrete, render_concrete_hvs, use_hvs)
    
   
    learning_rate = config['init_lr'] 
    
    # create train state
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply,
                                        params=params,
                                        tx=tx)
    
    # load from ckpt
    if 'ckpt_dir' in config:
        print(f'Loading checkpoint from : {config["ckpt_dir"]}')
        state = checkpoints.restore_checkpoint(ckpt_dir=config['ckpt_dir'], target=state)
 
    model_components = {
        'model': model,
        'render_fn': render_concrete,
        'render_hvs_fn': render_concrete_hvs,
        'grad_fn': grad_fn,
        'state': state,
    }

    return model_components

if __name__ == '__main__':
    get_model(100)
