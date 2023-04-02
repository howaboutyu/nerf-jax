import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import checkpoints, train_state
from flax.metrics import tensorboard
import tensorflow as tf
import numpy as np
import functools
import yaml
import cv2
from dataclasses import dataclass

from nerf import (
    get_model,
    get_nerf,
)
from nerf_config import get_config, NerfConfig
from datasets import dataset_factory


def get_nerf_eval_func(config: NerfConfig):
    """ Return NeRF function for evaluation """
    return get_nerf(
        near=config.near,
        far=config.far,
        L_position=config.L_position,
        L_direction=config.L_direction,
        num_samples_coarse=config.num_samples_coarse,
        num_samples_fine=config.num_samples_fine,
        use_hvs=config.use_hvs,
        use_direction=True,
        use_random_noise=False,
    )


def render_step(nerf_func, state, val_data, eval_batch_size):
    """
    Render step, takes in an entire image and returns the predicted image

    Inputs:
        nerf_func: a function performs the nerf algorithm
        state: replicated train state
        val_data: origins and directions of rays each with [H, W, 3]
        eval_batch_size: batch size for evaluation
    Outputs:
        pred_img: predicted images [H, W, 3]
        depth_img: predicted images [H, W, 3]

    """

    # for eval key stays the same
    key = jax.random.PRNGKey(0)

    eval_origins = val_data[0]
    eval_directions = val_data[1]
    
    origins_flattened = eval_origins.reshape(-1, 3)
    directions_flattened = eval_directions.reshape(-1, 3)

    pred_img_parts = []
    pred_depth_parts = []
    for i in range(0, origins_flattened.shape[0], eval_batch_size):
        origin = origins_flattened[i : i + eval_batch_size]
        direction = directions_flattened[i : i + eval_batch_size]

        (rendered, rendered_hvs), weights, ts = nerf_func(
            params=state.params,
            key=key,
            origins=origin,
            directions=direction,
        )

        depth_hvs = jnp.sum(weights * ts, -1)

        pred_img_parts.append(rendered_hvs)
        pred_depth_parts.append(depth_hvs)

    # Reshape image
    pred_img = jnp.concatenate(pred_img_parts, axis=0)
    pred_img = pred_img.reshape(eval_origins.shape)

    # expand dims tf.ssims expects [B, H, W, C]
    pred_img = jnp.expand_dims(pred_img, axis=0)

    # Reshape depth image
    pred_depth = jnp.concatenate(pred_depth_parts, axis=0)
    pred_depth = pred_depth.reshape(eval_origins.shape[:-1])
    pred_depth = jnp.expand_dims(pred_depth, axis=[0, -1])

    return pred_img, pred_depth


def nerf_to_mesh(config: NerfConfig):
    """
    This function converts a NeRF model into a 3D mesh
    using Open3D.
    
    """
    
    dataset = dataset_factory(config)

    if config.dataset_type == "llff":
        # set near and far to calculated values
        config.near = dataset["train"].near
        config.far = dataset["train"].far

    model, _ = get_model(config.L_position, config.L_direction)
    
    # Load model
    state_dict = checkpoints.restore_checkpoint(config.load_ckpt_dir, target=None)
    
    # Create a state for eval
    eval_state = train_state.TrainState.create(apply_fn=model, params=state_dict['params'], tx=optax.adam(1))
    
    # create nerf function
    eval_nerf_func = get_nerf_eval_func(config)
    jit_eval_nerf_func = jax.jit(
        functools.partial(eval_nerf_func, model_func=model)
    )
    
    
    for idx, data in enumerate(dataset['render']):
        print(f'idx {idx}')
        pred_img, pred_depth = render_step(
                    jit_eval_nerf_func, eval_state, data, config.batch_size
                )
        
        pred_img = np.asarray(pred_img * 255).astype(np.uint8)
        pred_img = np.squeeze(pred_img)
        
        cv2.imwrite(f'pred_img{idx}.png', pred_img)
        #import pdb; pdb.set_trace()

if __name__ == '__main__':
    config = get_config('configs/miso_shop.yaml')
    nerf_to_mesh(config)