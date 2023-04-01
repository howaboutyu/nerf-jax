
import jax
import jax.numpy as jnp
import optax

import flax
from flax.training import checkpoints, train_state
from flax.metrics import tensorboard

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

import numpy as np
import functools


import yaml

from nerf import (
    get_model,
    get_nerf,
)

from nerf_config import get_config, NerfConfig
from datasets import dataset_factory





def train_and_evaluate(config: NerfConfig):
    '''
    Train and evaluate the model
    Inputs:
        config: NerfConfig object with all the hyperparameters

    '''

    devices = jax.local_devices()
    print(f'Devices: {devices}')
 
    dataset = dataset_factory(config)
    
    model, params = get_model(config.L_position, config.L_direction)
    
    if config.load_ckpt_dir is not None:
        state = checkpoints.restore_checkpoint(config.load_ckpt_dir, target=None)
    else:
        # create train state
        tx = optax.adam(learning_rate=config.learning_rate)
        state = train_state.TrainState.create(apply_fn=model, params=params, tx=tx)
    
    # we need to replicate the state to all devices
    state = flax.jax_utils.replicate(state)
    
    # create nerf function
    nerf_func = get_nerf(
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
    def train_step(state, key, origins, directions,rgbs):
        '''
        Train step 
        '''

        def loss_func(params):
            (rendered, rendered_hvs), weights, ts = nerf_func(
                params=params,
                model_func=state.apply_fn,
                key=key,
                origins=origins,
                directions=directions,
            )
            
            loss = jnp.mean(jnp.square(rendered - rgbs))

            if config.use_hvs:
                loss += jnp.mean(jnp.square(rendered_hvs - rgbs))

            return loss, (rendered, weights, ts)

    
        # compute loss and grads
        (loss, (rgbs_pred, weights, ts)), grads  = jax.value_and_grad(loss_func, has_aux=True)(state.params)
    
        # combine grads and loss from all devices
        grads = jax.lax.pmean(grads, 'batch')
        loss = jax.lax.pmean(loss, 'batch')
    
        # apply updates on the combined grads
        state = state.apply_gradients(grads=grads)
    
        return state, loss, rgbs_pred, weights, ts 
    
    def eval_step(state, val_data, eval_batch_size):
        '''
        Evaluation step, takes in an entire image and returns the predicted image
        and also metrics. This is single device evaluation 
        
        Inputs:
            state: replicated train state 
            val_dtaa: img, origins and directions of rays each with [H, W, 3]
            eval_batch_size: batch size for evaluation 
        Outputs:
            pred_imgs: predicted images [H, W, 3]
            
        '''
        
        # for eval key stays the same
        key = jax.random.PRNGKey(0)
        
        eval_img = val_data[0]
        eval_origins = val_data[1]
        eval_directions = val_data[2]
   
        # We have to unreplicate the state to evaluate on single device
        state = flax.jax_utils.unreplicate(state)

        origins_flattened = eval_origins.reshape(-1, 3)
        directions_flattened = eval_directions.reshape(-1, 3)

        pred_img_parts = []
        for i in range(0, origins_flattened.shape[0], eval_batch_size):
            origin = origins_flattened[i:i+eval_batch_size]
            direction = directions_flattened[i:i+eval_batch_size]
            
            # expand dim
            (rendered, rendered_hvs), weights, ts = nerf_func(
                params=state.params,
                model_func=state.apply_fn,
                key=key,
                origins=origin,
                directions=direction,
            )
            
            pred_img_parts.append(rendered)


        pred_img = jnp.concatenate(pred_img_parts, axis=0)
        pred_img = pred_img.reshape(eval_origins.shape)

        # expand dims tf.ssims expects [B, H, W, C]
        pred_img = jnp.expand_dims(pred_img, axis=0)
        eval_img = jnp.expand_dims(eval_img, axis=0)

        ssim = tf.image.ssim(eval_img, pred_img, max_val=1.)
        return pred_img, ssim
   

    # summary writer
    summary_writer = tf.summary.create_file_writer(config.ckpt_dir) 
    #summary_writer.hparams(config.__dict__)
    
   
    key = jax.random.PRNGKey(0)
    
    for epoch in range(config.num_epochs):
        print(f'Epoch: {epoch}')

        for idx, (img, origins, directions) in enumerate(dataset['train']):
        
            key_train = jax.random.split(key, img.shape[0])
            key, _ = jax.random.split(key)
            # train step
            state, loss, rgb_pred, weights, ts = train_step(
                state,
                key_train,
                origins,
                directions,
                img,
            )

            with summary_writer.as_default(step=state.step): 
                tf.summary.scalar('train loss', loss[0], step=state.step)


            # Evaluation
            if state.step > 0 and state.step % config.steps_per_eval == 0:
                # Take the first image from the val set  
                eval_data = dataset['val'].get(0)
                pred_img, ssim = eval_step(state, eval_data, config.batch_size)
                with summary_writer.as_default(step=state.step):
                    tf.summary.image('pred_img', pred_img, step=state.step)
                    eval_img = jnp.array([eval_data[0]])
                    tf.summary.image('gt_img', eval_img, step=state.step)
                    tf.summary.scalar('val ssim', ssim[0], step=state.step)
                
            # Save checkpoint
            if state.step > 0 and state.step % config.steps_per_ckpt == 0:
                # unreplicate the state
                unreplicated_state = flax.jax_utils.unreplicate(state)
                checkpoint.save_checkpoint(
                    config.ckpt_dir,
                    unreplicted_state,
                    step=unreplicated_state.step,
                    keep=3, # <- keep last 3 checkpoints
                )

            

if __name__ == '__main__':
    fp = 'configs/lego.yaml'
    nerf_config = get_config(fp) 
    train_and_evaluate(nerf_config)
        
    
        
    
    
