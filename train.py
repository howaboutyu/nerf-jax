from flax import linen as nn
from flax.training import train_state, checkpoints

import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

import cv2
import yaml

from nerf import get_nerf_componets
from datasets import dataset_factory, patches2data

# TODO: add input args 
with open('configs/lego.yaml') as file:
  config = yaml.safe_load(file)

ckpt_dir = 'ckpt_lego' 
print('-----------------')
dataset = dataset_factory(config) 

nerf_components = get_nerf_componets(config)

state = nerf_components['state']
grad_fn = nerf_components['grad_fn']
render_fn = nerf_components['render_eval_fn']
model_fn = nerf_components['model']

key = jax.random.PRNGKey(0)

print(f'Hello here are your jax devices: {jax.devices()}')

#@jit
def train_step(data, state):
    loss_val, grads, pred_train, weights, ts = jax.pmap(grad_fn, in_axes=(None, 0))(state.params, data)
    loss_val = jnp.mean(loss_val)
    grads = jax.tree_map(lambda x : jnp.mean(x, 0), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss_val, pred_train, weights, ts

    
for i in range(config['num_epochs']):
    
    for idx, (img, origins, directions) in enumerate(dataset['train']):
        key_train = key
        if config['split_to_patches']:
            key_train = random.split(key, img.shape[-4])

            if config['use_batch']:
                key_train = random.split(key, len(key_train) * dataset['train'].batch_size).reshape((-1, img.shape[-4], 2))


        data = (origins, directions, img, key_train)
        state, loss_val, pred_train, weights, ts = train_step(data, state)
        print(f'Epoch {i} step {idx} loss : {loss_val}')
        
        key, _ = random.split(key)

    # Evaluation
    for idx, (img, origins, directions) in enumerate(dataset['val']):
        print('evaluating')
        
        data = (origins, directions, img, key_train)
         
        pred_train = pred_train[0] # just take first example
        weights = weights[0]

        if config['split_to_patches']:
            origins_eval = patches2data(origins[0], dataset['train'].split_h)
            directions_eval = patches2data(directions[0], dataset['train'].split_h)

        pred_train, weights, ts = render_fn(model_fn, state.params, origins_eval, directions_eval)

        pred_train = np.array(pred_train*255)
        print('max predicted image', np.max(pred_train))

        cv2.imwrite(f'/tmp/eval_image_{idx}_at_epoch_{i}.jpg', pred_train)

    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=i, overwrite=True)
 
