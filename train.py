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
with open('configs/fern.yaml') as file:
  config = yaml.safe_load(file)

ckpt_dir = config['ckpt_dir'] 
print('-----------------')
dataset = dataset_factory(config) 

nerf_components = get_nerf_componets(config)

state = nerf_components['state']
grad_fn = nerf_components['grad_fn']
model_fn = nerf_components['model']
#render_fn = jit(lambda params, origins, directions, key : nerf_components['render_fn'](model_fn, params, origins, directions, key))
#render_hvs_fn = jit(lambda params, origins, directions, weights, key : nerf_components['render_hvs_fn'](model_fn, params, origins, directions, weights, key))
render_fn = nerf_components['render_fn']
render_hvs_fn = nerf_components['render_hvs_fn']


key = jax.random.PRNGKey(0)

print(f'Hello here are your jax devices: {jax.devices()}')

@jit
def train_step(data, state):
    loss_val, grads, pred_train, weights, ts = jax.pmap(grad_fn, in_axes=(None, 0))(state.params, data)
    loss_val = jnp.mean(loss_val)
    grads = jax.tree_map(lambda x : jnp.mean(x, 0), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss_val, pred_train, weights, ts

def val_step(data, state, H, W):

    params = state.params 
    
    (img, origins, directions) = data
    origins = origins.reshape((-1, 3))
    directions = directions.reshape((-1, 3))
    img = img.reshape((-1, 3))
    eval_bs = 4096//4 
    pred_imgs = []
    for i in range(0, len(img), eval_bs):
        jax.debug.print('eval step {}', i)
        _, weights, _ = render_fn(params, origins[i:i+eval_bs], directions[i:i+eval_bs], key)


        pred_img, _, _ = render_hvs_fn(params, origins[i:i+eval_bs], directions[i:i+eval_bs], key, weights)


        pred_imgs.append(pred_img)
    pred_imgs = jnp.concatenate(pred_imgs)
    pred_img = jnp.reshape(pred_imgs, (H, W, 3))
    return pred_img
    #cv2.imwrite('/tmp/val_img.jpg', pred_img * 255)

#val_concrete = lambda data, state : val_step(data, state, dataset['train'].H, dataset['train'].W)
#val_step_jit = jit(val_concrete)

for i in range(config['num_epochs']):
    if i % 10 == 0 and i > 0:    
        pred_img = val_step(dataset['train'].get(0), state, dataset['train'].H, dataset['train'].W)
        cv2.imwrite(f'/tmp/eval_{i}.jpg', np.array(pred_img * 255))

        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=i, overwrite=True)
    #import pdb; pdb.set_trace()
    
    for idx, (img, origins, directions) in enumerate(dataset['train']):

        key_train = random.split(key, img.shape[0])
#        if config['split_to_patches']:
#            key_train = random.split(key, img.shape[-4])
#
#            if config['use_batch']:
#                key_train = random.split(key, len(key_train) * dataset['train'].batch_size).reshape((-1, img.shape[-4], 2))
#

        data = (origins, directions, img, key_train)
        #import pdb; pdb.set_trace()
        state, loss_val, pred_train, weights, ts = train_step(data, state)
        print(f'Epoch {i} step {idx} loss : {loss_val}')

        #pred_train = pred_train[0] # just take first example

        #pred_train = patches2data(pred_train, dataset['train'].split_h)

        #pred_train = np.array(pred_train*255)
        #print('max predicted image', np.max(pred_train))

        #succ = cv2.imwrite(f'/tmp/train_image_{idx}_at_epoch_{i}.jpg', pred_train)
        #print('write succ', succ)

        
        key, _ = random.split(key)

    #checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=i, overwrite=True)
    continue
    # Evaluation
    for idx, (img, origins, directions) in enumerate(dataset['val']):
        print('evaluating')
        
        data = (origins, directions, img, key_train)
         
        weights = weights[0]

        if config['split_to_patches']:
            origins_eval = patches2data(origins[0], dataset['train'].split_h)
            directions_eval = patches2data(directions[0], dataset['train'].split_h)

        pred_train, weights, ts = render_fn(model_fn, state.params, origins_eval, directions_eval)

        pred_train = np.array(pred_train*255)
        print('max predicted image', np.max(pred_train))

        cv2.imwrite(f'/tmp/eval_image_{idx}_at_epoch_{i}.jpg', pred_train)

 
