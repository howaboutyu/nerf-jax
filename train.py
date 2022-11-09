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
from datasets import dataset_factory

# TODO: add input args 
with open('configs/beer.yaml') as file:
  config = yaml.safe_load(file)

ckpt_dir = config['ckpt_dir'] 
print('-----------------')
dataset = dataset_factory(config) 

config.update({'near': dataset['train'].near, 'far': dataset['train'].far})
nerf_components = get_nerf_componets(config)

state = nerf_components['state']
grad_fn = nerf_components['grad_fn']
model_fn = nerf_components['model']
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

for i in range(config['num_epochs']):
    # train    
    for idx, (img, origins, directions) in enumerate(dataset['train']):

        key_train = random.split(key, img.shape[0])

        data = (origins, directions, img, key_train)
        state, loss_val, pred_train, weights, ts = train_step(data, state)
        print(f'Epoch {i} step {idx} loss : {loss_val}')
        
        key, _ = random.split(key)

    # val
    if i % 10 == 0 and i > 0:    
        pred_img = val_step(dataset['val'].get(0), state, dataset['train'].H, dataset['train'].W)
        cv2.imwrite(f'/tmp/eval_{i}.jpg', np.array(pred_img * 255))

        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=i, overwrite=True)

