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
 
with open('configs/lego.yaml') as file:
  config = yaml.safe_load(file)
  

dataset = dataset_factory(config) 

nerf_components = get_nerf_componets(config)

optimizer = nerf_components['optimizer']
opt_state = nerf_components['opt_state']
grad_fn = nerf_components['grad_fn']
params = nerf_components['params']
render_fn = nerf_components['render_eval_fn']
model_fn = nerf_components['model']

key = jax.random.PRNGKey(0)


@jax.jit
def train_step(params, data, opt_state):
    loss_val, grads, pred_train = grad_fn(params, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates) 
    return params, opt_state, loss_val, pred_train

    
for i in range(config['num_epochs']):
    
    for idx, (img, origins, directions) in enumerate(dataset['train']):
        key_train = key
        if config['split_to_patches']:
            key_train = random.split(key, len(img))

        data = (origins, directions, img, key_train)
        params, opt_state, loss_val, pred_train = train_step(params, data, opt_state)
        
        if config['split_to_patches']:
            pred_train = patches2data(pred_train, dataset['train'].split_h)
            
        pred_train = np.array(pred_train*255)
        cv2.imwrite(f'/tmp/train_{i}_{idx}.jpg', pred_train)
        print(f'Loss val {loss_val}')

        key, _ = random.split(key)
        
    
