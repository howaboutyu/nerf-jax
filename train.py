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

@jit
def train_step(data, state):
    loss_val, grads, pred_train, weights = jax.pmap(grad_fn, in_axes=(None, 0))(state.params, data)
    loss_val = jnp.mean(loss_val)
    grads = jax.tree_map(lambda x : jnp.mean(x, 0), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss_val, pred_train, weights

    
for i in range(config['num_epochs']):
    
    for idx, (img, origins, directions) in enumerate(dataset['train']):
        key_train = key
        if config['split_to_patches']:
            key_train = random.split(key, img.shape[-4])

            if config['use_batch']:
                key_train = random.split(key, len(key_train) * dataset['train'].batch_size).reshape((-1, img.shape[-4], 2))

        #if config['use_batch']:
        #    key_train = random.split(key, len(img))



        data = (origins, directions, img, key_train)
        state, loss_val, pred_train, weights = train_step(data, state)
        print(f'Epoch {i} step {idx} loss : {loss_val}')
        
        key, _ = random.split(key)
        break

    if config['split_to_patches']:
        print('evaluating')
        
        data = (origins, directions, img, key_train)
        import pdb; pdb.set_trace()
        image_pred, weights = render_fn(model_fn, state.params, origins[0], directions[0])
         
        if config['use_batch']: 
            pred_train = pred_train[0] # just take first example
            weights = weights[0]
        pred_train = patches2data(pred_train, dataset['train'].split_h)
        weights = patches2data(weights, dataset['train'].split_h)

        np.save('weights.npy', weights[50][50])
    
        pred_train = np.array(pred_train*255)

       
    cv2.imwrite(f'/tmp/train_{i}_{idx}.jpg', pred_train)

    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=i, overwrite=True)
 
