
import jax
from jax import grad, jit, vmap
from flax.training import checkpoints
import jax.numpy as jnp
import jax.random as random

import numpy as np
import cv2
import yaml
from absl import app, flags

from nerf import get_nerf_componets
from datasets import dataset_factory

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "config file path")
flags.DEFINE_string("mode", "train", "mode; can be train or render")
flags.mark_flag_as_required("config_path")


def train_step(data, state, grad_fn):
    loss_val, grads, pred_train, weights, ts = jax.pmap(grad_fn, in_axes=(None, 0))(state.params, data)
    loss_val = jnp.mean(loss_val)
    grads = jax.tree_map(lambda x : jnp.mean(x, 0), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss_val, pred_train, weights, ts

def render_step(data, state, H, W, render_fn, render_hvs_fn, eval_bs=2048):
    print('starting render step')
    params = state.params 
    
    (origins, directions) = data
    origins = origins.reshape((-1, 3))
    directions = directions.reshape((-1, 3))

    key = jax.random.PRNGKey(0)

    pred_imgs = []
    for i in range(0, len(origins), eval_bs):
        jax.debug.print('eval step {}', i)
        _, weights, _ = render_fn(params, origins[i:i+eval_bs], directions[i:i+eval_bs], key)
        pred_img, _, _ = render_hvs_fn(params, origins[i:i+eval_bs], directions[i:i+eval_bs], key, weights)
        pred_imgs.append(pred_img)

    pred_imgs = jnp.concatenate(pred_imgs)
    pred_img = jnp.reshape(pred_imgs, (H, W, 3))
    return pred_img

def main(argv):
    del argv

    with open(FLAGS.config_path) as file:
      config = yaml.safe_load(file)
    
    ckpt_dir = config['ckpt_dir'] 
    
    # get dataset iterator
    dataset = dataset_factory(config) 
   
    config.update({'near': dataset['train'].near, 'far': dataset['train'].far})
    nerf_components = get_nerf_componets(config)
    
    state = nerf_components['state']
    grad_fn = nerf_components['grad_fn']
    model_fn = nerf_components['model']
    render_fn = nerf_components['render_fn']
    render_hvs_fn = nerf_components['render_hvs_fn']
    
    
    key = jax.random.PRNGKey(0)
    
    print(f'jax devices: {jax.devices()}')

    psnr_fn = lambda mse: -10. / jnp.log(10.) * jnp.log(mse)
    if FLAGS.mode == 'train':
        for i in range(config['num_epochs']):
            # train    
            for idx, (img, origins, directions) in enumerate(dataset['train']):
        
                key_train = random.split(key, img.shape[0])
                print(f'Img batch shape : {img.shape}')
        
                data = (origins, directions, img, key_train)
                state, loss_val, pred_train, weights, ts = train_step(data, state, grad_fn)
                
                psnr = psnr_fn(loss_val)
                print(f'Epoch {i} step {idx} loss : {loss_val}, psnr : {psnr}')
                
                key, _ = random.split(key)
        
            # val and save checkpoint
            if i % 10 == 0 and i > 0:    
                data = dataset['val'].get(0)
                pred_img = render_step(
                        (data[1], data[2]), 
                        state, 
                        dataset['train'].H, 
                        dataset['train'].W,
                        render_fn,
                        render_hvs_fn,
                        )
                cv2.imwrite(f'/tmp/eval_{i}.jpg', np.array(pred_img * 255)) 

                checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=i, overwrite=True)
    elif FLAGS.mode == 'render':
        print('rendering started ...')
        for idx, data in enumerate(dataset['render']):
            pred_img = render_step(
                        data, 
                        state, 
                        dataset['train'].H, 
                        dataset['train'].W,
                        render_fn,
                        render_hvs_fn,
                        )

            cv2.imwrite(f'/tmp/{config["dataset_name"]}_render_' +str(idx).zfill(4)+ '.jpg', np.array(pred_img * 255))

    else:
        print('Mode can only be : [train, render]') 


if __name__ == '__main__':
    app.run(main)
