from flax import linen as nn
from flax.training import train_state, checkpoints

import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import cv2


num_samples = 64 * 2 

H, W = 100, 100
L_position = 5 


def get_rays(H, W, focal, pose):
    # get rays (origin and direction) split into `num_splits` arrays 
    #x, y = jnp.mgrid[0:W, 0:H]
    x, y = jnp.meshgrid(jnp.arange(W, dtype=jnp.float32), jnp.arange(H, dtype=jnp.float32), indexing='xy')
    x = (x - 0.5 * W)/focal
    y = -(y - 0.5 * H)/focal


    
    direction = jnp.stack([x, y, -jnp.ones_like(x)], -1)

    rot = pose[:3, :3] 
    direction = (direction[..., jnp.newaxis, :] * rot).sum(-1)

    # Normalize direction
    direction_norm = jnp.linalg.norm(direction, axis=-1)
    direction = direction/direction_norm[..., jnp.newaxis]

    translation = pose[:3, 3]
    origin = jnp.broadcast_to(translation, direction.shape)
    
    return origin, direction 

    
def encoding_func(x, L):
    encoded_array = [x]
    for i in range(L):
        encoded_array.extend([jnp.sin(2. ** i * jnp.pi * x), jnp.cos(2. ** i * jnp.pi * x)])
    return jnp.concatenate(encoded_array, -1)

def render(model_func, params, origin, direction, key, rand):
    t = jnp.linspace(2., 6., num_samples) 

    if rand: 
        random_shift = jax.random.uniform(key, (origin.shape[0], origin.shape[1], num_samples)) * (far-near)/num_samples  
        t = t+ random_shift 
    else:
        t = jnp.broadcast_to(t, (origin.shape[0], origin.shape[1], num_samples))

    points = origin[..., jnp.newaxis, :] + t[..., jnp.newaxis] * direction[..., jnp.newaxis, :]
    points = jnp.squeeze(points)
    points_flatten = points.reshape((-1, 3))
    encoded_x = encoding_func(points_flatten, L_position)
    
    rgb_array, opacity_array = [], []
    for _cc in range(0, encoded_x.shape[0], 4096*10):
        rgb, opacity = model_func.apply(params, encoded_x[_cc:_cc + 4096*10]) 
        rgb_array.append(rgb)
        opacity_array.append(opacity)
    
    rgb = jnp.concatenate(rgb_array, 0)
    opacity = jnp.concatenate(opacity_array, 0)
    
    rgb =rgb.reshape((points.shape[0], points.shape[1], num_samples, 3))
    opacity =opacity.reshape((points.shape[0], points.shape[1], num_samples, 1))

    rgb = jax.nn.sigmoid(rgb)
    opacity = jax.nn.relu(opacity) 
   
    t_delta = t[...,1:] - t[...,:-1]
    t_delta = jnp.concatenate([t_delta, jnp.broadcast_to(jnp.array([1e10]),   [points.shape[0], points.shape[1], 1])], 2)

    
    T_i = jnp.cumsum(jnp.squeeze(opacity) * t_delta + 1e-10, -1)   
    T_i = jnp.insert(T_i, 0, jnp.zeros_like(T_i[...,0]),-1)
    T_i = jnp.exp(-T_i)[..., :-1]
     
    c_array = T_i[..., jnp.newaxis]*(1.-jnp.exp(-opacity*t_delta[..., jnp.newaxis])) * rgb 
    c_sum =jnp.sum(c_array, -2)

    return c_sum 

render_concrete = lambda model_func, params, origin, direction, key: render(model_func, params, origin, direction, key, True)

class Model(nn.Module):

  @nn.compact
  def __call__(self, z):
    input = z
    z = nn.Dense(L_position*6+3, name='fc_in')(z)
    z = nn.relu(z)

    for i in range(8):
        z = nn.Dense(256, name=f'fc{i}')(z)
        z = nn.relu(z)
        if i == 4:
            z = jnp.concatenate([z, input], -1) 

        if i == 7: 
            d = nn.Dense(1, name='fcd2')(z)

    z = nn.Dense(128, name='fc_128')(z)
    
    z = nn.Dense(3, name='fc_f')(z)
    return z, d 

model = Model()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, L_position * 6 + 3)))

learning_rate = 5e-4


optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

#opt_state = checkpoints.restore_checkpoint(ckpt_dir='ckpt_mm', target=opt_state)

near = 2.
far = 6.
split_image = True
split_dim = 4
n_w = 2
n_h = 2
patch_w = W//n_w
patch_h = H//n_h

@jit
def get_grad(params, data):
    origins, directions, y_target, key = data
    def loss_func(params):
            
        image_pred = render_concrete(model, params, origins, directions, key)
        return jnp.mean((image_pred -  y_target) ** 2), image_pred

    (loss_val, image_pred), grads    = jax.value_and_grad(loss_func, has_aux=True)(params)
    return loss_val, grads, image_pred

@jit
def get_patches_grads(params, origins_split, directions_split, y_split, keys):
    loss_array, grads_array, pred_train_array = jax.lax.map(lambda grad_input : get_grad(params, grad_input), (origins_split, directions_split, y_split, keys))
    grads = jax.tree_map(lambda x : jnp.mean(x, 0), grads_array)

    loss_val = jnp.mean(loss_array)
    return loss_val, grads, pred_train_array[0]

def split2patches(data, n_w=n_w, n_h=n_h):
    #patches_array = jnp.hsplit(data, n_h) 
    #patches_array = [jnp.vsplit(p, n_w) for p in patches_array] 
    #import pdb; pdb.set_trace()
    patches_array = []

    for i_w in jnp.arange(0, n_w, 1):
        for i_h in jnp.arange(0, n_h, 1):
            patches_array.append(data[i_h*patch_h:(i_h+1)*patch_h, i_w*patch_w:(i_w+1)*patch_w])
    return jnp.stack(patches_array, 0)

#split2patches_concrete = jit(lambda data : split2patches(data, patch_w, patch_h, n_w, n_h))

def train_step(params, x, y, opt_state, key, split_image):
    
    origins, directions = x 
    y = jnp.array(y)

        
    key, _ = random.split(key) 
    
    if split_image:     
        #data = (origins, directions, y, key)
        # Split the image into parts to avoid OEM error for low memory GPUs
        origins_split = split2patches(origins) 
        directions_split = split2patches(directions) 
        y_split = split2patches(y) 
        keys = random.split(key, len(y_split))
        #import pdb; pdb.set_trace()

       
        loss_val, grads, pred_train = get_patches_grads(params, origins_split, directions_split, y_split, keys)
        #def loss_func(params):
        #    predictions = jax.lax.map(lambda data: get_prediction(params, data), (origins, directions, keys))
        #    #prediction_array = []
        #    #for i in [0, 1]:
        #    #    for j in [0, 1]:
        #    #        prediction_array.append(predictions[2*i + j].reshape((50, 50, 3)))


        #    return jnp.mean(jnp.square(predictions- y_target.reshape((4, -1))))
    else:
        data = (origins, directions, y, key)
        loss_val, grads, pred_train = get_grad(params, data)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates) 
     
    jax.debug.print('loss {}:', loss_val)
    return params, opt_state,key, pred_train


data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']


H, W = 100, 100
    

key = random.PRNGKey(0) 

batch_size = 2500 

num_splits =  (H * W) / batch_size
get_rays_jit = lambda pose: get_rays(H=H, W=W, focal=focal, pose=pose, num_splits=num_splits)



for epoch_cc in range(10000000):    
    img_idx = epoch_cc % len(images) 
    image_train = images[img_idx]

    pose = poses[img_idx]
    
    origins, directions=get_rays(100, 100, focal, pose)
    
    key, _ = random.split(key) 
    params, opt_state, key, pred_train  = train_step(params, (origins, directions), image_train, opt_state, key, split_image)

    cv2.imwrite(f'/tmp/train_{epoch_cc}.jpg', (np.array(pred_train)*255).astype(np.uint8))
    
    #checkpoints.save_checkpoint(ckpt_dir='ckpt', target=opt_state, step=epoch_cc, overwrite=True)

    
    if epoch_cc%50 == 0:
        print('Begin eval')
        origins, directions = get_rays(100, 100, focal, poses[101])


        pred_image = render(model, params, origins, directions, None, None)



        pred_image = np.array(pred_image) 
        pred_image = (pred_image * 255.).astype(np.uint8)[:,:,::-1]
        actual_img = (images[101]* 255.).astype(np.uint8)[:,:,::-1]
        cv2.imwrite(f'/tmp/pred_img{epoch_cc}.jpg', pred_image)
        cv2.imwrite(f'/tmp/actual.jpg', actual_img)
    
