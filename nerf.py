from flax import linen as nn
from flax.training import train_state

import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import cv2


num_samples = 64 

H, W = 100, 100
L_position = 7 


def get_rays(H, W, focal, pose, num_splits):
    x, y = jnp.mgrid[0:W, 0:H]
    x = (x - W/2)/focal
    y = (y - H/2)/focal

    y = -y # bender seems to use -y 

    x = x.flatten()
    y = y.flatten()

    direction = jnp.stack([x, y, -jnp.ones_like(x)])
    # Normalize direction
    direction_norm = jnp.linalg.norm(direction, ord=2, axis=0)
    direction = direction/direction_norm

    rot = pose[:3, :3] 
    direction = jnp.matmul(rot, direction)

    translation = pose[:3, 3]
    translation = translation[..., jnp.newaxis]
    origin = jnp.broadcast_to(translation, direction.shape)
    
    
    origin = jnp.transpose(origin)
    direction = jnp.transpose(direction)
    origin = jnp.split(origin, num_splits)
    direction = jnp.split(direction, num_splits)

 
    return jnp.array(origin), jnp.array(direction)


        

def encoding_func(x, L):
    encoded_array = [x]
    for i in range(L):
        encoded_array.extend([jnp.sin(jnp.power(2., i) * jnp.pi * x), jnp.cos(jnp.power(2.,i) * jnp.pi * x)])
    return jnp.array(encoded_array)

def integrate(model_func, params, origin, direction, key):
    # data - origins [num_samples, 3]
    # data - directions [num_samples, 3]
    t = jnp.linspace(2., 6., num_samples) 
    random_shift = jax.random.uniform(key, (3, num_samples)) * (far-near)/num_samples  
    t = t + random_shift 
    points = origin[..., jnp.newaxis] + direction[..., jnp.newaxis]*t
    encoded_x = encoding_func(points, L_position)
    encoded_x = jnp.reshape(encoded_x, [-1, num_samples]) 
    
    encoded_x = jnp.transpose(encoded_x)
    
    rgb, opacity = model_func.apply(params, encoded_x)   
   
    rgb = jax.nn.sigmoid(rgb)
    opacity = jax.nn.relu(opacity)
   
    t_delta = t[..., 1:] - t[...,:-1]
    #t_delta = jnp.concatenate([t_delta, jnp.array([1e10])])
    t_delta = jnp.concatenate([t_delta, jnp.broadcast_to(jnp.array([1e10]), [3, 1])], -1)
    #t_delta = jnp.reshape(t_delta, [num_samples, 1])
    
    t_delta = jnp.transpose(t_delta, [1,0]) 
    # Eq (3) in paper 
    T_sum = jnp.cumsum(opacity*t_delta, 1)
    T_i = jnp.insert(T_i, 0, jnp.array([[0.,0.,0.]]),0)
    
    T_i = jnp.exp(-T_sum)[:-1]
    
    
    #T_i = jnp.cumproduct(jnp.exp(-opacity * t_delta + 1e-10), 0)  
    ##T_i = jnp.insert(T_i, 0, 1.0) 
    #T_i = jnp.insert(T_i, 0, jnp.array([[1.,1.,1.]]),0)
    #T_i = T_i[:-1]

    
    #import pdb; pdb.set_trace() 
     
    c_array = T_i*(1.-jnp.exp(-opacity*t_delta)) * rgb 
    c_sum =jnp.sum(c_array, 0)

    return c_sum 

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
            d = nn.Dense(32, name='fcd')(z)
            d = nn.relu(d)
            d = nn.Dense(1, name='fcd2')(d)
            

    
    z = nn.Dense(128, name='fc_128')(z)
    
    z = nn.Dense(3, name='fc_f')(z)
    return z, d 

model = Model()
params = model.init(jax.random.PRNGKey(50), jnp.ones((2000, L_position * 6 + 3)))
learning_rate = 2e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

near = 2.
far = 6.

render = vmap(integrate, in_axes = (None, None, 0, 0, 0)) 
render_batched = vmap(render, (None, None, 0, 0, 0) )
def get_grad(params, data):
    origins, directions, y_target, key = data
    def loss_func(params):
        
        keys = random.split(key, len(origins)) 
        
        #image_pred = jnp.reshape(jnp.zeros_like(y), [-1, 3])
        #c_array = []
        #for idx, (b_key, b_origins, b_directions) in enumerate(zip(keys, origins, directions)):
        #    b_keys = random.split(b_key, len(b_origins))
        #    c = render(model, params, b_origins, b_directions, b_keys) 
        #    c_array.append(c)
        #    #image_pred = image_pred.at[idx*c.shape[0]:(idx + 1) * c.shape[0]].set(c)
        
        #import pdb; pdb.set_trace()
        image_pred = render(model, params, origins, directions, keys)
        
        #import pdb; pdb.set_trace() 
        #import pdb; pdb.set_trace()
        return jnp.mean(jnp.square(image_pred- y_target))

    loss_val, grads = jax.value_and_grad(loss_func)(params)
    return loss_val, grads

#get_grad_batched = vmap(get_grad, (None, 0, 0, 0, 0))
@jit
def train_step(params, x, y, opt_state, key):
    
    origins, directions = x 
    y_target = jnp.reshape(y, origins.shape)
    
    keys = random.split(key, len(x[0])) 
    
    
    loss_array, grads_array = jax.lax.map(lambda grad_input : get_grad(params, grad_input), (origins, directions, y_target, keys))

    grads = jax.tree_map(lambda x : jnp.mean(x, 0), grads_array)
    #import pdb; pdb.set_trace()
    #loss_val, grads = get_grad(params, x, y, key)    
    loss_val = jnp.mean(loss_array)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates) 
    
    jax.debug.print('loss {}:', loss_val)
    return params, opt_state, keys[0]
    
#c=jax.xla_computation(get_grad)(params, jnp.ones((64, 3)))
#with open("t.dot", "w") as f:
#    f.write(c.as_hlo_dot_graph()) 
# key = random.PRNGKey(0) 
# for i in range(10000000):    
#     j = np.random.randint(0, 2)
#     key, _ = random.split(key)
#     params, opt_state = train_step(params, (-jnp.ones(( 60, 3))/2., j*jnp.ones(( 60, 3))/2.), jnp.ones((1))*j, opt_state, key)

data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']

print(f'Images size : {images.shape}')
print(f'Pose : {poses[0]}')
print(f'Focal length: {focal}')


    

key = random.PRNGKey(0) 
H, W = 100, 100
batch_size =  2000 

num_splits = (H * W) / batch_size
get_rays_jit = jit(lambda pose: get_rays(H=H, W=W, focal=focal, pose=pose, num_splits=num_splits))



for i in range(10000000):    
    img_idx = np.random.randint(0, len(images) - 10)
    image_train = images[img_idx]

    pose = poses[img_idx]
    
    origins, directions = get_rays_jit(pose)
    
    key, _ = random.split(key) 
    params, opt_state, key = train_step(params, (origins, directions), image_train, opt_state, key)
    
    
    if i%50 == 0:
        print('Begin eval')
        origins, directions = get_rays_jit(poses[101])
        c_array = []
        for b_origins, b_directions in zip(origins, directions):
            b_keys = random.split(key, len(b_origins))
            c = render(model, params, b_origins, b_directions, b_keys) 
            c_array.append(c)
            
        c_array = jnp.concatenate(c_array) 
        c_array = jnp.reshape(c_array, [100, 100, 3])
        
        c_array = np.array(c_array) 
        pred_img = (c_array * 255.).astype(np.uint8)[:,:,::-1]
        actual_img = (images[101]* 255.).astype(np.uint8)[:,:,::-1]
        print('max image',np.max(pred_img))
        cv2.imwrite(f'/tmp/pred_img{i}.jpg', pred_img)
        cv2.imwrite(f'/tmp/actual.jpg', actual_img)
    
