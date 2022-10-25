import jax
import jax.numpy as jnp 
import os
import json 
import cv2
from dataclasses import dataclass, field
from typing import List 


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

def split2patches(data, n_w, n_h):
    patches_array = jnp.hsplit(data, n_h) 
    patches_array = jnp.array([jnp.vsplit(p, n_w) for p in patches_array])
    patches = jnp.concatenate(patches_array)
    return patches

def patches2data(img, n_v):
    data = jnp.hstack([jnp.vstack(s) for s in jnp.split(img, n_v)])
    return data

@dataclass
class Dataset:    
    W: float    
    H: float
    focal: float
    split_to_patch: bool
    split_w: int
    split_h: int
    batch_size: int = 1
    use_batch: bool = False
    imgs: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
    poses: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
        
    def __iter__(self):
        self.n = 0
        return self 

    def __next__(self):
        if self.n < len(self.imgs):
            img_batch, origins_batch, directions_batch = [], [], []
            for _ in range(self.batch_size): 
                if self.n not in self.cache:
                    origins, directions = self.get_rays_jit(self.poses[self.n])
                    img = self.imgs[self.n]
                    if self.split_to_patch: 
                        img, origins, directions = [split2patches(data, self.split_w, self.split_h) \
                            for data in [img, origins, directions] ]
    
                    self.cache[self.n] = [origins, directions, img]
                else:
                    origins, directions, img = self.cache[self.n]
                img_batch.append(img)
                origins_batch.append(origins)
                directions_batch.append(directions)
                
                self.n += 1
             
            return jnp.array(img_batch), jnp.array(origins_batch), jnp.array(directions_batch)
        else:
            raise StopIteration
        


class LegoDataset(Dataset): 
    
    def __init__(self, config, data_path='nerf_synthetic/lego', subset='train'):
        
        self.data_path = data_path
        self.key_to_data = subset 
        
        self.normalizer = lambda x : x/255.
                
        self.scale = config['scale'] 
        
        self.split_w = config['split_w']
        self.split_h = config['split_h']

        self.split_to_patch = config['split_to_patches']

        if config['use_batch']:
            self.batch_size = jax.local_device_count() 
            print(f'Using batch mode with {self.batch_size} local devices')

        self.get_raw_data()
        
        self.get_rays_jit = jax.jit(lambda pose: get_rays(self.H, self.W, self.focal, pose))
        
        self.cache = dict() 


    def get_raw_data(self):


        json_p = os.path.join(self.data_path, f'transforms_{self.key_to_data}.json')
        
        with open(json_p, 'r') as fp: 
            transforms = json.load(fp)
        
        imgs, poses = [], [] 
        for t in transforms['frames']:
            img = cv2.imread(os.path.join(self.data_path, t['file_path'] + '.png'))
            
            if self.scale: img = cv2.resize(img, dsize=None, fx=self.scale, fy=self.scale)

            pose = jnp.array(t['transform_matrix'])
            imgs.append(self.normalizer(img))
            poses.append(pose)

        self.imgs = jnp.array(imgs) 
        self.poses = jnp.array(poses)         

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(transforms['camera_angle_x'])
        focal = .5 * W / jnp.tan(.5 * camera_angle_x)
        
        self.H = H
        self.W = W
        self.focal = focal 

        
        
def dataset_factory(config):
    if config['dataset_name'] == 'lego':
        return {
            'train': LegoDataset(config, subset='train'),
            'val': LegoDataset(config, subset='val'),
            'test': LegoDataset(config, subset='test'),
        }

if __name__ == '__main__':
    dataset = LegoDataset() 
    
    for _ in range(10): 
        for i, a in enumerate(dataset):
            print(i)
    
    
