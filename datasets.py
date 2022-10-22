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


@dataclass
class Dataset:    
    W: float    
    H: float
    focal: float
    imgs: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
    poses: List[jnp.array] = field(default_factory=lambda: jnp.array([]))

        
    def __iter__(self):
        self.n = 0
        return self 

    def __next__(self):
        if self.n < len(self.imgs):
            self.n += 1
            
            if self.n not in self.cache:
                origins, directions = self.get_rays_jit(self.poses[self.n])
                self.cache[self.n] = [origins, directions]
            else:
                origins, directions = self.cache[self.n]
            return self.imgs[self.n], origins, directions 
        else:
            raise StopIteration
        


class LegoDataset(Dataset): 
    
    def __init__(self, data_path='nerf_synthetic/lego', subset='train', half_res=True):
        
        self.data_path = data_path
        self.key_to_data = subset 
        
        self.normalizer = lambda x : x/255.
                
        self.half_res = half_res

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
            
            if self.half_res: img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

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

        if self.half_res:
            self.H = self.H//2
            self.W = self.W//2
            self.focal = self.focal/2.
        
        


if __name__ == '__main__':
    dataset = LegoDataset() 
    
    for _ in range(10): 
        for i, a in enumerate(dataset):
            print(i)
    
    