import jax
import jax.numpy as jnp 
import os
import json 
import cv2
from dataclasses import dataclass, field
from typing import List 
import numpy as np


def get_rays(H, W, focal, pose):
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
    max_eval: int = 2 
    imgs: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
    poses: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
        
    def __iter__(self):
        self.n = 0
        return self 

    def __next__(self):
        if self.n > self.max_eval and self.subset == 'val':
            raise StopIteration 

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
        self.subset = subset 
        
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


        json_p = os.path.join(self.data_path, f'transforms_{self.subset}.json')
        
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

class LLFF(Dataset): 
    '''
    most of this is from: https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/datasets.py
    '''
    
    def __init__(self, config, data_path='nerf_llff_data/fern', subset='train'):
        
        self.data_path = data_path
        
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

        self.subset = subset 


    def get_raw_data(self):
        img_paths = sorted(os.listdir(os.path.join(self.data_path, 'images')))

        images = np.array([self.normalizer(cv2.imread(os.path.join(self.data_path, 'images',  ip))) for ip in img_paths])


        if self.scale: 
            images = np.array([cv2.resize(img, dsize=None, fx=self.scale, fy=self.scale) for img in images])


        
        poses_arr = np.load(os.path.join(self.data_path, 'poses_bounds.npy'))

        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])

        factor = 1./self.scale 
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / factor


        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale
    
        # Recenter poses.
        poses = self._recenter_poses(poses)

        self.imgs = images
        self.poses = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]

        self.H, self.W = images.shape[1:3]

        print('focal ', self.focal)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses
    
    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w
    
    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m
    
    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

        

       
        
def dataset_factory(config):
    if config['dataset_name'] == 'lego':
        return {
            'train': LegoDataset(config, subset='train'),
            'val': LegoDataset(config, subset='val'),
            'test': LegoDataset(config, subset='test'),
        }

    elif config['dataset_name'] == 'fern':
        return {
            'train': LLFF(config, subset='train'),
            'val': LLFF(config, subset='val'),
        }

if __name__ == '__main__':
    import yaml
    with open('configs/lego.yaml') as file:
        config = yaml.safe_load(file)

    dataset = LLFF(config=config) 

    #for a in dataset:
    #    print(a)
    
     
    
