'''
Datasets for nerf
Many functions are taken from: https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/datasets.py
'''

import jax
import jax.numpy as jnp 
import os
import json 
import cv2
from dataclasses import dataclass, field
from typing import List, Any
import numpy as np
import time


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

@dataclass
class Dataset:    
    W: float    
    H: float
    focal: float
    near: float
    far: float
    split_frac: float = 0.9 
    mini_batch_size: int = 1024 
    batch_size: int = 1
    use_batch: bool = False
    max_eval: int = 2 
    imgs: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
    poses: List[jnp.array] = field(default_factory=lambda: jnp.array([]))
    key: Any = field(default=jax.random.PRNGKey(0)) 


    def get(self, idx):
        img = self.imgs[idx]
        pose = self.poses[idx]
        origins, directions = self.get_rays_jit(self.poses[idx])

        return img, origins, directions
        

    def __iter__(self):
        self.n = 0
        return self 

    def __next__(self):
        if self.subset == 'render':
            if self.n < len(self.render_poses):
                pose = self.render_poses[self.n]
                origins, directions = self.get_rays_jit(pose)
                self.n += 1
                return origins, directions 
            else:
                raise StopIteration
        else:
            tic = time.perf_counter()
            if self.n > self.max_eval and self.subset == 'val':
                raise StopIteration 
             
            if self.n < len(self.imgs):
                img_batch, origins_batch, directions_batch = [], [], []
                
                # Populate the batch - batch_size should be 1 if GPU  and >1 if TPU
                while len(img_batch) < self.batch_size: 
                    rand_image_selector = np.random.randint(0, len(self.imgs))
                    print(f'Getting image index : {rand_image_selector}')
                    if rand_image_selector not in self.cache:
                        img, origins, directions = self.get(rand_image_selector)
                        origins = origins.reshape((-1, 3))
                        directions = directions.reshape((-1, 3))
                        img = img.reshape((-1, 3))
    
                        self.cache[rand_image_selector] = [origins, directions, img]
                    else:
                        origins, directions, img = self.cache[rand_image_selector]

                    # TODO : use np instead of jnp 
                    rand_idx = jax.random.randint(self.key, (self.mini_batch_size, 1), 0, len(img)) 
                    rand_idx = jnp.squeeze(rand_idx)
    
                    img = img[rand_idx]
                    origins = origins[rand_idx]
                    directions = directions[rand_idx]
    
                    img_batch.append(img)
                    origins_batch.append(origins)
                    directions_batch.append(directions)
                    
                    self.n += 1
                    self.key, _ = jax.random.split(self.key)
                toc = time.perf_counter()
                print(f"getting one batch took {toc - tic:0.4f} seconds")
                return jnp.array(img_batch), jnp.array(origins_batch), jnp.array(directions_batch)
            else:
                raise StopIteration
        
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

    
    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]
    
        def min_line_dist(rays_o, rays_d):
          a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
          b_i = -a_i @ rays_o
          pt_mindist = np.squeeze(-np.linalg.inv(
              (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
          return pt_mindist
    
        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
            np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad**2 - zh**2)
        new_poses = []
    
        for th in np.linspace(0., 2. * np.pi, 120):
          camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
          up = np.array([0, 0, -1.])
          vec2 = self._normalize(camorigin)
          vec0 = self._normalize(np.cross(vec2, up))
          vec1 = self._normalize(np.cross(vec2, vec0))
          pos = camorigin
          p = np.stack([vec0, vec1, vec2, pos], 1)
          new_poses.append(p)
    
        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate([
            new_poses,
            np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        return poses_reset 
        return new_poses 
    
    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable "focus depth" for this dataset.
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
          c = np.dot(c2w[:3, :4], (np.array(
              [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
          z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
          render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4] 

class LegoDataset(Dataset): 
    
    def __init__(self, config, data_path='nerf_synthetic/lego', subset='train'):
        
        self.data_path = data_path
        self.subset = subset 
        
        self.normalizer = lambda x : x/255.

        self.near = config['near']
        self.far = config['far']
                
        self.scale = config['scale'] 
        
        if 'mini_batch_size' in config:
            self.mini_batch_size = config['mini_batch_size']

        if config['use_batch']:
            self.batch_size = jax.local_device_count() 
            print(f'Using batch mode with {self.batch_size} local devices')

        self.get_raw_data()
        
        self.get_rays_jit = jax.jit(lambda pose: get_rays(self.H, self.W, self.focal, pose))
        
        self.cache = dict() 


    def get_raw_data(self):


        json_p = os.path.join(self.data_path, f'transforms_{self.subset}.json')

        if self.subset == 'render':

            json_p = os.path.join(self.data_path, f'transforms_train.json')

        
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
        if self.subset == 'render':
            self._generate_spiral_poses(self.poses, jnp.array([2., 16.]))



class LLFF(Dataset): 
    '''
    '''
    
    def __init__(self, config, subset='train'):
         
        self.data_path =config['data_path'] 
        
        self.normalizer = lambda x : x/255.
                
        self.scale = config['scale'] 
        
        if config['use_batch']:
            self.batch_size = jax.local_device_count() 
            print(f'Using batch mode with {self.batch_size} local devices')

        if 'mini_batch_size' in config:
            self.mini_batch_size = config['mini_batch_size']


        self.subset = subset 

        self.get_raw_data()
        
        self.get_rays_jit = jax.jit(lambda pose: get_rays(self.H, self.W, self.focal, pose))
        
        self.cache = dict() 



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

        if self.subset == 'render':
            self._generate_spiral_poses(poses, bds)



        self.focal = poses[0, -1, -1]
        
        sample_n = int(len(images) * self.split_frac)
        if self.subset == 'train':
            self.imgs = images[:sample_n]
            self.poses = poses[:, :3, :4][:sample_n]
        elif self.subset == 'val':
            self.imgs = images[sample_n:]
            self.poses = poses[:, :3, :4][sample_n:]
        else:
            self.imgs = images
            self.poses = poses[:, :3, :4]

        self.H, self.W = images.shape[1:3]

        self.near = np.min(bds) * .9
        self.far = np.max(bds) * 1.

        print('focal ', self.focal)
        print('far ', self.far)
        print('near', self.near)


      
        
def dataset_factory(config):
    if config['dataset_name'] == 'lego':
        return {
            'train': LegoDataset(config, subset='train'),
            'val': LegoDataset(config, subset='val'),
            'render': LegoDataset(config, subset='render'),
        }

    elif config['data_type'] == 'llff':
        return {
            'train': LLFF(config, subset='train'),
            'val': LLFF(config, subset='val'),
            'render': LLFF(config, subset='render'),
        }

if __name__ == '__main__':
    import yaml
    with open('configs/beer.yaml') as file:
        config = yaml.safe_load(file)

    dataset = LLFF(config=config, subset='render') 

    for p in dataset.poses:
        print(p.shape)
    
