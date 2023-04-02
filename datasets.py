"""
Datasets for nerf
Ref: https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/datasets.py
TODO: Clean this script
"""

import jax
import jax.numpy as jnp
import os
import json
import cv2
from dataclasses import dataclass, field
from typing import List, Any
import numpy as np
import time

from nerf_config import NerfConfig


def get_rays(H, W, focal, pose):
    x, y = jnp.meshgrid(
        jnp.arange(W, dtype=jnp.float32),
        jnp.arange(H, dtype=jnp.float32),
        indexing="xy",
    )
    x = (x - 0.5 * W) / focal
    y = -(y - 0.5 * H) / focal

    direction = jnp.stack([x, y, -jnp.ones_like(x)], -1)

    rot = pose[:3, :3]
    direction = (direction[..., jnp.newaxis, :] * rot).sum(-1)

    # Normalize direction
    direction_norm = jnp.linalg.norm(direction, axis=-1)
    direction = direction / direction_norm[..., jnp.newaxis]

    translation = pose[:3, 3]
    origin = jnp.broadcast_to(translation, direction.shape)

    return origin, direction


# Modified from : https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
def trans(x=0, y=0, z=0):
    return jnp.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )


def rot_phi(phi):
    return jnp.array(
        [
            [1, 0, 0, 0],
            [0, jnp.cos(phi), -jnp.sin(phi), 0],
            [0, jnp.sin(phi), jnp.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )


def rot_theta(th):
    return jnp.array(
        [
            [jnp.cos(th), 0, -jnp.sin(th), 0],
            [0, 1, 0, 0],
            [jnp.sin(th), 0, jnp.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )


def rot_z(th):
    return jnp.array(
        [
            [jnp.cos(th), -jnp.sin(th), 0, 0],
            [jnp.sin(th), jnp.cos(th), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans(z=radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def generate_render_poses(thetas, phis, radii):
    num_poses = len(thetas)
    new_poses = []
    for i in range(num_poses):
        c2w = pose_spherical(thetas[i], phis[i], radii[i])
        new_poses.append(c2w)
    return np.stack(new_poses, axis=0)


@dataclass
class Dataset:
    W: float
    H: float
    focal: float
    near: float
    far: float
    split_frac: float = 0.9
    batch_size: int = 1024  # <- number of rays per batch
    num_devices: int = 1  # <- number of devices, i.e. number of GPUs
    max_eval: int = 2
    imgs: List[jnp.array] = field(default_factory=lambda: [])
    poses: List[jnp.array] = field(default_factory=lambda: [])

    @property
    def num_examples(self):
        return len(self.imgs)

    def get(self, idx):
        img = self.imgs[idx]
        pose = self.poses[idx]
        origins, directions = self.get_rays_jit(self.poses[idx])

        return img, origins, directions

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.subset == "render":
            if self.n < len(self.render_poses):
                pose = self.render_poses[self.n]
                origins, directions = self.get_rays_jit(pose)
                self.n += 1
                return origins, directions
            else:
                raise StopIteration
        else:
            tic = time.perf_counter()
            if self.n > self.max_eval and self.subset == "val":
                raise StopIteration

            if self.n < len(self.imgs):
                img_batch, origins_batch, directions_batch = [], [], []

                while len(img_batch) < self.num_devices:
                    rand_image_selector = np.random.randint(0, len(self.imgs))
                    if rand_image_selector not in self.cache:
                        img, origins, directions = self.get(rand_image_selector)
                        origins = origins.reshape((-1, 3))
                        directions = directions.reshape((-1, 3))
                        img = img.reshape((-1, 3))

                        self.cache[rand_image_selector] = [origins, directions, img]
                    else:
                        origins, directions, img = self.cache[rand_image_selector]

                    rand_idx = jax.random.randint(
                        self.key, (self.batch_size, 1), 0, len(img)
                    )
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
                return (
                    jnp.array(img_batch),
                    jnp.array(origins_batch),
                    jnp.array(directions_batch),
                )
            else:
                raise StopIteration

    def gen_render_poses(self, poses, num_poses=5, spherical=False):
        """
        Generates some render poses
        """

        if spherical:
            thetas = np.linspace(0, 360, num_poses)
            phis = np.full(num_poses, -30)
            radii = np.full(num_poses, self.far * 0.8)
            render_poses = generate_render_poses(thetas, phis, radii)
            self.render_poses = render_poses
        else:
            c2w = self.poses.mean(0)
            last_row = np.array([0, 0, 0, 1])
            c2w = np.vstack([c2w, last_row])
            render_poses = []
            for t in np.arange(-1, 1, 0.1):
                transformed_c2w = rot_z(t) @ trans(x=t) @ c2w
                render_poses.append(transformed_c2w)

            self.render_poses = render_poses

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
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


class LegoDataset(Dataset):
    def __init__(self, config: NerfConfig, subset: str):
        self.data_path = config.dataset_path
        self.subset = subset

        self.normalizer = lambda x: x / 255.0

        self.near = config.near
        self.far = config.far

        self.scale = config.scale

        self.batch_size = config.batch_size

        self.num_devices = config.num_devices

        self.get_raw_data()

        self.get_rays_jit = jax.jit(
            lambda pose: get_rays(self.H, self.W, self.focal, pose)
        )

        self.cache = dict()

        self.key = jax.random.PRNGKey(0)

    def get_raw_data(self):
        json_p = os.path.join(self.data_path, f"transforms_{self.subset}.json")

        if self.subset == "render":
            json_p = os.path.join(self.data_path, f"transforms_train.json")

        with open(json_p, "r") as fp:
            transforms = json.load(fp)

        imgs, poses = [], []
        for t in transforms["frames"]:
            img = cv2.imread(os.path.join(self.data_path, t["file_path"] + ".png"))

            if self.scale:
                img = cv2.resize(img, dsize=None, fx=self.scale, fy=self.scale)

            pose = jnp.array(t["transform_matrix"])
            imgs.append(self.normalizer(img))
            poses.append(pose)

        self.imgs = jnp.array(imgs)
        self.poses = jnp.array(poses)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(transforms["camera_angle_x"])
        focal = 0.5 * W / jnp.tan(0.5 * camera_angle_x)

        self.H = H
        self.W = W
        self.focal = focal
        if self.subset == "render":
            self.gen_render_poses(self.poses, spherical=True)


class LLFF(Dataset):
    """ """

    def __init__(self, config: NerfConfig, subset: str = "train"):
        self.normalizer = lambda x: x / 255.0

        self.config = config
        self.subset = subset
        self.scale = config.scale
        self.data_path = config.dataset_path

        self.get_raw_data()

        self.get_rays_jit = jax.jit(
            lambda pose: get_rays(self.H, self.W, self.focal, pose)
        )

        self.cache = dict()

        self.key = jax.random.PRNGKey(0)

    def get_raw_data(self):
        img_paths = sorted(os.listdir(os.path.join(self.data_path, "images")))

        images = np.array(
            [
                self.normalizer(cv2.imread(os.path.join(self.data_path, "images", ip)))
                for ip in img_paths
                if "png" in ip or "jpg" in ip
            ]
        )

        if self.scale:
            images = np.array(
                [
                    cv2.resize(img, dsize=None, fx=self.scale, fy=self.scale)
                    for img in images
                ]
            )

        poses_arr = np.load(os.path.join(self.data_path, "poses_bounds.npy"))

        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])

        factor = 1.0 / self.scale
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
        )
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        scale = 1.0 / (bds.min() * 0.75)
        poses[:, :3, 3] *= scale
        bds *= scale

        # Recenter poses.
        poses = self._recenter_poses(poses)

        self.focal = poses[0, -1, -1]

        sample_n = int(len(images) * self.split_frac)
        if self.subset == "train":
            self.imgs = images[:sample_n]
            self.poses = poses[:, :3, :4][:sample_n]
        elif self.subset == "val":
            self.imgs = images[sample_n:]
            self.poses = poses[:, :3, :4][sample_n:]
        else:
            self.imgs = images
            self.poses = poses[:, :3, :4]

        self.H, self.W = images.shape[1:3]

        self.near = np.min(bds) * 0.9
        self.far = np.max(bds) * 1.0

        if self.subset == "render":
            self.gen_render_poses(poses)

        print("focal ", self.focal)
        print("far ", self.far)
        print("near", self.near)


def dataset_factory(config):
    """
    Given a NeRF configuration object, return a dictionary of datasets
    for training, validation, and rendering based on the specified dataset type.

    Args:
        config: A NeRF configuration object.

    Returns:
        A dictionary of datasets for training, validation, and rendering.
        The dictionary keys are "train", "val", and "render", and the values
        are dataset objects based on the specified dataset type.

    Raises:
        ValueError: If the dataset type is not "lego" or "llff".
    """

    if config.dataset_type == "lego":
        return {
            "train": LegoDataset(config, subset="train"),
            "val": LegoDataset(config, subset="val"),
            "render": LegoDataset(config, subset="render"),
        }

    elif config.dataset_type == "llff":
        return {
            "train": LLFF(config, subset="train"),
            "val": LLFF(config, subset="val"),
            "render": LLFF(config, subset="render"),
        }
    else:
        raise ValueError(
            "Invalid dataset type. Must be 'lego' or 'llff', please implement."
        )
