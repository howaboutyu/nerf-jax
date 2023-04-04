

##  JAX Implementation of NeRF for Scene Synthesis

[![pytest](https://github.com/higgsboost/nerf-jax/actions/workflows/pytest.yml/badge.svg)](https://github.com/higgsboost/nerf-jax/actions/workflows/pytest.yml)

[Accompanying blog post](https://www.david-yu.com/blog/nerf/nerf.html#jax-learnings)


## Overview

This repository offers a JAX-based implementation of the [NeRF](https://arxiv.org/abs/2003.08934) method to synthesize novel views of a scene from sparse input data. It enables users to convert video files into NeRF datasets and train & evaluate NeRF models on GPU and TPU-VM. 


## Getting Started

### Installation

To install the required packages, run the following commands:

```bash
pip install -r requirements.txt
apt install libgl1 # for opencv
```

For JAX installation, refer to the official [installation documentation](https://github.com/google/jax#installation).

### Converting Video to NeRF Dataset

Convert a video file to a NeRF dataset using the following command on your local machine:

```bash
# Example converting an IPhone video to NeRF dataset
VID_FILE=IMG_1425.MOV
OUT_PATH=llff_datasets/lego_miso

mkdir -p $OUT_PATH/images
docker run --gpus all -v`pwd`:/nerf -i bmild/tf_colmap bash -c \
                "git clone https://github.com/Fyusion/LLFF; \
                ffmpeg -i /nerf/$VID_FILE -vf fps=2  -pix_fmt bgr8 /nerf/$OUT_PATH/images/img%03d.png; \
                python LLFF/imgs2poses.py /nerf/$OUT_PATH"
```

This command leverages the docker image from [LLFF](https://github.com/Fyusion/LLFF) to run the necessary conversion scripts. Note that a GPU is required for this process, as [COLMAP](https://colmap.github.io/index.html) needs a GPU for feature extraction and matching.

`ffmpeg` extracts frames from the input video file at a rate of 2 frames per second, while `imgs2poses.py` computes camera poses for each frame. The final dataset, suitable for NeRF model training, contains both image files and computed poses.

### Training NeRF Models

Train your NeRF model using the following command:

```bash
python main.py --config_file=<config file>
```

Replace `<config file>` with the path to the configuration file containing the necessary training parameters. For training on TPU set `num_devices` to the corresponding value, i.e. for v3-8 `num_devices=8`


### Rendering NeRF

After training, the trained NeRF model can be rendered with the following:

```bash
python3 main.py --config_path=<config file>  --mode=render --render_output_folder=<rendered output folder>
```

## Resources 

Other implementations of NeRF:

* [Original NeRF implementation in tf](https://github.com/bmild/nerf) by `@bmild`
* [NeRF in Jax](https://github.com/google-research/google-research/tree/master/jaxnerf) by `@google-research`

ImageNet example using `Flax` by `@google`:

* [ImageNet example](https://github.com/google/flax/tree/main/examples/imagenet)

