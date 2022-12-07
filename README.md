# nerf-jax

## Data - video to Nerf dataset

Conversion from video files to nerf datasets require a GPU. You can either convert it locally - if you have a gpu locally - or on google cloud platform.

For example, if you take a video with the name `IMG_123.MOV`, then you can convert to Nerf dataset using the following examples.
### Cloud GPU example

```bash
# 1) setup
make start_gpu_convert:
# 2) convert - example
make vid_to_nerf_cloud VID_FILE=IMG_123.MOV OUT_PATH=IMG_123
```
After

### Local GPU example
WIP


## Training - GPU and TPU
WIP


