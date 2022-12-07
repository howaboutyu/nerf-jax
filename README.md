# nerf-Jax                  

## Data - video to Nerf dataset

Conversion from video files to nerf datasets requires a GPU. You can convert it locally - if you have a GPU locally - or on the google cloud platform.


### Cloud GPU example

```bash
# 1) Setup
make start_gpu_convert:
# 2) convert - example
make vid_to_nerf_cloud VID_FILE=IMG_123.MOV OUT_PATH=IMG_123
```
After, you will have a folder of images and poses in the specified `OUT_PATH` folder.

### Local GPU example
WIP


## Training - GPU and TPU
WIP

First, create a TPU-VM - by default, it is a preemptible TPU.

```bash
make create_gpu_vm
```

Second, start training where the specified `CONFIG_PATH` exists locally, and `DATA_PATH` is the output of the video to nerf data generation scripts.

```bash
make train_tpu CONFIG_PATH=configs/tpu.YAML DATA_PATH=miso_shop
```

After running this command, a `tmux` session `nerf` will be created, so attach to that session to see the logs - `tmux a -t nerf`.
