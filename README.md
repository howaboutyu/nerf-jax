[![pytest](https://github.com/higgsboost/nerf-jax/actions/workflows/pytest.yml/badge.svg)](https://github.com/higgsboost/nerf-jax/actions/workflows/pytest.yml)

# nerf-Jax                  

## Install jax on GPU

For CUDA 11 simply run

```bash
./scripts/setup_jax.sh
```

and for TPU-vm

```bash
./script/setup_tpu.sh
```

# 

## Data - video to Nerf dataset

Conversion from video files to nerf datasets requires a GPU. You can convert it locally - if you have a GPU locally - or on the google cloud platform.


### Cloud GPU example

```bash
# 1) Setup
make start_gpu_convert:
# 2) Convert 
make vid_to_nerf_cloud VID_FILE=IMG_123.MOV OUT_PATH=IMG_123
# 3) Delete gpu vm 
make delete_gpu_vm
```
After, you will have a folder of images and poses in the specified `OUT_PATH` folder.

### Local GPU example
WIP


## Training - GPU and TPU
WIP

First, create a TPU-VM; by default, it is a preemptible TPU.

```bash
make create_gpu_vm
```

Then you can start training. In the example below, the specified `CONFIG_PATH` should exist locally and has Nerf configuration parameters, and `DATA_PATH` is the output of the video-to-nerf data generation scripts outlined above.

```bash
make train_tpu CONFIG_PATH=configs/tpu.YAML DATA_PATH=miso_shop
```

After running this command, the training will start on the TPU-VM within a `tmux` session called `nerf`. `ssh` into the VM and attach to the session to see the logs (`tmux a -t nerf`).
