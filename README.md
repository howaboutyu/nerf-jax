[![pytest](https://github.com/higgsboost/nerf-jax/actions/workflows/pytest.yml/badge.svg)](https://github.com/higgsboost/nerf-jax/actions/workflows/pytest.yml)

# nerf-Jax
This repository contains a JAX implementation of the NeRF algorithm for synthesizing novel views of a scene from sparse input views. It provides functionality for converting video files to NeRF datasets, training and evaluating NeRF models on GPU and TPU-VM.

## Installation 

```bash
pip install -r requirements.txt
apt install libgl1 # for opencv
```


### Jax

Please refer to the offical install [documentation](https://github.com/google/jax#installation)



## Data - Video to Nerf Dataset
To convert video files to Nerf datasets, a GPU is required because `COLMAP` currently requires a GPU. 

### Local Conversion
To convert a video file to a Nerf dataset on your local machine, you can use the following command:

```bash
make vid_to_nerf VID_FILE=path/to/video_file OUT_PATH=path/to/output_folder
```

This command will use a Docker container to run the necessary conversion scripts. The output folder will contain a set of images and poses.

### Cloud Conversion
To convert a video file to a Nerf dataset on the Google Cloud Platform, you can use the following commands:

```bash
# 1) Create a GPU VM and run setup scripts
make start_gpu_convert

# 2) Convert the video file to a Nerf dataset
make vid_to_nerf_cloud VID_FILE=path/to/video_file OUT_PATH=path/to/output_folder

# 3) Delete the GPU VM
make delete_gpu_vm
```

## Training 

```bash
python main.py --config_file=<config file>
```

where <config file> is the path to the configuration file containing the necessary parameters for the training. 


