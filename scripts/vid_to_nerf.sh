#!/bin/bash
set -e
VID_FILE=IMG_1425.MOV
OUT_PATH=llff_datasets/lego_miso

mkdir -p $OUT_PATH/images
docker run --gpus all -v`pwd`:/nerf -i bmild/tf_colmap bash -c \
		"git clone https://github.com/Fyusion/LLFF; \
                ffmpeg -i /nerf/$VID_FILE -vf fps=2  -pix_fmt bgr8 /nerf/$OUT_PATH/images/img%03d.png; \
                python LLFF/imgs2poses.py /nerf/$OUT_PATH"
