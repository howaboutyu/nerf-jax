build:
	@echo "building a jax image with gpu"


start:
	wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest bash

