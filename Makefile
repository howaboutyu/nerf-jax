build:
	@echo "building a jax image with gpu"
	sudo docker build . -t jax-gpu -f Dockerfile.gpu


start:
	wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest bash

train_lego:
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest python3 train.py


create_tpu_vm:
	gcloud compute tpus tpu-vm create nerf \
		--zone europe-west4-a \
		--accelerator-type v3-8 \
		--version tpu-vm-base \
		--preemptible

start_tpu_vm:
	gcloud compute tpus tpu-vm start nerf --zone europe-west4-a 

connect_tpu_vm:
	gcloud compute tpus tpu-vm ssh nerf --zone europe-west4-a

setup_vm:
	pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

delete_tpu_vm:
	gcloud compute tpus tpu-vm delete nerf  --zone europe-west4-a

list_tpu_vm:
	gcloud compute tpus tpu-vm list --zone europe-west4-a
