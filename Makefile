build:
	@echo "building a jax image with gpu"
	sudo docker build . -t jax-gpu -f Dockerfile.gpu


get_mini_nerf_data:
	wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz

get_nerf_example_data:
	wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
	unzip nerf_example_data.zip

start:
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest bash

train_lego:
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest python3 train.py

llff_pose_from_vid:
	echo 'let.s get it'

	#mkdir LLFF && cd LLFF && 
	#git clone https://github.com/Fyusion/LLFF
	#sudo docker run --gpus all -v`pwd`:/nerf -it bmild/tf_colmap bash

vid_to_nerf:
	echo 'Converting' $(VID_FILE)', the output will be in' $(OUT_PATH)
	
	mkdir -p $(OUT_PATH)/images
	sudo docker run --gpus all -v`pwd`:/nerf -it bmild/tf_colmap bash -c \
		"ffmpeg -i /nerf/$(VID_FILE) -vf fps=2 /nerf/$(OUT_PATH)/images/img%03d.jpg; \
		python /nerf/LLFF/imgs2poses.py /nerf/$(OUT_PATH)"

create_tpu_vm:
	gcloud compute tpus tpu-vm create nerf \
		--zone europe-west4-a \
		--accelerator-type v3-8 \
		--version tpu-vm-base \
		--preemptible

start_tpu_vm:
	gcloud compute tpus tpu-vm start nerf --zone europe-west4-a 

stop_tpu_vm:
	gcloud compute tpus tpu-vm stop nerf --zone europe-west4-a 


connect_tpu_vm:
	gcloud compute tpus tpu-vm ssh nerf --zone europe-west4-a

setup_vm:
	pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

delete_tpu_vm:
	gcloud compute tpus tpu-vm delete nerf  --zone europe-west4-a

list_tpu_vm:
	gcloud compute tpus tpu-vm list --zone europe-west4-a
