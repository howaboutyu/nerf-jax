
GCP_GPU_VM = 'nerf-gpu-instance'

build:
	@echo "building a jax image with gpu"
	sudo docker build . -t jax-gpu -f Dockerfile.gpu


get_mini_nerf_data:
	# Unused in this repo
	wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz
	unzip tiny_nerf_data.npz

get_nerf_example_data:
	wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
	unzip nerf_example_data.zip

start:
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest bash

train_lego:
	sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest python3 train.py

start_gpu_convert:
	# create gpu vm and run setup scripts - install cuda, docker, nerf-jax repo
	gcloud compute instances create $(GCP_GPU_VM) \
    --machine-type n1-standard-2 \
    --zone us-east1-d \
    --boot-disk-size 40GB \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --image-family ubuntu-1804-lts \
    --image-project ubuntu-os-cloud \
    --maintenance-policy TERMINATE --restart-on-failure

	gcloud compute scp scripts/setup_gpu.sh $(GCP_GPU_VM):/tmp
	gcloud compute ssh $(GCP_GPU_VM) --command \
		'bash /tmp/setup_gpu.sh' 

delete_gpu_vm:
	gcloud compute instances delete $(GCP_GPU_VM)

vid_to_nerf_cloud:
	# cloud vid -> nerf converter

	echo Converting $(VID_FILE) on $(GCP_GPU_VM) the output will be in $(OUT_PATH)
	gcloud compute scp $(VID_FILE) $(GCP_GPU_VM):~/nerf

	gcloud compute ssh $(GCP_GPU_VM) --command \
		"cd ~/nerf && sudo make vid_to_nerf VID_FILE=$(VID_FILE) OUT_PATH=$(OUT_PATH)"

	gcloud compute scp --recurse $(GCP_GPU_VM):~/nerf/$(OUT_PATH) .

vid_to_nerf:
	# Local vid -> nerf converter
	echo 'Converting' $(VID_FILE)', the output will be in' $(OUT_PATH)
	
	mkdir -p $(OUT_PATH)/images
	sudo docker run --gpus all -v`pwd`:/nerf -i bmild/tf_colmap bash -c \
		"ffmpeg -i /nerf/$(VID_FILE) -vf fps=2 /nerf/$(OUT_PATH)/images/img%03d.png; \
		python /nerf/LLFF/imgs2poses.py /nerf/$(OUT_PATH)"

create_tpu_vm:
	# Setup 
	-gcloud compute tpus tpu-vm create nerf \
		--zone europe-west4-a \
		--accelerator-type v3-8 \
		--version tpu-vm-base \
		--preemptible
	
	gcloud compute tpus tpu-vm scp --zone europe-west4-a scripts/setup_tpu.sh nerf:/tmp 
	gcloud compute tpus tpu-vm ssh --zone europe-west4-a nerf --command 'bash /tmp/setup_tpu.sh'

train_tpu:
	
	# Copy over config and data 
	# config -> tpu
	# nerf data -> tpu
	gcloud compute tpus tpu-vm scp --zone europe-west4-a $(CONFIG_PATH) nerf:~/nerf/configs 
	gcloud compute tpus tpu-vm scp --recurse --zone europe-west4-a $(DATA_PATH) nerf:~/nerf/llff_data

	gcloud compute tpus tpu-vm scp --zone europe-west4-a scripts/setup_tpu.sh nerf:/tmp 
	gcloud compute tpus tpu-vm ssh --zone europe-west4-a nerf --command \
		'cd ~/nerf && tmux new-session -d -s nerf "python3 main.py --config_path=$(CONFIG_PATH) --mode=train"'

start_tpu_vm:
	gcloud compute tpus tpu-vm start nerf --zone europe-west4-a 

stop_tpu_vm:
	gcloud compute tpus tpu-vm stop nerf --zone europe-west4-a 

connect_tpu_vm:
	gcloud compute tpus tpu-vm ssh nerf --zone europe-west4-a

delete_tpu_vm:
	gcloud compute tpus tpu-vm delete nerf  --zone europe-west4-a

list_vm:
	echo '############### TPU ##############'
	gcloud compute tpus tpu-vm list --zone europe-west4-a
	echo '############### GPU ##############'
	gcloud compute instances list


test:
	CUDA_VISIBLE_DEVICES=-1 python3 -m pytest test*
