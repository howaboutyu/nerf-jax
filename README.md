# nerf-jax
Try to do nerf in jax

# start docker with GPU
```bash
sudo docker run --gpus all -p 8888:8888 -v/tmp:/tmp -v`pwd`:/nerf -it jax-gpu:latest bash
```

## TPU


Create a `tpu` vm.
```bash
gcloud compute tpus tpu-vm create nerf \
	--zone europe-west4-a \
	--accelerator-type v3-8 \
	--version tpu-vm-base
```




