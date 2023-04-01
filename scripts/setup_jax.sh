#!/bin/bash
set -e 
# Setup Jax on GPU
sudo apt update -y
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo You have successfully install JAX 😁 on the GPU
