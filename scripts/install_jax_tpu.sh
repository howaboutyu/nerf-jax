#!/bin/bash
set -e
# tpu
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


echo You have successfully install JAX 😁 on TPU 
