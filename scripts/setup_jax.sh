# Setup Jax on GPU
sudo apt update -y
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# nerf 
if [ -d "nerf" ]; then
	git clone https://github.com/higgsboost/nerf-jax ~/nerf
else
	ehco nerf also exists
fi

cd ~/nerf
mkdir llff_data
sudo apt-get install ffmpeg libsm6 libxext6  -y

pip install -r ~/nerf/requirements.txt

echo You have successfully install JAX 😁 on the GPU
