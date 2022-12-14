

sudo apt update -y
# tpu
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# nerf 
rm -rv ~/nerf
git clone https://github.com/higgsboost/nerf-jax ~/nerf
cd ~/nerf
mkdir llff_data
sudo apt-get install ffmpeg libsm6 libxext6  -y

pip install -r ~/nerf/requirements.txt

