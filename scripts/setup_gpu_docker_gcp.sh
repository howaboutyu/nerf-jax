# Install nvidia drivers and cuda for GCP GPU instances
# https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver

# Cuda stuff
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py


# docker stuff

curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# nerf stuff
sudo rm -rv ~/nerf
git clone https://github.com/higgsboost/nerf-jax ~/nerf
cd ~/nerf
sudo docker pull bmild/tf_colmap
git clone https://github.com/Fyusion/LLFF

