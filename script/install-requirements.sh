#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install -y libgeos-dev
sudo pip3 install --upgrade keras
sudo pip3 install shapely sympy
# Install magenta requirement cuda 8.0 v6 for tf 1.2
# From https://gitlab.com/nvidia/cuda/blob/c5e8c8d7a9fd444c4e45573f36cbeb8f4e10f71c/8.0/runtime/cudnn6/Dockerfile
# And https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
tar xvzf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig