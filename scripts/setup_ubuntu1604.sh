#!/bin/bash

# Check if CUDA and cuDNN are installed and working
if /usr/local/cuda-10.0/bin/nvcc --version && echo "/usr/local/cuda-10.0" | ldconfig -p | grep libcudnn; then
    echo "CUDA and cuDNN are installed and working. Skipping download and installation."
else
    # Update and install necessary packages
    sudo apt-get update && sudo apt-get install -y \
        wget \
        curl \
        ca-certificates \
        sudo \
        git \
        bzip2 \
        libx11-6 \
        software-properties-common \
        g++ \
        libboost-all-dev \
        libsparsehash-dev \
        freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

    # Install build tools and libraries for Python
    sudo apt-get update && sudo apt-get install -y \
        build-essential checkinstall \
        libreadline-gplv2-dev libncursesw5-dev libssl-dev \
        libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev zlib1g-dev openssl \
        libffi-dev python3-dev python3-setuptools wget

    # Download and install CMake
    wget -O - https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz | sudo tar -xz -C /usr/local --strip-components=1
    /usr/local/bin/cmake --version

    # Download and install CUDA 9.0
    # wget -O cuda.run https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_10.1.105_418.39_linux-run
    # rm cuda.run

    # Install CUDA 10.0
    # Make the .run file executable
    chmod +x ./tools/cuda_10.0.130_410.48_linux.run
    sudo ./tools/cuda_10.0.130_410.48_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.0

    # Set up the environment variables
    export PATH=/usr/local/cuda-10.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

    # Extract the cuDNN files to a temporary location
    sudo tar -xzvf ./tools/cudnn-10.0-linux-x64-v7.6.5.32.tgz -C /tmp

    # Move the cuDNN files to the correct locations
    sudo cp -P /tmp/cuda/include/* /usr/local/cuda-10.0/include
    sudo cp -P /tmp/cuda/lib64/* /usr/local/cuda-10.0/lib64

    # Set permissions
    sudo chmod a+r /usr/local/cuda-10.0/include/*
    sudo chmod a+r /usr/local/cuda-10.0/lib64/*

    # Remove the extracted cuDNN files from the temporary location
    sudo rm -rf /tmp/cuda

    # Add ubuntu-toolchain-r/test repository
    yes | sudo add-apt-repository ppa:ubuntu-toolchain-r/test

    # Upgrade libstdc++6
    sudo apt-get update && sudo apt-get upgrade -y libstdc++6
fi