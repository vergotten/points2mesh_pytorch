#!/bin/bash

# Store the path to the project directory in a variable
PROJECT_DIR=$(pwd)

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed. Skipping download and installation."
else
    # Download Anaconda (Python 3.7 version)
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

    # Install Anaconda and automatically answer 'yes' to all prompts
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p $HOME/anaconda3

    # Delete the Anaconda installer file
    rm Anaconda3-2023.09-0-Linux-x86_64.sh

    # Add conda to PATH
    export PATH="$HOME/anaconda3/bin:$PATH"

    # Initialize conda for shell interaction
    yes | conda init

    # Source .bashrc to update the PATH in the current shell session
    source ~/.bashrc
fi

# Create the 'myenv' conda environment with Python 3.7
conda create -n myenv python=3.7 -y

# Activate the 'myenv' conda environment
source ~/anaconda3/bin/activate myenv

# Add CUDA to PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH

# Set CUDACXX environment variable
export CUDACXX=/usr/local/cuda-10.0/bin/nvcc

# Install PyTorch, torchvision, and cudatoolkit
yes | conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Install requirements from the project directory
yes | pip install -r "$PROJECT_DIR/requirements.txt"

# Install spconv
cd $PROJECT_DIR/lib/spconv && \
rm -rf build && \
pip install wheel && \
python setup.py bdist_wheel && \
cd dist && \
pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl

# Install pointgroup_ops
cd $PROJECT_DIR/lib/pointgroup_ops && \
rm -rf build && \
python setup.py develop

# Install bspt_ops
cd $PROJECT_DIR/lib/bspt && \
rm -rf build && \
python setup.py develop

# Check if the libraries are installed and working
python -c "
try:
    import torch
    import torchvision
    import spconv
    from lib.pointgroup_ops.functions import pointgroup_ops
    import bspt
    print('All libraries are installed and working.')
except ImportError as e:
    print(f'Error: {e}')
"
