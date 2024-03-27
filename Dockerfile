FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
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
 && apt-get update \
 && apt-get install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# Install build tools and libraries for Python
RUN apt-get update && apt-get install -y \
    build-essential checkinstall \
    libreadline-gplv2-dev libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev zlib1g-dev openssl \
    libffi-dev python3-dev python3-setuptools wget

RUN wget -O - https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz | tar -xz -C /usr/local --strip-components=1 && \
    echo "CMake has been downloaded and extracted." && \
    /usr/local/bin/cmake --version

# Install CUDA 9.0
RUN wget -O cuda.run https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run \
 && chmod +x cuda.run \
 && ./cuda.run --silent --toolkit --toolkitpath=/usr/local/cuda-9.0 \
 && rm cuda.run

# Check the installation of CUDA
RUN if [ -d "/usr/local/cuda-9.0" ]; then echo "CUDA installed"; else echo "CUDA not installed"; fi

# Set up the environment variables
ENV PATH /usr/local/cuda-9.0/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
RUN ls /usr/local/cuda-9.0 || echo "CUDA not found"

RUN ls /usr/local/cuda-9.0/bin

# Check the installation of CUDA
RUN /usr/local/cuda-9.0/bin/nvcc --version

# Copy the cuDNN files
COPY ./tools/cuda/include/* /usr/local/cuda-9.0/include/
COPY ./tools/cuda/lib64/* /usr/local/cuda-9.0/lib64/

# Set permissions
RUN chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*

# Add ubuntu-toolchain-r/test repository
RUN add-apt-repository ppa:ubuntu-toolchain-r/test

# Upgrade libstdc++6
RUN apt-get update && apt-get upgrade -y libstdc++6

# Download Python 3.7
RUN cd /opt && wget https://www.python.org/ftp/python/3.7.16/Python-3.7.16.tar.xz

# Extract the downloaded package and go to the Python source directory
RUN cd /opt && tar -xvf Python-3.7.16.tar.xz && cd Python-3.7.16

# Configure the build with an installation location
RUN cd /opt/Python-3.7.16 && ./configure --enable-optimizations --with-ensurepip=install

# Compile and install Python 3.7
RUN cd /opt/Python-3.7.16 && make -j 2 && make altinstall

# Create a symbolic link for python3
RUN ln -s /usr/local/bin/python3.7 /usr/local/bin/python3

# Create a virtual environment with Python 3.7
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Make port 80 available to the world outside this container
EXPOSE 80

# Upgrade pip
RUN /opt/venv/bin/python -m pip install --upgrade pip

# Install the Python dependencies
COPY requirements.txt .
RUN pip install torch==1.1.0 torchvision==0.3.0 && \
    pip install -r requirements.txt

# Set the working directory in the container to /app
WORKDIR /app

COPY ./lib /app/lib

# Install spconv
RUN cd lib/spconv && \
    rm -rf build && \
    python3 -m pip install wheel && \
    python3 setup.py bdist_wheel && \
    cd dist && \
    python3 -m pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl

# Install pointgroup_ops
RUN cd lib/pointgroup_ops && \
    rm -rf build && \
    python3 setup.py develop

# Install bspt_ops
RUN cd lib/bspt && \
    rm -rf build && \
    python3 setup.py develop
