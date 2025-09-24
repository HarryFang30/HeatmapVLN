FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6.0 is already installed in the base image with CUDA 12.4
# Keep existing PyTorch 2.6.0 - it's the latest available for CUDA 12.4
# Upgrade pip
RUN pip install --upgrade pip

# Create a pip configuration file to permanently set the mirror
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf && \
    echo "[install]" >> /root/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> /root/.pip/pip.conf
    
# Install core ML dependencies
RUN pip install transformers==4.51.3 accelerate==1.5.2 qwen_vl_utils decord

# Install additional dependencies from your project structure
RUN pip install \
    opencv-python \
    matplotlib \
    seaborn \
    wandb \
    tensorboard \
    einops \
    timm \
    huggingface_hub \
    datasets \
    pillow \
    numpy \
    scipy \
    scikit-learn \
    tqdm \
    pyyaml \
    h5py \
    imageio \
    av \
    omegaconf

RUN pip install flash_attn==2.5.8 --no-build-isolation

# Install development tools
RUN pip install \
    jupyter \
    ipython \
    black \
    flake8 \
    pytest

# Create workspace directory
WORKDIR /workspace

# Copy project files (when building)
# COPY . /workspace/

# Set up Python path
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# Default command
CMD ["bash"]