# HeatmapVLN Dockerfile
# Qwen2.5-VL + HeatmapVLN + NextDiT System 1

FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/root/miniconda3/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git vim ca-certificates \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libncursesw5-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh && \
    /root/miniconda3/bin/conda clean -ay

RUN /root/miniconda3/bin/conda init bash && \
    /root/miniconda3/bin/conda create -n models python=3.11 -y

WORKDIR /workspace/HeatmapVLN

COPY requirements.txt .

RUN /bin/bash -c "source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate models && \
    pip install -U pip setuptools wheel && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements.txt"

# Optional: Flash Attention 2 (shared env defaults to SDPA)
# RUN /bin/bash -c "source /root/miniconda3/etc/profile.d/conda.sh && \
#     conda activate models && \
#     pip install flash-attn --no-build-isolation"

COPY . .

RUN mkdir -p models/internnav_backbone /workspace/tf-logs

ENV CONDA_DEFAULT_ENV=models \
    CONDA_PREFIX=/root/miniconda3/envs/models

EXPOSE 6006

CMD ["/bin/bash", "-c", "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && exec bash"]
