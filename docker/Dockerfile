# HeatmapVLN Dockerfile
# 基于 Qwen3-VL 视觉语言模型的视觉语言导航系统

# 使用 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/root/miniconda3/bin:$PATH

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    wget \
    curl \
    git \
    vim \
    ca-certificates \
    # Python 编译依赖
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # OpenCV 依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 其他工具
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh && \
    /root/miniconda3/bin/conda clean -ay

# 初始化 conda
RUN /root/miniconda3/bin/conda init bash

# 创建 conda 环境 (Python 3.12)
RUN /root/miniconda3/bin/conda create -n models python=3.12 -y

# 设置工作目录
WORKDIR /root/HeatmapVLN

# 复制 requirements.txt
COPY requirements.txt .

# 激活环境并安装 Python 依赖
# 注意：先安装 torch，然后安装其他依赖
RUN /bin/bash -c "source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate models && \
    pip install -U pip setuptools wheel && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements.txt"

# 可选：安装 FlashAttention 2（需要时间编译，可以注释掉以加快构建）
# RUN /bin/bash -c "source /root/miniconda3/etc/profile.d/conda.sh && \
#     conda activate models && \
#     pip install flash-attn --no-build-isolation"

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p models/qwen_3_vl \
    vln_training_outputs \
    /root/tf-logs \
    outputs_inference

# 设置环境变量以便自动激活 conda 环境
ENV CONDA_DEFAULT_ENV=models \
    CONDA_PREFIX=/root/miniconda3/envs/models

# 暴露端口
EXPOSE 6006

# 默认启动命令（进入 bash 并激活环境）
CMD ["/bin/bash", "-c", "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && exec bash"]

# 使用说明：
# 
# 构建镜像：
#   docker build -t heatmapvln:latest .
#
# 运行容器：
#   docker run --gpus all -it --rm \
#     -v /path/to/models:/root/HeatmapVLN/models \
#     -v /path/to/dataset:/root/HeatmapVLN/dataset_with_actions \
#     -p 6006:6006 \
#     heatmapvln:latest
#
# 训练模型：
#   docker run --gpus all -it --rm \
#     -v $(pwd)/models:/root/HeatmapVLN/models \
#     -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
#     -v $(pwd)/vln_training_outputs:/root/HeatmapVLN/vln_training_outputs \
#     heatmapvln:latest \
#     bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/train.py --config configs/train_config.yaml"
#
# TensorBoard:
#   docker run --gpus all -it --rm \
#     -v $(pwd)/tf-logs:/root/tf-logs \
#     -p 6006:6006 \
#     heatmapvln:latest \
#     bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && tensorboard --logdir=/root/tf-logs --host=0.0.0.0 --port=6006"
