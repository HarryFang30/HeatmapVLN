# Docker 部署指南

本指南介绍如何使用 Docker 部署和运行 HeatmapVLN 项目。

## 📋 前置要求

1. **Docker** (版本 >= 20.10)
2. **NVIDIA Docker 运行时** (nvidia-docker2)
3. **NVIDIA GPU** 及驱动 (CUDA 12.8 兼容)

### 安装 NVIDIA Docker 运行时

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 验证安装
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 快速开始

### 方式一：使用快速启动脚本（推荐）

```bash
# 从项目根目录运行
./docker/docker-run.sh

# 或进入 docker 目录运行
cd docker
./docker-run.sh
```

脚本会自动处理路径问题，无论从哪里运行都能正常工作。

脚本提供以下功能：
- 构建镜像
- 运行交互式容器
- 后台训练
- 启动 TensorBoard
- 运行推理和评估
- 查看日志
- 容器管理

### 方式二：使用 docker-compose

```bash
# 进入 docker 目录
cd docker

# 构建镜像
docker-compose build

# 启动交互式容器
docker-compose run --rm heatmapvln

# 后台运行训练
docker-compose run -d --name heatmapvln-train heatmapvln \
  bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py train --config configs/train_config.yaml"

# 启动 TensorBoard
docker-compose up tensorboard
```

### 方式三：使用原生 Docker 命令

```bash
# 1. 从项目根目录构建镜像
docker build -f docker/Dockerfile -t heatmapvln:latest .

# 2. 运行容器
docker run --gpus all -it --rm \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  -v $(pwd)/vln_training_outputs:/root/HeatmapVLN/vln_training_outputs \
  -v $(pwd)/tf-logs:/root/tf-logs \
  -p 6006:6006 \
  --shm-size 8g \
  heatmapvln:latest
```

## 📁 目录挂载说明

容器会自动挂载以下主机目录：

| 主机目录 | 容器目录 | 用途 |
|---------|---------|------|
| `./models` | `/root/HeatmapVLN/models` | 模型权重（Qwen2.5-VL / InternNav） |
| `./dataset_with_actions` | `/root/HeatmapVLN/dataset_with_actions` | 训练/验证数据集 |
| `./vln_training_outputs` | `/root/HeatmapVLN/vln_training_outputs` | 训练输出和检查点 |
| `./tf-logs` | `/root/tf-logs` | TensorBoard 日志 |
| `./outputs_inference` | `/root/HeatmapVLN/outputs_inference` | 推理输出 |

**注意**：首次运行前，请确保这些目录存在：

```bash
mkdir -p models/internnav_backbone dataset_with_actions vln_training_outputs tf-logs outputs_inference
```

## 🎯 常用操作

### 1. 训练模型

**前台训练（可看输出）**：
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  -v $(pwd)/vln_training_outputs:/root/HeatmapVLN/vln_training_outputs \
  -v $(pwd)/tf-logs:/root/tf-logs \
  --shm-size 8g \
  heatmapvln:latest \
  bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py train --config configs/train_config.yaml"
```

**后台训练**：
```bash
docker run --gpus all -d --name heatmapvln-train \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  -v $(pwd)/vln_training_outputs:/root/HeatmapVLN/vln_training_outputs \
  -v $(pwd)/tf-logs:/root/tf-logs \
  --shm-size 8g \
  heatmapvln:latest \
  bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py train --config configs/train_config.yaml"

# 查看日志
docker logs -f heatmapvln-train

# 停止训练
docker stop heatmapvln-train
```

### 2. TensorBoard 监控

```bash
# 启动 TensorBoard
docker run --gpus all -d --name heatmapvln-tensorboard \
  -v $(pwd)/tf-logs:/root/tf-logs \
  -p 6006:6006 \
  heatmapvln:latest \
  bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && tensorboard --logdir=/root/tf-logs --host=0.0.0.0 --port=6006"

# 访问 http://localhost:6006
```

### 3. 模型推理

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  -v $(pwd)/vln_training_outputs:/root/HeatmapVLN/vln_training_outputs \
  -v $(pwd)/outputs_inference:/root/HeatmapVLN/outputs_inference \
  heatmapvln:latest \
  bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py inference --clip dataset_with_actions/val_unseen/<scene_id>/clip_000000 --config configs/train_config.yaml --checkpoint vln_training_outputs/best_model.pth --output-dir ./outputs_inference"
```

### 4. 模型评估

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  -v $(pwd)/vln_training_outputs:/root/HeatmapVLN/vln_training_outputs \
  heatmapvln:latest \
  bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py evaluate --config configs/train_config.yaml --checkpoint vln_training_outputs/best_model.pth --split val_unseen"
```

### 5. 进入容器调试

```bash
# 进入正在运行的容器
docker exec -it heatmapvln-train bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && exec bash"

# 或启动新的交互式容器
docker run --gpus all -it --rm \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  --shm-size 8g \
  heatmapvln:latest
```

## 🔧 优化配置

### 1. 多 GPU 训练

```bash
# 使用所有 GPU
docker run --gpus all ...

# 使用特定 GPU
docker run --gpus '"device=0,1"' ...

# 通过环境变量控制
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,1 ...
```

### 2. 调整共享内存

如果遇到 DataLoader 相关错误，增加 `--shm-size`：

```bash
docker run --shm-size 16g ...
```

### 3. 镜像优化

**跳过 FlashAttention 编译（加快构建）**：

在 `Dockerfile` 中注释掉 FlashAttention 安装部分（已默认注释）。

**多阶段构建（减小镜像大小）**：

可以考虑将构建工具和运行环境分离（高级用法）。

## 🐛 常见问题

### Q1: CUDA 不可用

**错误信息**：
```
RuntimeError: CUDA not available
```

**解决方案**：
```bash
# 检查 NVIDIA Docker 运行时
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 确保使用 --gpus all 参数
docker run --gpus all ...
```

### Q2: 显存不足

**错误信息**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**：
1. 减小 batch size（修改 `configs/train_config.yaml`）
2. 限制使用的 GPU 数量
3. 清理 GPU 缓存

### Q3: 容器无法访问文件

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**：
1. 检查目录挂载是否正确
2. 确保主机目录存在且有读写权限
3. 使用绝对路径挂载

### Q4: 构建镜像很慢

**解决方案**：
1. 使用国内镜像源（修改 `Dockerfile`）
2. 注释掉 FlashAttention 安装
3. 使用构建缓存：`docker build --cache-from heatmapvln:latest ...`

### Q5: DataLoader 多进程错误

**错误信息**：
```
RuntimeError: DataLoader worker (pid xxx) is killed by signal
```

**解决方案**：
```bash
# 增加共享内存
docker run --shm-size 8g ...

# 或在 docker-compose.yml 中设置
shm_size: '8gb'
```

## 📊 性能优化

### 1. 使用缓存加速构建

```bash
# 首次构建
docker build -t heatmapvln:latest .

# 后续构建使用缓存
docker build --cache-from heatmapvln:latest -t heatmapvln:latest .
```

### 2. 使用 BuildKit

```bash
# 启用 BuildKit（更快的构建）
DOCKER_BUILDKIT=1 docker build -t heatmapvln:latest .
```

### 3. 镜像导出和导入

```bash
# 导出镜像（用于迁移）
docker save heatmapvln:latest | gzip > heatmapvln.tar.gz

# 导入镜像
docker load < heatmapvln.tar.gz
```

## 🔄 更新和维护

### 更新代码

```bash
# 宿主机上更新代码
git pull

# 重新构建镜像
docker build -t heatmapvln:latest .

# 或不重新构建，代码会通过挂载自动更新
```

### 清理资源

```bash
# 停止所有容器
docker stop $(docker ps -q --filter "ancestor=heatmapvln:latest")

# 删除所有容器
docker rm $(docker ps -aq --filter "ancestor=heatmapvlN:latest")

# 删除镜像
docker rmi heatmapvln:latest

# 清理未使用的资源
docker system prune -a
```

## 📝 最佳实践

1. **定期备份**：备份 `vln_training_outputs` 和 `models` 目录
2. **使用版本标签**：`docker build -t heatmapvln:v1.0 .`
3. **监控资源**：使用 `docker stats` 查看容器资源使用
4. **日志管理**：定期清理 Docker 日志
5. **安全性**：不要在镜像中包含敏感数据

## 🔗 相关资源

- [Docker 官方文档](https://docs.docker.com/)
- [NVIDIA Docker 文档](https://github.com/NVIDIA/nvidia-docker)
- [项目主 README](README.md)
