# Docker 部署文件

本目录包含 HeatmapVLN 项目的 Docker 部署相关文件。

## 📁 文件说明

- **Dockerfile** - Docker 镜像构建文件
- **.dockerignore** - 构建时忽略的文件
- **docker-compose.yml** - Docker Compose 配置文件
- **docker-run.sh** - 快速启动脚本（推荐使用）
- **DOCKER.md** - 详细部署文档

## 🚀 快速开始

### 使用快速启动脚本（最简单）

```bash
# 从项目根目录或 docker 目录运行
./docker/docker-run.sh

# 或进入 docker 目录
cd docker
./docker-run.sh
```

脚本会自动处理路径问题，无论你从哪里运行都能正常工作。

### 使用 docker-compose

```bash
# 从 docker 目录运行
cd docker

# 构建镜像
docker-compose build

# 启动容器
docker-compose run --rm heatmapvln
```

### 使用原生 Docker 命令

```bash
# 从项目根目录
docker build -f docker/Dockerfile -t heatmapvln:latest .

# 运行容器
docker run --gpus all -it --rm \
  -v $(pwd)/models:/root/HeatmapVLN/models \
  -v $(pwd)/dataset_with_actions:/root/HeatmapVLN/dataset_with_actions \
  -p 6006:6006 \
  heatmapvln:latest
```

## 📖 完整文档

详细使用说明、常见问题和最佳实践，请查看 [DOCKER.md](DOCKER.md)
