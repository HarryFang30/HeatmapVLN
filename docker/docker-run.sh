#!/bin/bash
# HeatmapVLN Docker 快速启动脚本

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  HeatmapVLN Docker 快速启动工具${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${YELLOW}项目根目录: $PROJECT_ROOT${NC}"
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: 未安装 Docker${NC}"
    exit 1
fi

# 检查 NVIDIA Docker 运行时
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}警告: NVIDIA Docker 运行时可能未正确配置${NC}"
fi

# 菜单选项
echo "请选择操作："
echo "1) 构建 Docker 镜像"
echo "2) 运行交互式容器"
echo "3) 后台运行训练"
echo "4) 启动 TensorBoard"
echo "5) 运行推理"
echo "6) 运行评估"
echo "7) 查看训练日志"
echo "8) 停止所有容器"
echo "9) 进入运行中的容器"
echo "0) 退出"
echo ""

read -p "请输入选项 (0-9): " choice

case $choice in
    1)
        echo -e "${GREEN}开始构建镜像...${NC}"
        docker build -f docker/Dockerfile -t heatmapvln:latest .
        echo -e "${GREEN}构建完成！${NC}"
        ;;
    2)
        echo -e "${GREEN}启动交互式容器...${NC}"
        docker run --gpus all -it --rm \
            -v "$PROJECT_ROOT/models":/root/HeatmapVLN/models \
            -v "$PROJECT_ROOT/dataset_with_actions":/root/HeatmapVLN/dataset_with_actions \
            -v "$PROJECT_ROOT/vln_training_outputs":/root/HeatmapVLN/vln_training_outputs \
            -v "$PROJECT_ROOT/tf-logs":/root/tf-logs \
            -v "$PROJECT_ROOT/outputs_inference":/root/HeatmapVLN/outputs_inference \
            -p 6006:6006 \
            --shm-size 8g \
            heatmapvln:latest
        ;;
    3)
        echo -e "${GREEN}后台启动训练任务...${NC}"
        read -p "是否自动恢复训练? (y/n): " resume
        if [ "$resume" = "y" ]; then
            RESUME_FLAG="--auto-resume"
        else
            RESUME_FLAG=""
        fi
        docker run --gpus all -d --name heatmapvln-train \
            -v "$PROJECT_ROOT/models":/root/HeatmapVLN/models \
            -v "$PROJECT_ROOT/dataset_with_actions":/root/HeatmapVLN/dataset_with_actions \
            -v "$PROJECT_ROOT/vln_training_outputs":/root/HeatmapVLN/vln_training_outputs \
            -v "$PROJECT_ROOT/tf-logs":/root/tf-logs \
            --shm-size 8g \
            heatmapvln:latest \
            bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py train --config configs/train_config.yaml $RESUME_FLAG"
        echo -e "${GREEN}训练任务已启动！${NC}"
        echo -e "查看日志: ${YELLOW}docker logs -f heatmapvln-train${NC}"
        ;;
    4)
        echo -e "${GREEN}启动 TensorBoard...${NC}"
        docker run --gpus all -d --name heatmapvln-tensorboard \
            -v "$PROJECT_ROOT/tf-logs":/root/tf-logs \
            -p 6006:6006 \
            heatmapvln:latest \
            bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && tensorboard --logdir=/root/tf-logs --host=0.0.0.0 --port=6006"
        echo -e "${GREEN}TensorBoard 已启动！${NC}"
        echo -e "访问地址: ${YELLOW}http://localhost:6006${NC}"
        ;;
    5)
        echo -e "${GREEN}运行推理...${NC}"
        read -p "请输入 checkpoint 路径: " ckpt_path
        read -p "请输入 clip 路径: " clip_path
        docker run --gpus all -it --rm \
            -v "$PROJECT_ROOT/models":/root/HeatmapVLN/models \
            -v "$PROJECT_ROOT/dataset_with_actions":/root/HeatmapVLN/dataset_with_actions \
            -v "$PROJECT_ROOT/vln_training_outputs":/root/HeatmapVLN/vln_training_outputs \
            -v "$PROJECT_ROOT/outputs_inference":/root/HeatmapVLN/outputs_inference \
            heatmapvln:latest \
            bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py inference --clip $clip_path --config configs/train_config.yaml --checkpoint $ckpt_path --output-dir ./outputs_inference"
        ;;
    6)
        echo -e "${GREEN}运行评估...${NC}"
        read -p "请输入 checkpoint 路径: " ckpt_path
        read -p "请输入 split (train/val_seen/val_unseen): " split
        docker run --gpus all -it --rm \
            -v "$PROJECT_ROOT/models":/root/HeatmapVLN/models \
            -v "$PROJECT_ROOT/dataset_with_actions":/root/HeatmapVLN/dataset_with_actions \
            -v "$PROJECT_ROOT/vln_training_outputs":/root/HeatmapVLN/vln_training_outputs \
            heatmapvln:latest \
            bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && python scripts/run.py evaluate --config configs/train_config.yaml --checkpoint $ckpt_path --split $split"
        ;;
    7)
        echo -e "${GREEN}查看训练日志...${NC}"
        docker logs -f heatmapvln-train
        ;;
    8)
        echo -e "${YELLOW}停止所有容器...${NC}"
        docker stop heatmapvln-train heatmapvln-tensorboard 2>/dev/null || true
        docker rm heatmapvln-train heatmapvln-tensorboard 2>/dev/null || true
        echo -e "${GREEN}所有容器已停止${NC}"
        ;;
    9)
        echo -e "${GREEN}进入运行中的容器...${NC}"
        echo "可用容器："
        docker ps --filter "ancestor=heatmapvln:latest" --format "{{.Names}}"
        read -p "请输入容器名称: " container_name
        docker exec -it $container_name bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && exec bash"
        ;;
    0)
        echo -e "${GREEN}退出${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac
