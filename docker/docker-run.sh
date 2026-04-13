#!/bin/bash
# HeatmapVLN Docker launcher

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

IMAGE="heatmapvln:latest"
CONFIG="configs/train_config_internnav.yaml"
DATA_ROOT="${DATA_ROOT:-/workspace/r2r_panoramic_data}"
OUT_DIR="${OUT_DIR:-/workspace/vln_training_outputs}"
TB_DIR="${TB_DIR:-/workspace/tf-logs}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

COMMON_VOLUMES=(
    -v "$PROJECT_ROOT/models":/workspace/HeatmapVLN/models
    -v "$DATA_ROOT":/workspace/r2r_panoramic_data
    -v "$OUT_DIR":/workspace/vln_training_outputs
    -v "$TB_DIR":/workspace/tf-logs
)

CONDA_ACTIVATE="source /root/miniconda3/etc/profile.d/conda.sh && conda activate models"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  HeatmapVLN Docker Launcher${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${YELLOW}Project: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}Config:  $CONFIG${NC}"
echo ""

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

echo "Select an action:"
echo "  1) Build image"
echo "  2) Interactive shell"
echo "  3) Train (background)"
echo "  4) TensorBoard"
echo "  5) Inference"
echo "  6) Evaluate"
echo "  7) View training logs"
echo "  8) Stop all containers"
echo "  9) Attach to running container"
echo "  0) Exit"
echo ""

read -p "Choice (0-9): " choice

case $choice in
    1)
        echo -e "${GREEN}Building image...${NC}"
        docker build -f docker/Dockerfile -t "$IMAGE" .
        echo -e "${GREEN}Done.${NC}"
        ;;
    2)
        docker run --gpus all -it --rm \
            "${COMMON_VOLUMES[@]}" \
            -p 6006:6006 --shm-size 8g \
            "$IMAGE"
        ;;
    3)
        read -p "Auto-resume? (y/n): " resume
        RESUME_FLAG=""
        [ "$resume" = "y" ] && RESUME_FLAG="--auto-resume"
        docker run --gpus all -d --name heatmapvln-train \
            "${COMMON_VOLUMES[@]}" \
            --shm-size 8g \
            "$IMAGE" \
            bash -c "$CONDA_ACTIVATE && python scripts/run.py train --config $CONFIG $RESUME_FLAG"
        echo -e "${GREEN}Training started.${NC}"
        echo -e "Logs: ${YELLOW}docker logs -f heatmapvln-train${NC}"
        ;;
    4)
        docker run -d --name heatmapvln-tensorboard \
            -v "$TB_DIR":/workspace/tf-logs \
            -p 6006:6006 \
            "$IMAGE" \
            bash -c "$CONDA_ACTIVATE && tensorboard --logdir=/workspace/tf-logs --host=0.0.0.0 --port=6006"
        echo -e "${GREEN}TensorBoard: ${YELLOW}http://localhost:6006${NC}"
        ;;
    5)
        read -p "Checkpoint path: " ckpt_path
        read -p "Clip directory: " clip_path
        docker run --gpus all -it --rm \
            "${COMMON_VOLUMES[@]}" \
            "$IMAGE" \
            bash -c "$CONDA_ACTIVATE && python scripts/run.py inference --clip $clip_path --config $CONFIG --checkpoint $ckpt_path --output-dir ./outputs"
        ;;
    6)
        read -p "Checkpoint path: " ckpt_path
        read -p "Split (train/val_seen/val_unseen): " split
        docker run --gpus all -it --rm \
            "${COMMON_VOLUMES[@]}" \
            "$IMAGE" \
            bash -c "$CONDA_ACTIVATE && python scripts/run.py evaluate --config $CONFIG --checkpoint $ckpt_path --split $split"
        ;;
    7)
        docker logs -f heatmapvln-train
        ;;
    8)
        docker stop heatmapvln-train heatmapvln-tensorboard 2>/dev/null || true
        docker rm heatmapvln-train heatmapvln-tensorboard 2>/dev/null || true
        echo -e "${GREEN}All containers stopped.${NC}"
        ;;
    9)
        docker ps --filter "ancestor=$IMAGE" --format "{{.Names}}"
        read -p "Container name: " container_name
        docker exec -it "$container_name" bash -c "$CONDA_ACTIVATE && exec bash"
        ;;
    0) exit 0 ;;
    *) echo -e "${RED}Invalid option${NC}"; exit 1 ;;
esac
