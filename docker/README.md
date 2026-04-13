# Docker Deployment

Docker setup for the HeatmapVLN training and inference environment.

## Files

| File | Description |
|:-----|:------------|
| `Dockerfile` | Image build definition (CUDA 12.8 + Conda + PyTorch 2.7) |
| `docker-compose.yml` | Compose config with GPU reservation and volume mounts |
| `docker-run.sh` | Interactive launcher script with menu |
| `.dockerignore` | Build context exclusion rules |

## Quick Start

### Option 1: Launcher Script (recommended)

```bash
./docker/docker-run.sh
```

The script provides a menu for building, training, evaluation, inference, and TensorBoard.

### Option 2: Docker Compose

```bash
cd docker

# Build
docker compose build

# Interactive shell
docker compose run --rm heatmapvln

# TensorBoard (port 6007)
docker compose up tensorboard
```

### Option 3: Docker CLI

```bash
# Build from project root
docker build -f docker/Dockerfile -t heatmapvln:latest .

# Interactive shell with GPU and volume mounts
docker run --gpus all -it --rm \
    -v $(pwd)/models:/workspace/HeatmapVLN/models \
    -v /path/to/data:/workspace/r2r_panoramic_data \
    --shm-size 8g -p 6006:6006 \
    heatmapvln:latest
```

## Volume Mounts

| Host | Container | Purpose |
|:-----|:----------|:--------|
| `./models` | `/workspace/HeatmapVLN/models` | Pretrained weights (InternNav backbone, System 1, DAv2) |
| Data directory | `/workspace/r2r_panoramic_data` | R2R panoramic training data |
| Output directory | `/workspace/vln_training_outputs` | Checkpoints, logs, visualizations |
| TensorBoard logs | `/workspace/tf-logs` | TensorBoard event files |

Paths are configurable via environment variables `DATA_ROOT`, `OUT_DIR`, and `TB_DIR`.

## Common Operations

### Training

```bash
# Foreground
docker run --gpus all -it --rm \
    -v $(pwd)/models:/workspace/HeatmapVLN/models \
    -v /path/to/data:/workspace/r2r_panoramic_data \
    -v /path/to/output:/workspace/vln_training_outputs \
    -v /path/to/tb:/workspace/tf-logs \
    --shm-size 8g \
    heatmapvln:latest \
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && \
    python scripts/run.py train --config configs/train_config_internnav.yaml"

# Background
docker run --gpus all -d --name heatmapvln-train \
    ... \
    bash -c "... && python scripts/run.py train --config configs/train_config_internnav.yaml --auto-resume"

docker logs -f heatmapvln-train
```

### Multi-GPU

```bash
docker run --gpus all ...    # all GPUs
docker run --gpus '"device=0,1"' ...    # specific GPUs

# Inside container
torchrun --nproc_per_node=2 scripts/run.py train \
    --config configs/train_config_internnav.yaml --distributed
```

### TensorBoard

```bash
docker run -d --name heatmapvln-tb \
    -v /path/to/tb:/workspace/tf-logs \
    -p 6006:6006 \
    heatmapvln:latest \
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && \
    tensorboard --logdir=/workspace/tf-logs --host=0.0.0.0 --port=6006"
```

## Troubleshooting

| Problem | Solution |
|:--------|:---------|
| `CUDA not available` | Ensure `--gpus all` is passed and `nvidia-smi` works inside the container |
| `CUDA out of memory` | Reduce `optim.batch_size` in config or limit GPU count |
| DataLoader worker killed | Increase `--shm-size` (default 8g, try 16g) |
| `FileNotFoundError` | Verify volume mounts match config paths; use absolute host paths |
| Slow image build | Use `DOCKER_BUILDKIT=1` and `--cache-from heatmapvln:latest` |
