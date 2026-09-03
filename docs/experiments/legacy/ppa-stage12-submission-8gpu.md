# Past → Plan → Action 8-GPU website submission

This launcher runs two fresh formal jobs in sequence while keeping the audited
effective global batch unchanged at 8:

1. `stage1_map_pretrain`: load the completed AMB3R-adapted 79-tensor History
   Heatmap Head, then train Future and approved shared map modules.
2. `stage2_joint`: load the Stage-1 EMA/deployment `best.pth` with a fresh
   optimizer and exact-zero bridge, then optimize Action + History + Future +
   preserve + delta losses.

Eight ranks use per-rank batch 1 and accumulation 1. Learning rates, loss
weights, four epochs per stage, EMA, and checkpoint-selection metrics are
unchanged from the validated four-GPU version. The launcher never uses
`--resume` across stages and does not pin a checkpoint digest.

The current random-walk AMB3R pose-domain adaptation is an upstream dependency:
do not start PPA Stage 1 until it finishes, and replace the placeholder below
with that run's final adapted `best.pth`. The launcher intentionally has no
hard-coded fallback weight.

## Optional prerequisite: full R2R endpoint-v2 cache (8 GPU)

This cache is independent of the random-walk History adaptation, but both jobs
read and write the same AFS. Run it concurrently only when that AFS contention
is acceptable. Its runtime caches use a separate directory.

```bash
export PPA_REPO_ROOT=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN
export PPA_ALLOWED_ROOT=/mnt/afs/liwenhao/agent/370910109
export PPA_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_paronamic_data/train
export PPA_AMB3R_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v2_full_r2r

cd "$PPA_REPO_ROOT"
ALLOWED_ROOT="$PPA_ALLOWED_ROOT" \
DATASET_ROOT="$PPA_DATA_ROOT" \
CACHE_ROOT="$PPA_AMB3R_CACHE_ROOT" \
RUNTIME_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/amb3r/checkpoints/runtime_cache_ppa_r2r_8gpu \
SPLITS=train,val \
MAX_CLIPS_PER_SPLIT=0 \
AMB3R_GPU_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh
```

## Formal two-stage training (8 GPU)

```bash
export PPA_REPO_ROOT=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN
export PPA_ALLOWED_ROOT=/mnt/afs/liwenhao/agent/370910109
export PPA_QWEN_PYTHON=/mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python
export INTERNNAV_MODEL_PATH=/mnt/afs/liwenhao/agent/370910109/InternNav-Model
export PPA_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_paronamic_data/train

export PPA_AMB3R_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v2_full_r2r
export PPA_PAST_INIT_CHECKPOINT=/mnt/afs/liwenhao/agent/370910109/model/REPLACE_WITH_FINAL_AMB3R_HISTORY_ADAPT_BEST/checkpoints/best.pth
export PPA_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_v1_8gpu

export PPA_GPU_DEVICES=0,1,2,3,4,5,6,7
export PPA_WAIT_FOR_CACHE=1
export PPA_CACHE_POLL_SECONDS=60
export PPA_CACHE_WAIT_TIMEOUT_SECONDS=0
export PPA_PREWARM_IMPORTS=1

cd "$PPA_REPO_ROOT"
bash scripts/run_past_plan_action_8gpu_mxc500.sh
```

The formal expert root is the direct flat scene root
`r2r_paronamic_data/train/<scene>/clip_*`; the contract checker reproduces the
dataset's deterministic scene-level MD5 train/validation split and requires
complete causal endpoint-v2 sidecar coverage with no GT pose or per-episode GT
scale fallback.

Every path is canonicalized under `PPA_ALLOWED_ROOT`. The output, expert data,
AMB3R cache, and initialization checkpoint must be non-overlapping. Runtime
caches are placed under `$PPA_OUTPUT_ROOT/_runtime_cache`. The default is two
data workers per rank (16 total), matching the total worker count of the
validated four-GPU configuration instead of doubling AFS metadata pressure.
