# Past → Plan → Action 4-GPU website submission

This package defines two fresh runs in one website job:

1. `stage1_map_pretrain`: exact 79-tensor AMB3R-adapted Past Head → train
   Future Head + shared Past map modules; no Action loss and a frozen bridge.
2. `stage2_joint`: Stage-1 EMA/deployment `best.pth` → fresh optimizer and
   scheduler, exact-zero fresh bridge, then Action + History + Future +
   preserve + delta losses.

The stage transition always uses `--load-weights`; it never uses `--resume`.
There is no checkpoint hash lock and no file lock.

The full expert endpoint-v2 AMB3R cache is a separate prerequisite. The
launcher may be submitted before it is complete: it waits for
`$PPA_AMB3R_CACHE_ROOT/_control/cache.ready.json`, then validates causal/no-GT
semantics and exact clip coverage across the dataset's deterministic MD5 scene
`train` + `val` auto-split. Set `PPA_WAIT_FOR_CACHE=0` to fail immediately
instead of waiting.

`PPA_DATA_ROOT` is deliberately the physical R2R **scene root**
`r2r_paronamic_data/train`, whose direct children are `<scene>/clip_*`. Do not
pass its parent `r2r_paronamic_data`: the training dataset would interpret the
physical `train` directory as a logical split and would not reproduce the
intended scene-level train/validation partition.

## Website cache-phase command (4 GPU, full R2R)

Submit this once to build and atomically publish the strict endpoint-v2 cache.
`MAX_CLIPS_PER_SPLIT=0` means full coverage, not a smoke subset. The existing
four-worker wrapper delegates to the audited endpoint-v2 cache pipeline; it
does not train the PPA heads.

```bash
export PPA_REPO_ROOT=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN
export PPA_ALLOWED_ROOT=/mnt/afs/liwenhao/agent/370910109
export PPA_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_paronamic_data/train
export PPA_AMB3R_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v2_full_r2r

cd "$PPA_REPO_ROOT"
ALLOWED_ROOT="$PPA_ALLOWED_ROOT" \
DATASET_ROOT="$PPA_DATA_ROOT" \
CACHE_ROOT="$PPA_AMB3R_CACHE_ROOT" \
SPLITS=train,val \
MAX_CLIPS_PER_SPLIT=0 \
AMB3R_GPU_DEVICES=0,1,2,3 \
bash scripts/run_amb3r_pose_training_cache_4gpu_mxc500.sh
```

## Website training command (4 GPU, Stage 1 → Stage 2)

```bash
export PPA_REPO_ROOT=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN
export PPA_ALLOWED_ROOT=/mnt/afs/liwenhao/agent/370910109
export PPA_QWEN_PYTHON=/mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python
export INTERNNAV_MODEL_PATH=/mnt/afs/liwenhao/agent/370910109/InternNav-Model
export PPA_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_paronamic_data/train

export PPA_AMB3R_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v2_full_r2r
export PPA_PAST_INIT_CHECKPOINT=/mnt/afs/liwenhao/agent/370910109/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402/checkpoints/best.pth
export PPA_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_v1_4gpu

export PPA_GPU_DEVICES=0,1,2,3
export PPA_WAIT_FOR_CACHE=1
export PPA_CACHE_POLL_SECONDS=60
export PPA_CACHE_WAIT_TIMEOUT_SECONDS=0
export PPA_PREWARM_IMPORTS=1

cd "$PPA_REPO_ROOT"
bash scripts/run_past_plan_action_4gpu_mxc500.sh
```

Every formal path is canonicalized with `realpath -m` semantics before the
first Python process. Existing symlinks are resolved and the result must be a
strict descendant of `PPA_ALLOWED_ROOT`. The output, expert data, AMB3R cache,
and initialization checkpoint scopes must be pairwise non-overlapping. HF,
Hugging Face, Torch, XDG, Matplotlib, and Triton caches are forcibly placed in
`$PPA_OUTPUT_ROOT/_runtime_cache` before Python starts.

Defaults are 4 epochs per stage, per-rank batch 1, gradient accumulation 2,
effective global batch 8, and four data workers per rank. They can be changed
with `PPA_STAGE1_EPOCHS`, `PPA_STAGE2_EPOCHS`, and `PPA_NUM_WORKERS`.
When `PPA_NUM_WORKERS=0`, the launcher omits `--prefetch-factor`; positive
worker counts pass the configured `PPA_PREFETCH_FACTOR`.

`PPA_PREWARM_IMPORTS=1` (default) imports `scripts.train` exactly once after
the full cache contract passes and before torchrun. This has no training or
checkpoint side effect; it warms AFS-backed module pages so four ranks do not
cold-scan them concurrently. Set it to `0` to skip the prewarm.

Within-stage exact resume is intentionally not part of this fresh two-stage
launcher. A failed output is retained for diagnosis, and rerunning with the
same output root fails closed instead of silently reusing it.
