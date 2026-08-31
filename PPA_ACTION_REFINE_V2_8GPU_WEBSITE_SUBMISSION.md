# Past → Plan → Action v2 action-refine 8-GPU website submission

Retrains the PPA bridge from its exact-zero fresh state under a hard trust
region, with sampled-rollout checkpoint selection. Motivation (v1 post-mortem,
R2R val-unseen 1839 episodes): the unconstrained bridge drifted to per-element
delta RMS ≈ 0.7 for a within-noise (≤4%) teacher-forced gain and closed-loop
SR collapsed 62.5% → 18.1% (OS 70.6% → 20.4%). v2 changes, all in
`configs/ppa_action_refine_v2_8gpu.yaml`:

- `model.past_plan_action.max_delta_ratio: 0.05` — per-token ‖Δ‖ ≤ 5% of the
  native ‖plan_z0‖ token norm, enforced in training **and** deployment.
- `past_plan_action_reset_bridge: true` — the Stage-2 bridge weights are not
  warm-started; the bridge restarts from the audited exact-zero state.
- `preserve_weight: 2.0`, `delta_z_weight: 10.0` with `delta_z_relative: true`
  — flat directions decay to an exactly silent bridge.
- `action_advantage_enabled: true` (reference MSE 0.125, cap 4×) — the action
  loss only pushes where frozen native System1 is measurably wrong.
- `save_best_metric: val_rollout_endpoint_error` — best checkpoint is chosen
  by real sampled rollouts (bridged vs native under shared noise, exact
  deployment post-processing), not by teacher-forced velocity MSE.

## Prerequisite 1: re-collect the R2R panoramic expert data

The original `r2r_paronamic_data` was deleted. Re-collect with the fixed
reset-driven collector (`--depth-directions front front_down`) through
`run_collect_panoramic_mxc500.sh` — deployed in `<root>/habitat/VLN-CE`,
canonical copy in `scripts/run_collect_panoramic_mxc500.sh`. It starts one
bundle Xvfb + llvmpipe display per worker (no NVIDIA EGL on this cluster),
shards episodes by stable hash with disjoint clip-id blocks so all workers
share one output root, and re-running the same submission resumes.  The
script is blank-container safe: parameters arrive as environment variables
and the vlnce python is called by absolute path (no conda activation).

Website submission — audit 20 clips first (finishes in minutes):

```bash
cd /mnt/afs/liwenhao/agent/370910109/habitat/VLN-CE

export OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_panoramic_audit_v2
export SPLIT=train
export TOTAL_CLIPS=20
export NUM_WORKERS=4
export BASE_DISPLAY=230

bash run_collect_panoramic_mxc500.sh
```

Verify every clip has `depth_front_down` in its first chunk and that
`meta.json` scene/episode ids come from the reset-driven episode (the audit
snippet in `docs/server_habitat_panoramic_recollect_plan.md` §4, pointed at
this output). Then submit the formal collection (measured ~15-20 s/clip per
worker, so 5000 clips on 8 workers is an afternoon):

```bash
cd /mnt/afs/liwenhao/agent/370910109/habitat/VLN-CE

export OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_panoramic_data_v2
export SPLIT=train
export TOTAL_CLIPS=5000
export NUM_WORKERS=8
export BASE_DISPLAY=230

bash run_collect_panoramic_mxc500.sh
```

Notes: a worker stops early once Habitat's episode iterator cycles through
its whole residue class, so per-worker counts can land slightly under target
on full-split runs; the launcher prints the final clip count. The training
dataset reproduces the deterministic scene-level MD5 train/val split from the
direct scene root (`<output>/train/<scene>/clip_*`).

## Prerequisite 2: rebuild the AMB3R endpoint cache on the NEW data

The old `amb3r_endpoint_v2_full_r2r` cache was keyed to the deleted clips and
has been removed. The config sets `require_amb3r_pose_cache: true`, which
fail-closes on any miss, so build a fresh endpoint cache into a new
directory. Submit only AFTER the collection job has fully finished: the plan
is frozen against the clips present on disk at build time. The job must be
allocated exactly 8 GPUs (the script fail-closes otherwise); it is
resume-safe — resubmitting the same command skips valid clips.
`RUNTIME_CACHE_ROOT` reuses the existing warm model/HF/triton runtime cache,
which is not keyed to the data.

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export ALLOWED_ROOT=/mnt/afs/liwenhao/agent/370910109
export DATASET_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_panoramic_data_v2/train
export CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v3_full_r2r
export RUNTIME_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/amb3r/checkpoints/runtime_cache_ppa_r2r_8gpu
export SPLITS=train,val
export MAX_CLIPS_PER_SPLIT=0
export AMB3R_GPU_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh
```

If the re-collected data or cache lands elsewhere, change only
`PPA_DATA_ROOT` / `PPA_AMB3R_CACHE_ROOT` here and in the training block below.

## Formal v2 action-refine training (8 GPU)

`--load-weights` points at the completed Stage-2 deployment/EMA best.pth; its
79 Heatmap + 11 Future tensors are loaded and its trained bridge is
intentionally ignored. Optimizer/scheduler are fresh; never pass `--resume`.

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export PPA_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_panoramic_data_v2/train
export PPA_AMB3R_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v3_full_r2r
export INTERNNAV_MODEL_PATH=/mnt/afs/liwenhao/agent/370910109/InternNav-Model

export PPA_ACTION_REFINE_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_refine_v2_8gpu
export PPA_TENSORBOARD_ROOT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_refine_v2_8gpu/tensorboard

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1

export MACA_HOME=/opt/maca-3.3.0
export MACA_PATH=/opt/maca-3.3.0
export MACA_DIR=/opt/maca-3.3.0
export LD_LIBRARY_PATH=/opt/maca-3.3.0/lib:/opt/maca-3.3.0/ompi/lib:/opt/maca-3.3.0/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}

# Reuse the Stage-2 run's runtime caches (HF/torch/triton warm state).
export RUNTIME_ROOT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache
export HF_HOME=$RUNTIME_ROOT/huggingface
export TORCH_HOME=$RUNTIME_ROOT/torch
export XDG_CACHE_HOME=$RUNTIME_ROOT/xdg
export MPLCONFIGDIR=$RUNTIME_ROOT/matplotlib
export TRITON_CACHE_DIR=$RUNTIME_ROOT/triton

/mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29684 \
  scripts/train.py \
  --config configs/ppa_action_refine_v2_8gpu.yaml \
  --load-weights /mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_v1_8gpu_stage2_retry1/stage2_joint/run_20260818_104438/checkpoints/best.pth \
  --distributed \
  --epochs 3 \
  --num-workers 2 \
  --pin-memory \
  --prefetch-factor 2
```

## What to watch during the run

- `delta_token_ratio_mean` / `delta_at_boundary_frac` (train metrics): the
  mean post-clamp ‖Δ‖/‖z0‖ and the fraction of tokens pinned at the 0.05
  boundary. Sustained boundary saturation means the optimizer still wants to
  drift — do not raise the ratio in response; that pressure is exactly what
  v1 measured as pure noise.
- `val_rollout_endpoint_error` vs `val_rollout_endpoint_error_native`: the
  bridged rollout should never be meaningfully worse than native; the gap and
  `val_rollout_action_agreement` (fraction of identical deployment action
  queues) quantify behavioral deviation per checkpoint.
- Startup log must show `PPA bridge retrains from its exact-zero fresh state`
  and `max_delta_ratio=0.05`.

## Outcome (2026-09-01)

This pipeline, together with the byte-exact native System2 evaluation fix
(commit ed46c76), closed the loop at **SR 62.81% / SPL 55.04%** on the full
1839-episode R2R val-unseen — statistically equal to the native baseline
(62.48% / 55.23%) with the bridge active on 99.6% of episodes. Full evidence
chain and artifact index: `docs/ppa_v2_sr_recovery_report.md`.

## Acceptance gate before any full evaluation

Deploy the selected `best_deployment_full.pth` with `PPA_EVAL_CONFIG` pointing
at `configs/ppa_action_refine_v2_8gpu.yaml` (the trust region must be active
at inference; the RPC server builds the model from this config). Run the fixed
200-episode closed-loop screen first; proceed to the full 1839-episode R2R
val-unseen evaluation only if screen SR ≥ native screen − 2 pts.
