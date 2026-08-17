# AMB3R-VO pose provider for the frozen heatmap head

This path keeps Qwen, the single-view heatmap decoder, its Fourier trajectory
encoding, and System1 unchanged.  Only the source of historical poses changes:

```text
continuous rgb_front
  -> AMB3R-VO (DA3)
  -> OpenCV c2w trajectory
  -> right-multiply diag(1,-1,-1,1)
  -> existing compute_history_rel_poses(..., camera_forward_axis="-z")
  -> original [K,4] heatmap-head input
```

AMB3R receives every primitive-action frame. The heatmap head and Qwen retain
the evaluator's exact original K=8 prompt slots. A replan can leave duplicate
slots, including a capture of the unchanged current state; VO preserves those
slots as repeated poses (the current-state relative pose is identity). This is
necessary for a controlled pose-provider comparison: RGB, count, order, and
age stay unchanged. Feeding only the sampled sparse RGB frames to VO is invalid.

## Checkpoints

- `checkpoints/amb3r.pt` is the 4.14 GB AMB3R-base checkpoint from the Google
  Drive URL in the upstream README.  It is retained for the base backend.
- AMB3R-VO (DA3) instead loads the local Hugging Face snapshot at
  `checkpoints/DA3NESTED-GIANT-LARGE/`.  Passing `amb3r.pt` as the DA3 path is
  an error.
- The heatmap checkpoint is resolved by path at run time.  The new path does
  not require or enforce a checkpoint hash.  It still loads with
  `torch.load(weights_only=True)` and requires exact heatmap parameter
  coverage.

All caches are explicitly placed below `/mnt/afs/lixiaoou/intern/fjl/`; the
existing `qwen25` environment is used and no Conda environment is created.

## Causal online service

Run the pose model in its own process.  The service is deliberately
single-session and single-worker because AMB3R's keyframe map is mutable:

```bash
cd /mnt/afs/lixiaoou/intern/fjl/HeatmapVLN

AMB3R_GPU_DEVICE=0 \
AMB3R_RPC_PORT=50081 \
AMB3R_TRANSLATION_SCALE=1.0 \
bash scripts/run_amb3r_vo_rpc_server_mxc500.sh
```

It exposes only `reset_episode`, `ingest_frame`, and
`query_relative_poses`.  Requests contain no GT pose or episode-specific
scale.  There is no checkpoint hash pin, run lock, or file lock in this path.

The real 20-frame initialization plus one-frame incremental mapping smoke is:

```bash
/mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python \
  scripts/amb3r_vo/smoke_online_rpc_client.py \
  --repo /mnt/afs/lixiaoou/intern/fjl/HeatmapVLN \
  --rpc-root /mnt/afs/lixiaoou/intern/fjl/rpc \
  --clip /mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data/train/17DRP5sb8fy/clip_004345 \
  --server 127.0.0.1:50081 \
  --max-frames 21
```

For the candidate-support navigation client, opt in explicitly with:

```text
--history_pose_source amb3r_vo_da3
--amb3r_vo_rpc_server 127.0.0.1:50081
```

Without those flags the existing `habitat_gt_c2w` path is unchanged.  One VO
server cannot be shared by concurrent Habitat shards: an eight-shard run needs
eight independent VO processes/ports (normally one colocated with each model
shard), or a future multi-session backend.

## Smoke run

```bash
cd /mnt/afs/lixiaoou/intern/fjl/HeatmapVLN

AMB3R_GPU_DEVICE=0 \
AMB3R_MAX_FRAMES=20 \
AMB3R_MAX_HEATMAP_SAMPLES=2 \
RUN_TAG=smoke \
bash scripts/run_amb3r_vo_heatmap_eval_mxc500.sh
```

## Full 33-frame paired audit

```bash
cd /mnt/afs/lixiaoou/intern/fjl/HeatmapVLN

AMB3R_GPU_DEVICE=0 \
RUN_TAG=expert_clip_004345_full \
bash scripts/run_amb3r_vo_heatmap_eval_mxc500.sh
```

The paired evaluator uses only the final current frame by default.  Earlier
poses from a completed offline SLAM run may already have been updated by later
frames, so treating every prefix as an online sample would leak future video.
The output contains raw, unaligned pose errors; frozen-head metrics for both GT
and AMB3R pose treatments; direct heatmap agreement; and comparison PNGs.  The
reported per-clip oracle scale is diagnostic only and is never applied.
Deployment may use only native scale or one constant fitted on train scenes.

## Measured result on this integration

The real online smoke passed both state transitions:

| Query | Phase | Translation MAE | Yaw MAE |
|---|---|---:|---:|
| frame 19 (20-frame initialization) | stateful backend, revision 1 | 0.109 m | 0.641 deg |
| frame 20 (one new-frame tail) | stateful backend, revision 2 | 0.119 m | 1.578 deg |

A causal paired audit used one 20-frame expert clip from each of all seven
held-out train-split scenes (52 visible history targets).  It applied no
per-episode GT alignment or scale.  With identical RGB, prompt slots, frozen
visual features, heatmap head, and targets, changing only the pose provider
gave:

| Metric | GT pose | Native AMB3R pose |
|---|---:|---:|
| Joint PCK@4 | 0.904 | 0.615 |
| Joint PCK@8 | 0.962 | 0.827 |
| View-5 accuracy | 0.982 | 0.911 |
| Visibility F1 | 0.981 | 0.914 |

The native AMB3R arm lost 13.46 PCK@8 points and 7.14 view-5 points.  Mean
GT-versus-VO heatmap peak shift was 4.74 px; 85.71% of peaks stayed within
8 px.  Across clips, raw translation MAE averaged 0.191 m, yaw MAE averaged
3.24 degrees, and the median native scale ratio was 0.895.  The long tail is
real: one clip had a 0.742 scale ratio, while another had 12.39-degree yaw
error.  The summary is stored at
`model/eval_amb3r_vo_heatmap/heldout_7scene_causal_summary.json`.

This is an integration gate, not the final navigation benchmark: it uses one
final-current sample per scene, and all 52 visible historical targets happen
to lie in the back view.  A larger causal prefix audit is still required for
front/right/left coverage and confidence intervals.

Therefore the implementation is operational, but raw AMB3R replacement is
**not yet GT-equivalent** on the navigation domain.  A fixed train-split scale
can correct only systematic scale bias; it cannot remove clip-dependent yaw
and scale errors.  The next controlled step is to generate train-split VO
poses and adapt only `proj_traj` (then, if needed, the first trajectory fusion
block), while keeping Qwen, the heatmap decoder, and System1 unchanged.
