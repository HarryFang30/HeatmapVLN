# EXP-12 复现命令（三个门控诊断）

判据见台账 [README.md](README.md) 的 EXP-12 条目。**开跑前把判据再读一遍，跑完只填结果，不改判据。**

三项都在**开发机**上直跑，不走网站提交（D1/D3 是纯 CPU，D2 只要 4 张卡、分钟量级）。
GPU 占用顺序 **7 → 6 → 5 → 4，不占 0 卡**。

共用变量：

```bash
R=/mnt/afs/liwenhao/agent/370910109
REPO=$R/HeatmapVLN
PY=$R/envs/qwen25/bin/python
DAGGER=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
OUT=$R/model/exp12_recovery_gate
```

远程命令一律走 login shell（`CLAUDE.md` §3.1），否则 `MACA_PATH` 为空、triton metax 后端直接抛错。

---

## D1 + D3a — 恢复状态几何与自然发生率（纯 CPU）

一次遍历 4 个 shard 的 10804 个 episode tar，同时产出 D1 的方向统计与 D3a 的标签发生率。

```bash
ssh finn_cci_c500 'bash -lc "
R=/mnt/afs/liwenhao/agent/370910109
\$R/envs/qwen25/bin/python \$R/HeatmapVLN/scripts/tools/summarize_recovery_state_geometry.py \
  --collection-root \$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17 \
  --output-json \$R/model/exp12_recovery_gate/d1_d3a_recovery_geometry.json
"'
```

要点：

- 样本记录里已存 `native_future_poses`、`current_camera_pose`、`candidate_signals`、`failure_tags`，
  **不需要重放动作**，也不需要 GPU 或模型。
- 投影用生产参数：HFOV 90°、384×384、四视角 `(front, right, back, left)`、`camera_forward_axis=-z`
  （取自采集 manifest 的 `contract.target`）。
- oracle 未来位姿从 `arrays/oracle_future_poses.npy` 按样本的 `future_pose_start:future_pose_end` 切片。
- D3a 的 episode 失败判定：末帧位置到 `episode.json` 的 `goal_position` 距离 > 3 m。

## D3b — val_unseen 徘徊型失败占比（纯 CPU）

评测 worker **没有留逐步位置轨迹**，所以这里只能做超额步数代理（判据里已声明它是上界）。

```bash
ssh finn_cci_c500 'bash -lc "
R=/mnt/afs/liwenhao/agent/370910109
\$R/envs/qwen25/bin/python \$R/HeatmapVLN/scripts/tools/summarize_wandering_failures.py \
  --progress \$R/model/eval_internnav_native_r2r_val_unseen_4gpu_rpcv2_x11bundle_v4/merged/progress.jsonl \
  --progress \$R/model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/merged/progress.jsonl \
  --progress \$R/model/eval_internnav_native_r2r_val_unseen_8gpu_rpcv2_x11bundle_v4_seed1337/merged/progress.jsonl \
  --progress \$R/model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu_seed1337/merged/progress.jsonl \
  --dataset \$R/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --step-size 0.25 --excess-ratio 3.0 \
  --output-json \$R/model/exp12_recovery_gate/d3b_wandering_failures.json
"'
```

## D2 — 未来头零样本方向准确率（GPU 7 → 6 → 5 → 4）

四片并行，每片一张卡，**从 7 号卡开始往下排**。分片是 `dataset index % shard_count`，与 GPU 编号无关，
所以换卡不改分片结果。（原文写成"按 episode key 哈希"，是笔误，2026-09-05 随结果一起更正。）
`--max-states` 是**四片合计**预算，每片实际跑 `max-states // shard-count`。

**`configs/ppa_action_refine_v2_8gpu.yaml` 里有 5 个 `$VAR` 占位，不导就会在加载 processor 时
抛 `HFValidationError: ... '$INTERNNAV_MODEL_PATH'`——而且是在冷 AFS 载完模型之后才抛，白等 20 分钟。**
（第一次四片启动就是这么挂的，2026-09-05。）

```bash
ssh finn_cci_c500 'bash -l -s' <<'SH'
R=/mnt/afs/liwenhao/agent/370910109
export PPA_DATA_ROOT=$R/r2r_panoramic_data_v2/train
export PPA_AMB3R_CACHE_ROOT=$R/data/amb3r_endpoint_v3_full_r2r
export INTERNNAV_MODEL_PATH=$R/InternNav-Model
export PPA_ACTION_REFINE_OUTPUT_ROOT=$R/model/exp12_recovery_gate/_cfg_unused
export PPA_TENSORBOARD_ROOT=$R/model/exp12_recovery_gate/_cfg_unused/tensorboard
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
RUNTIME=$R/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache
export HF_HOME=$RUNTIME/huggingface TORCH_HOME=$RUNTIME/torch XDG_CACHE_HOME=$RUNTIME/xdg
export MPLCONFIGDIR=$RUNTIME/matplotlib TRITON_CACHE_DIR=$RUNTIME/triton
cd $R/HeatmapVLN
for i in 0 1 2 3; do
  gpu=$((7 - i))
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup $R/envs/qwen25/bin/python scripts/tools/probe_future_head_recovery.py \
    --collection-root $R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17 \
    --checkpoint $R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth \
    --config configs/ppa_action_refine_v2_8gpu.yaml \
    --per-state-jsonl $R/model/exp12_recovery_gate/d1_per_state.jsonl \
    --bucket dagger_hard --max-states 4000 \
    --shard-index $i --shard-count 4 \
    --output-json $R/model/exp12_recovery_gate/d2_future_head_shard${i}.json \
    > $R/model/exp12_recovery_gate/d2_shard${i}.log 2>&1 < /dev/null &
done
SH
```

合并四片并按判据读数：

```bash
ssh finn_cci_c500 'bash -lc "
R=/mnt/afs/liwenhao/agent/370910109
\$R/envs/qwen25/bin/python \$R/HeatmapVLN/scripts/tools/probe_future_head_recovery.py \
  --merge \$R/model/exp12_recovery_gate/d2_future_head_shard0.json \
  --merge \$R/model/exp12_recovery_gate/d2_future_head_shard1.json \
  --merge \$R/model/exp12_recovery_gate/d2_future_head_shard2.json \
  --merge \$R/model/exp12_recovery_gate/d2_future_head_shard3.json \
  --output-json \$R/model/exp12_recovery_gate/d2_future_head_merged.json
"'
```

要点：

- 用 `setsid nohup` 起，**掉线不杀进程**（这条 ssh 链路经中转）；冷 AFS 下模型加载要几分钟才见第一行进度。
- `--per-state-jsonl` 是必填的：oracle 方向必须从 D1 的产物读，不能在 D2 里重算。
- 位姿用 DAgger tar 里的 **Habitat 真值位姿**（判据边界③已声明这让 D2 的正面结果成为上界）。
- oracle 方向与 D1 用**同一个投影函数**，两边的 `top1` 定义必须逐字一致，否则 D2 的对照失效。
- 四视角都不可见的状态在两侧**同时**剔除，剔除数写进产物。

---

## 判据读数顺序

1. 先读 D1。若 `oracle_outside_front_frac < 0.15` 或 `frac(angle>45°) < 0.20` → **停**，D2/D3 不必再看。
2. 再读 D3。若 `revisit_state_frac < 0.02` 或 `wandering_fail_frac < 0.05` → **停**。
3. 最后读 D2，按台账"总门"表决定进实现时是否需要先加 DAgger 微调臂。
