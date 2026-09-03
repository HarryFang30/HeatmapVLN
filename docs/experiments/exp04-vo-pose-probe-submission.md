# EXP-04 提交物：位姿换成 AMB3R VO 之后的捷径探针（8 卡）

台账条目见 [README.md → EXP-04](README.md#exp-04-位姿有噪声时外观是否变重要)。判据在 EXP-02 之后、
本次运行之前写死；本文件只放复现命令与已知边界。

**与 EXP-02 的唯一区别**：历史相对位姿不再来自仿真器真值，而是来自 AMB3R VO endpoint 缓存
（部署时用的那条路径）。四种输入配置（full / pose-only / vision-only / no-input）× 两个种子，
共 8 个探针，一次占满 8 卡。

## 网站提交（8 卡）

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export SHORTCUT_ARCHITECTURE=internnav_single_view
export SHORTCUT_CONFIG=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/configs/train_heatmap_internnav_single_view_8gpu.yaml
export SHORTCUT_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/data/heatmap_randomwalk_train_v1
export SHORTCUT_AMB3R_POSE_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu
export SHORTCUT_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/heatmap_shortcut_probe_vo_v1
export INTERNNAV_MODEL_PATH=/mnt/afs/liwenhao/agent/370910109/InternNav-Model

export SHORTCUT_SEEDS=42,1337
export SHORTCUT_NUM_HISTORY=8
export SHORTCUT_TRAIN_STEPS=12000
export SHORTCUT_TRAIN_SAMPLES=12000
export SHORTCUT_VAL_SAMPLES=400
export SHORTCUT_GPU_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_heatmap_shortcut_diagnostic_8gpu_mxc500.sh
```

预算与 EXP-02 相同：约 6 小时（`full` 最慢，多跑六项干预）。重提同一命令会跳过已有 `report.json`。

## 已验证的机制（2026-09-04，开发机 CPU 冒烟）

- `--amb3r-pose-cache-root` 打通到 `VLNSlidingWindowDataset`：样本的 `history_pose_provider`
  为 `amb3r_vo_cache`，相对位姿有限且与真值不同。
- 守卫：该开关只接受 `internnav_single_view`（缓存路径对单视角历史契约 fail-closed）。
- summarizer 新增 `same_history_pose_source` / `same_pose_cache_root` 两项匹配检查，
  **拒绝把真值域与 VO 域的探针混进同一张表**；EXP-02 的两份既有报告在新检查下仍然 passed
  （缺字段时默认为 `simulator_ground_truth`）。

## 必须写进结论的边界

1. **不与 EXP-02 样本匹配。** 缓存只覆盖 62/78 个场景并限制可用帧：同一 val 划分下
   带缓存 6578 个样本、真值 6584 个。因此本实验只能做**内部比较**（VO 域内 full vs pose-only
   vs vision-only vs no-input），不能把绝对数字与 EXP-02 的表对齐着读。
2. **"真值训练 / VO 评测"不另跑**，直接引用既有产物：
   `model/output_heatmap_amb3r_pose_adapt_endpoint_v2_4gpu/runs/run_20260814_234429/logs/metrics.jsonl`
   的 `pre_training_validation`（真值训练的部署头在 VO 位姿上评测：joint_pck8 **0.4984**、
   可见性 F1 0.6276），对照同期真值域预训练 run 的 joint_pck8 **0.9079**
   （`model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402`）。
   边界：两者不是同一 val 集、是部署头而非从零探针，只能当作"位姿域偏移代价很大"的量级证据；
   适配两个 epoch 后回到 0.5935（同一 VO val 集）。
