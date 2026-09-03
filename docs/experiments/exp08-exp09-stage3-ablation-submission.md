# EXP-08 / EXP-09 提交物：阶段三消融四臂链式任务（8 卡）

台账条目见 [README.md → EXP-08](README.md#exp-08-阶段二联合训练买到了什么) 与
[EXP-09](README.md#exp-09-阶段三配方里哪一项真的在起作用)。判据在那里，开跑前已 commit；本文件只放复现命令。

四臂在一台 8 卡机器上顺序执行（A → B → C → EXP-08），每臂约 5–6 h，合计约 20–24 h。
每臂 = 认证 v2 阶段三的启动方式（同父 checkpoint 或指定父、fresh optimizer、桥从零、3 epoch、
按 rollout 终点误差选点），配置只改一处；所有臂验证用 512 对 bridged-vs-native rollout。

| 臂 | 配置 | 唯一变量 | 父 checkpoint |
|---|---|---|---|
| exp09a | `configs/ablation/exp09a_stage3_no_trust_region_8gpu.yaml` | `max_delta_ratio: null` | Stage-2 best |
| exp09b | `configs/ablation/exp09b_stage3_no_advantage_8gpu.yaml` | `action_advantage_enabled: false` | Stage-2 best |
| exp09c | `configs/ablation/exp09c_stage3_v1_penalties_8gpu.yaml` | preserve 0.5 / delta 0.01 绝对 | Stage-2 best |
| exp08 | `configs/ablation/exp08_stage3_from_stage1_heads_8gpu.yaml` | 父 checkpoint = **Stage-1** best | Stage-1 best |

## 网站提交（8 卡）

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export PPA_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/r2r_panoramic_data_v2/train
export PPA_AMB3R_CACHE_ROOT=/mnt/afs/liwenhao/agent/370910109/data/amb3r_endpoint_v3_full_r2r
export INTERNNAV_MODEL_PATH=/mnt/afs/liwenhao/agent/370910109/InternNav-Model
export PPA_ABLATION_ROOT=/mnt/afs/liwenhao/agent/370910109/model/ablation_stage3
export ABLATION_ARMS="exp09a exp09b exp09c exp08"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_stage3_ablation_chain_8gpu_mxc500.sh
```

- 每臂输出在 `$PPA_ABLATION_ROOT/<arm>/run_<stamp>/`（`logs/metrics.jsonl` 的 `epoch_summary` 里有
  `val_rollout_*`、`delta_token_ratio_mean`、`delta_at_boundary_frac`、`val_preserve_loss` 与认知指标）。
- 单臂失败会打印 `<arm>: FAILED` 并继续下一臂；结尾打印 summary。重提时把已完成的臂从
  `ABLATION_ARMS` 里去掉即可（本脚本不做 resume）。
- 启动日志必须出现 `PPA bridge retrains from its exact-zero fresh state`；exp09a 的日志里
  `max_delta_ratio` 应为 None。

## 跑完后的判读

按台账 EXP-09 / EXP-08 的三档判据读各臂 best checkpoint 所在 epoch 的 `epoch_summary`；
与 v2 参照比较时只用 512 对重验（EXP-09-R）的数字，不用 v2 run 自己的 64 对数字。
