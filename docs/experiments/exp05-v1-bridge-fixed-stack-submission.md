# EXP-05 提交物：v1 无约束桥 · 修复后的评测栈 · R2R val-unseen 全量

台账条目见 [README.md → EXP-05](README.md#exp-05-信赖域重训到底值多少-sr)。判据在那里，开跑前已 commit；本文件只放复现命令。

**这跑的是什么.** 与 EXP-01 终局评测（62.81%）**同一套修复栈**（逐字节对齐的 System2 前端、在线 AMB3R VO、协议种子 42、同一批 1839 集），唯一区别是 checkpoint 与 config 换成 v1 阶段三：

| 变量 | EXP-01 终局 | 本实验 |
|---|---|---|
| checkpoint | `output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth` | `output_past_plan_action_action_refine_v1_8gpu/run_20260818_225001/checkpoints/best_deployment_full.pth` |
| config | `configs/ppa_action_refine_v2_8gpu.yaml`（ρ=0.05 部署端截断） | `configs/ppa_action_refine_8gpu.yaml`（无 ρ，不截断） |
| 桥 out_proj 权重 RMS | 0.00059 | 0.0032 |

**必须用 `best_deployment_full.pth`，不能用 `best.pth`**：评测端预检（`rpc_model_server._preflight_*`）要求 checkpoint 自带 79 热力图 + 11 未来头 + 10 桥张量；v1 阶段三冻结了未来头，`best.pth` 里只有 89 个（未来头 0 个），会在启动时报 `PPA deployment tensor coverage mismatch`。`best_deployment_full.pth` 是 `scripts/tools/repair_ppa_action_refine_checkpoint.py` 从阶段二父 checkpoint 逐位拷回未来头后的自足文件（100 张量），桥张量与 `best.pth` 相同；2026-08 的 v1 评测（18.11%）加载的正是它。

`configs/ppa_action_refine_8gpu.yaml` 与该 run 的 `manifest/config.yaml` 在 `model.past_plan_action` / `model.heatmap` / `model.action_head` 上逐项相同（差异只在训练损失权重与路径占位符），评测端只读模型字段。

## 网站提交（8 卡）

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export PPA_EVAL_CHECKPOINT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_action_refine_v1_8gpu/run_20260818_225001/checkpoints/best_deployment_full.pth
export PPA_EVAL_CONFIG=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/configs/ppa_action_refine_8gpu.yaml
export PPA_EVAL_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/eval_ppa_refine_v1_unconstrained_nativefix_r2r_val_unseen_8gpu
export PPA_EVAL_ARM=ppa_refine_v1_unconstrained_online_amb3r
export PPA_EVAL_PROTOCOL_SEED=42
export PPA_EVAL_GPU_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh
```

- 启动脚本自带 `--resume`：任务被杀后原样重提即接着跑，不会重复 episode。
- 空白容器冷启动 16 个 RPC 服务要十几分钟才见第一行 episode 日志，不要急着杀。
- 成功标志：日志末尾 `[ppa-eval] COMPLETE result=.../merged/result.json`，且 merge 后的自检打印 `"status": "passed"`。

## 跑完后的判读

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN
R=/mnt/afs/liwenhao/agent/370910109
/mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python scripts/tools/paired_closed_loop_bootstrap.py \
  --treatment $R/model/eval_ppa_refine_v1_unconstrained_nativefix_r2r_val_unseen_8gpu/merged/progress.jsonl \
  --control $R/model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/merged/progress.jsonl \
  --dataset $R/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --geodesic-min 10 --label v1_unconstrained_vs_v2_seed42 \
  --output-json $R/model/eval_ppa_refine_v1_unconstrained_nativefix_r2r_val_unseen_8gpu/analysis/paired_vs_v2_seed42.json
```

再把 `--control` 换成 native（`eval_internnav_native_r2r_val_unseen_4gpu_rpcv2_x11bundle_v4/merged/progress.jsonl`）跑一次。判据看 SR 相对 62.81% 的配对差：< −3pt 支持、±1.5pt 内否定、中间为没测出来。
