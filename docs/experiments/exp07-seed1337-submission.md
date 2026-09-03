# EXP-07 提交物：主表第二个协议种子（1337），两臂全量 1839 集

台账条目见 [README.md → EXP-07](README.md#exp-07-主表第二个种子parity--ne-是否可复现长路径增益是否存在)。判据在那里，开跑前已 commit；本文件只放复现命令。

两臂只改协议种子（42 → 1337），其余与种子 42 的两次认证评测完全一致。种子决定 NextDiT 采样噪声（`heatmapvln-nextdit-sha256-v1` 协议下按 episode/step 派生），不影响 System2 贪心解码。

## 臂 1：完整方法（修复栈，桥在线）

`scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh` 自 2026-09-03 起把种子和 arm 标签暴露为
`PPA_EVAL_PROTOCOL_SEED` / `PPA_EVAL_ARM`（默认 42 / `ppa_stage2_online_amb3r`，即 EXP-01 的用法）。
第二个种子必须用**新的输出根**：脚本按 `--resume` 续跑，同一目录混两个种子会被当成已完成的 episode 跳过。

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export PPA_EVAL_CHECKPOINT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export PPA_EVAL_CONFIG=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/configs/ppa_action_refine_v2_8gpu.yaml
export PPA_EVAL_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu_seed1337
export PPA_EVAL_PROTOCOL_SEED=1337
export PPA_EVAL_GPU_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh
```

成功标志同 EXP-05：`[ppa-eval] COMPLETE` + 自检 `"status": "passed"`。

## 臂 2：native（认证复刻栈）

待补。做法：把 `evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802` 复制为
`evaluation_plans/internnav_native_r2r_val_unseen_8gpu_seed1337`，只把 `scripts/run_8gpu_rpc_eval.sh`
第 47 行的 `PROTOCOL_SEED=42` 改为 1337、输出根改名；原 plan 目录（62.48% 的 golden 参照）一个字节不动。
准备好后把提交命令补到这里再提。

## 跑完后的判读

对每个种子分别、再对两种子合并（把两个 progress.jsonl 各自配对后拼接）跑
`scripts/tools/paired_closed_loop_bootstrap.py --geodesic-min 10`，按台账 EXP-07 的六档判据读。
