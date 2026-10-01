# EXP-21 参数敏感性：复现命令（RTX 4090，容器 `fjl-habitat`）

判据与设置见台账 [EXP-21](README.md#exp-21-参数敏感性历史帧数-k快系统采样条数-s去噪步数-m)。容器路径，宿主机 `/home/fangjialei` = 容器 `/workspace`。

## 子集（2026-09-30 已生成）

```bash
docker exec fjl-habitat bash -lc "cd /workspace/exp20/src_3ffd382 && /opt/conda/bin/python scripts/exp21/make_subset.py \
  --cohorts /workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/cohorts \
  --out /workspace/exp21/subset500 --n 500"
```

- 500 集，按测地距离四分位每层 125 集。分界是 3.85 / 6.93 / 8.43 / 10.43 / 21.04 m。
- 各分片的集数：71 / 54 / 65 / 49 / 56 / 59 / 74 / 72。
- `manifest.json` 记了规则、分界，以及输入和输出的 sha256。

## 一个点

在 [exp20-ablation-runbook.md](exp20-ablation-runbook.md) §1 的命令基础上：
- 不加臂变量（模型是 A0）。
- 种子 42。
- 加 `PPA_EVAL_EPISODE_LISTS_DIR=/workspace/exp21/subset500`，再加下面这个点自己的开关：

| 点 | 开关 |
|---|---|
| K = 5 / 6 / 7 | `PPA_EVAL_NUM_HISTORY=5`（6、7 同理） |
| S = 1 / 8 / 64 | `PPA_EVAL_NUM_SAMPLE_TRAJS=1`（8、64 同理） |
| M = 2 / 5 / 20 | `PPA_EVAL_NUM_INFERENCE_STEPS=2`（5、20 同理） |

- 代码至少要用 `src_3ffd382`。
- 输出目录：`/workspace/eval_runs/exp21_<K5|S1|M2…>_seed42_4090`。
- 启动脚本会检查服务端日志里有没有 `Sensitivity override (EXP-21)` 这一行，没有就退出。
- 用子集时不做全量合并，结尾只汇总各分片。

## 分析

- 默认点：从 A0 种子 42 全量（`/workspace/eval_runs/exp20_a0_seed42_4090/merged/progress.jsonl`）里取这 500 集。
- 每个点与默认点按集配对，用 `scripts/tools/paired_closed_loop_bootstrap.py` 做 2000 次重采样。
