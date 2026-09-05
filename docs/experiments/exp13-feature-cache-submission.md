# EXP-13 A 读出探针 网站提交物

判据见台账 [EXP-13](README.md#exp-13-把认知的作用点挪到-system2-的提示里)。
**开跑前把判据再读一遍，跑完只填结果，不改判据。**

**为什么走网页而不是开发机**：8 张卡。台账 §0 第 7 条规定开发机最多 3 张。
本任务的 8 个 worker 互相独立（不是 DDP），所以理论上可以拆开跑，但拆开跑要自己
盯八次、合并时还要保证八片来自同一份权重——一次提交更省事也更不容易出错。

**先跑第 0 步预检**（纯 CPU，几分钟，见
[exp13-decision-layer-runbook.md](exp13-decision-layer-runbook.md) §0）。
预检里 `corrected_fraction` 接近 0 的话，这一步照跑不误——13-A 不依赖重标——
但 13-B 就得先修规则。

---

## 提交物

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export EXP13_FJL_ROOT=$R
export EXP13_CACHE_ROOT=$R/model/exp13_decision_features
export EXP13_DAGGER_ROOT=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl
export EXP13_CHECKPOINT=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_exp13_feature_cache_8gpu_mxc500.sh
```

脚本自己导全 `configs/ppa_action_refine_v2_8gpu.yaml` 的 5 个 `$VAR` 占位并在开跑前
`ls` 验证（台账 §5 第 11 条：漏导会在冷加载 20 分钟后才炸）。

---

## 它做什么

| 步 | 内容 | 位置 |
|---|---|---|
| 1 | 8 个单卡 worker，各处理 `index % 8 == rank`，缓存全部 30816 个 DAgger 候选状态的特征 | GPU，约 1.5 h |
| 2 | 合并八片，校验状态数一致、`sample_key` 无重复 | CPU，秒级 |
| 3 | 拟合四个读出臂，写 `readout.json` | CPU，分钟级 |

缓存的特征（每状态）：

| 名字 | 形状 | 是什么 |
|---|---|---|
| `traj_hidden` | [4, 3584] | System2 自己的摘要。**13-A 的零假设臂用它**，是 System2 侧最强的表征 |
| `plan_z0` | [4, 768] | 冻结投影后真正交给 System1 的条件，次要臂 |
| `history_memory` | [8, 256] | `M_t`，桥读的那个瓶颈 |
| `history_rel_poses` / `history_visibility` | [8, 4] / [8, 4] | 几何对照臂（EXP-02/04 预测它可能就够了） |
| `future_visibility` | [4, 4] | 未来头首个时间 bin，顺带把 EXP-12 D2 的"常数前"读数从 4k 扩到全部 30.8k |

标签 `oracle_view` / `native_view` 从 EXP-12 的 `d1_per_state.jsonl` join，**不重算**——
D1、D2 与本实验共用同一个投影实现，两边不可能对不上。

---

## 启动后必须先确认的两件事

日志（`$EXP13_CACHE_ROOT/cache_shard0.log`）里应当出现：

```
oracle index: 30816 states (all buckets)
dataset: 30816 states from 4 shards (buckets=['dagger_hard', 'dagger_normal'], policy internnav-native-v1:8653...)
checkpoint init: {... 'loaded_heatmap_head_tensors': 79, 'loaded_future_head_tensors': 11, 'loaded_bridge_tensors': 10, ...}
```

- **`loaded_bridge_tensors` 必须是 10。** 探的是**部署时那个模型**（`Z = Z0 + bridge(M)`），
  桥归零跑出来的是另一个模型，与其它 EXP-1x 的数字不同源。
- **`skipped_unjoined` 应当是 0**（写在 `features_shard*.npz.json` 里）。
  不是 0 说明 join 掉了状态，先查 `--per-state-jsonl` 是不是同一轮采集的产物。

冷 AFS 下模型加载要几分钟才见第一行进度，**不要急着杀**（`CLAUDE.md` §4）。

---

## 产物

| 物 | 路径（`/mnt/afs/liwenhao/agent/370910109/` 下） |
|---|---|
| 逐片特征 + 元数据 | `model/exp13_decision_features/features_shard{0..7}.npz(.json)` |
| 合并特征 | `model/exp13_decision_features/features_merged.npz(.json)` |
| **读出结果（判据来源）** | `model/exp13_decision_features/readout.json` |
| 日志 | `model/exp13_decision_features/{cache_shard*,merge,readout}.log` |

合并后的 `.npz` 约 1.2 GB（30816 × 约 40 KB）。逐片文件跑完可以删，
但 `readout.json` 与两份 `.json` 元数据要留着，台账里的每个数字都要能在这里查到。
