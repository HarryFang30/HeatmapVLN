# EXP-14 停 + 掉头 同一次微调 网站提交物

判据见台账 [EXP-14](README.md#exp-14-把停的决定也放进同一次微调)。
**开跑前把判据再读一遍，跑完只填结果，不改判据。**

⛔ **门控：先看 13-A 的判定。**

- 13-A **不是否定** → 提交下面的两臂（`exp14a exp14b`）。它们**顶替** 13-B 的那次训练：
  同样的两臂、同样的卡时，数据多了停的重标。13-B 的转向判据和 EXP-14 的停判据
  在同一对 checkpoint 上读。
- 13-A **否定** → 记忆这条腿已死，只提交 **`exp14b`** 一臂。它是"没有状态、只有锚定
  微调"的方法，和 native 比；论文不得把任何涨幅记在空间理解头上。

**为什么走网页**：8 卡（手工 all-reduce，不是 DDP），两臂串行约 7 小时。台账 §0 第 7 条。

---

## 提交物

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export EXP13_FJL_ROOT=$R
export EXP13_TRAIN_ROOT=$R/model/exp14_system2_memory_stop
export EXP13_DAGGER_ROOT=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl
export EXP13_PARENT_CHECKPOINT=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export EXP13_ARMS="exp14a exp14b"
export EXP13_EPOCHS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_exp13_system2_memory_8gpu_mxc500.sh
```

启动脚本与 13-B 是同一个（变量名也沿用 `EXP13_*`），只是 `EXP13_ARMS` 与
`EXP13_TRAIN_ROOT` 不同。13-A 否定时把 `EXP13_ARMS` 改成 `"exp14b"`。

### 4 卡版本（2026-09-06 加）

同一个脚本按 `CUDA_VISIBLE_DEVICES` 的张数选配置：4 张就用 `configs/ablation/exp14{a,b}_*_4gpu.yaml`。
两份 4 卡配置与 8 卡的差别**只有两行**（`tests/test_exp14_configs.py` 钉死）：`gpu.devices` 四张、
`optim.grad_accum_steps` 2 → 4。采样器把每个优化步的 16 个样本分给 4 个 rank 而不是 8 个，
样本集合、优化步数、cosine 调度、EMA 预热都与 8 卡相同，只有 `val_lm_loss` 的分片略有不同。
代价约 **4 卡 × 7 h × 2 臂 ≈ 14 h**。

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export EXP13_FJL_ROOT=$R
export EXP13_TRAIN_ROOT=$R/model/exp14_system2_memory_stop
export EXP13_DAGGER_ROOT=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl
export EXP13_PARENT_CHECKPOINT=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export EXP13_ARMS="exp14a exp14b"
export EXP13_EPOCHS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/run_exp13_system2_memory_8gpu_mxc500.sh
```

日志第一行应是 `[exp13-train] world size: 4`，train.log 里 `world_size=4`；其余确认项同上。

---

## 两臂差在哪里

| | `exp14a` | `exp14b` |
|---|---|---|
| `model.system2_memory.mode` | **`memory`** | **`constant`** |
| 相对 `exp13a` / `exp13b` 多出的行 | `stop_supervision: true`、`stop_horizon_m: 1.0`、`stop_oversample` | 同左 |
| 其余 | 与 13-B 两臂逐字相同 | |

`tests/test_exp14_configs.py` 把这三组 diff 钉死了：exp14a−exp13a、exp14b−exp13b
各恰好是 stage 名 + 三个 `stop_*` 键；exp14a−exp14b 恰好是 `mode` 一行 + stage 名。

---

## 停的标签是怎么来的

`src/data/dagger_system2_sft.py` 多了一条规则，排在转向规则**前面**：

- oracle 的影子路线**以终点收尾**（`oracle.terminal`）且到终点的路程 `travelled_m ≤ 1.0 m`，
  而 native 还在走（发了像素目标）→ 改成 **`STOP`**。
- 其余状态按 13-B 的规则不变。

采集器**从不**把 STOP 写进 `oracle.actions`，它用 `terminal` + `travelled_m` 表达"到了"；
规则读的就是这两个字段。**采集器也从不保留 native 自己说 STOP 的状态**
（`native_kind != "trajectory"` 一律丢弃），所以"native 停早了"这一类改正在这份数据里
**不存在**，早停只能由判据里的误报率兜住。

`stop_horizon_m = 1.0` 在看到任何读数之前写死。`stop_oversample` 由预检的计数按台账里
写死的规则算出，**不是**从结果调的。

---

## 重跑（2026-09-06；第一次提交两臂都作废）

第一次 4 卡提交里 `exp14a` 死于半同步的检出、`exp14b` 训的代码不对应任何 commit（台账 EXP-14「运行记录」
与 §5 第 27 条）。**判据一字不动**，两臂在同一个 commit 上重跑一次。8 卡约 3–4 h × 2：

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export EXP13_FJL_ROOT=$R
export EXP13_TRAIN_ROOT=$R/model/exp14_system2_memory_stop
export EXP13_DAGGER_ROOT=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl
export EXP13_PARENT_CHECKPOINT=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export R2R_TRAIN_JSON=$R/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
export EXP13_ARMS="exp14a exp14b"
export EXP13_EPOCHS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_exp13_system2_memory_8gpu_mxc500.sh
```

`R2R_TRAIN_JSON` 在本实验里用不到（`cognition_prefix: false`），但启动脚本要它存在。
新的 run 目录会多出 `manifest/source_fingerprint.json`；**两臂的指纹必须逐字相同**，
日志开头也会打印 `[exp13-train] source fingerprint:` 与 `commit: <短哈希> (clean)`。
检出有未提交改动时脚本直接退出，不会再出现"半同步的树"。
旧的 `run_20260906_09*` 保留作参照，不作判据来源。

## 起步很慢是正常的

同 13-B：sealed 账本要过一遍（约 15 分钟），之后才载 7B 模型；第一行训练日志约 20 分钟后出现。

## 启动后必须先确认的四件事

`run_*/logs/train.log` 里：

```
System2 memory tokens: mode=memory, tokens=8 ...
System2 SFT relabelling (train): {... "stop_supervision": true, "kinds": {... "correct_stop": N, "correct_turn": M ...}, "stop_oversample": K ...}
✓ Loaded the deployed Past Head as a frozen input: tensors=79
🔄 Broadcasting trainable module: system2_memory
```

1. `mode` 与臂一致（`exp14b` 必须是 `constant`）。
2. **`"stop_supervision": true`**，且 `correct_stop` / `correct_turn` 的计数与预检
   `model/exp14_relabel_audit.json` 里 train 场景的数字一致。少了 `stop_supervision`
   说明跑成了 13-B 的复本。
3. `tensors=79`。
4. `vlm_lora` 与 `system2_memory` 两条 Broadcasting 都在。

看到 `sentinel embeddings were prepared but never substituted` → 注入断了，作废重来。

---

## 评测（每臂一次，单卡）

命令与 13-B 相同（[runbook §2](exp13-decision-layer-runbook.md)），config 换成 `exp14a_*` / `exp14b_*`，
输出写到 `model/exp14_system2_memory_stop/{exp14a,exp14b}/decisions.json`。

判据要的数：`stop_recall`、`stop_false_alarm`（EXP-14），以及 `recovery_turn_accuracy`、
`normal_preservation`（13-B 的判据照旧在这对 checkpoint 上读）。

**读数之前先看 `by_kind.*.first_token_texts` 和 `by_kind.*.predicted_stop`。**
一个恒答 STOP 的模型 `stop_recall` 是 1.0——它会被 `stop_false_alarm` 判死，
但先看分布能省掉一次误读。

---

## 产物

| 物 | 路径（`/mnt/afs/liwenhao/agent/370910109/` 下） |
|---|---|
| 预检（含终点路线分布，不是判据） | `model/exp14_relabel_audit.json` |
| 两臂训练 run | `model/exp14_system2_memory_stop/{exp14a,exp14b}/run_*/` |
| **决策评测（判据来源）** | `model/exp14_system2_memory_stop/{exp14a,exp14b}/decisions.json` |

---

## 边界（跑之前就要知道）

- 早停的改正样本在这份采集里**不存在**，早停只由 `stop_false_alarm`（决策级）和
  闭环的早停率约束兜住。
- `correct_stop` 的"到了"由影子 oracle 定义（Habitat 真值位姿、`goal_tolerance_m = 0.3`），
  horizon 是沿 oracle 路线的路程，不是直线距离。
- 1–3 m 之间的状态（停下也算成功）仍是 `keep_pixel`，`stop_recall` 不度量它们。
- 记忆臂沿用 13-A/B 的 Habitat 真值位姿边界；单训练种子。
