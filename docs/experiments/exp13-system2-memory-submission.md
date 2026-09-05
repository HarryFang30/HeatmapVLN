# EXP-13 B System2 提示注入 + LoRA 网站提交物

判据见台账 [EXP-13](README.md#exp-13-把认知的作用点挪到-system2-的提示里)。
**开跑前把判据再读一遍，跑完只填结果，不改判据。**

⛔ **门控：只有 13-A 的判定不是"否定"才提交这一步。**
13-A 否定意味着 `M_t` 里没有 System2 之外的决策信息，那么本实验即便涨了，
涨的也不是记忆的功劳，两臂之差会趋近 0，7 个卡时白烧。

**为什么走网页**：8 卡（torchrun 起 8 个 rank，梯度由 `scripts/training/distributed.py` 按可训练模块白名单手工 all-reduce，不是 DDP 包装），两臂串行，合计约 7 小时。台账 §0 第 7 条。

---

## 提交物

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export EXP13_FJL_ROOT=$R
export EXP13_TRAIN_ROOT=$R/model/exp13_system2_memory
export EXP13_DAGGER_ROOT=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl
export EXP13_PARENT_CHECKPOINT=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export EXP13_ARMS="exp13a exp13b"
export EXP13_EPOCHS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_exp13_system2_memory_8gpu_mxc500.sh
```

脚本自己从 sealed manifest 读出四个 shard 路径与采集策略指纹并导出，
所以 config 里的 `expected_policy_fingerprint` 不会和数据漂开。

---

## 两臂差在哪里（这是本实验的全部）

| | `exp13a` | `exp13b` |
|---|---|---|
| `model.system2_memory.mode` | **`memory`** | **`constant`** |
| 记忆 token 数 / 位置 | 8 个，用户轮首 | 同左 |
| 可训练参数 | LoRA + 投影模块 | 同左（形状逐个相同） |
| token 内容依赖 `M_t` | 是 | **否** |
| 数据、标签、LoRA 容量、步数 | | 完全相同 |

`configs/ablation/exp13{a,b}_*.yaml` 的 diff 只有 `mode` 一行和 stage 名。
**控制臂不是可选的**：在 DAgger 恢复状态上微调 System2 本身就可能涨分，
没有它，任何涨幅都归因不到记忆头上——这正是 EXP-01 同时改两个变量、
事后要靠 EXP-05 才拆开的那个错误。

---

## 标签是怎么来的

`src/data/dagger_system2_sft.py`，规则只有两条：

- native 的方向与 oracle 一致 → **原样复现 native 自己的输出**（自蒸馏）。
  多数状态走这条，包括占 hard 状态 58.4% 的 `avoidable_revisit`
  （EXP-12 测到那里 native 已经有 92.1% 正确，不该动它）。
- 不一致 **且** oracle 首步是转向 → 改成 oracle 的转向箭头。

两种都是合法的 native 输出：带数字的答案按像素目标执行，纯箭头按基元动作执行。
所以训练出来的策略能直接走已认证的 RPC 路径，**评测端一行不用改**。

无法在不猜的情况下标注的行**在构造时就被丢掉并计数**，不会被默认成某个值。

---

## 起步很慢是正常的

`TrajectoryDaggerDataset` 构造时要把 10804 个 `episode.tar` 的账本过一遍，
**实测在 AFS 上约 15 分钟**（2026-09-05 冒烟测得）。训练集与验证集是同一个 sealed 池的
两个场景切片，所以 `src/data/factory.py` 把这个 reader **按进程缓存**，两边共用一次扫描；
否则每个 rank 都要扫两遍。之后才开始载 7B 模型。

**所以第一行训练日志出现得很晚（约 20 分钟）是预期行为，不要急着杀**（`CLAUDE.md` §4）。

## 启动后必须先确认的三件事

下面四句都会出现在 `run_*/logs/train.log` 里（2026-09-05 冒烟逐条核对过）：

```
System2 memory tokens: mode=memory, tokens=8 (the treatment and control arms differ only here)
System2 SFT relabelling (train): {... "corrected_fraction": ...}
✓ Loaded the deployed Past Head as a frozen input: tensors=79
🔄 Broadcasting trainable module: system2_memory
```

1. **`mode` 必须与臂一致。** 跑 `exp13b` 那一段时它必须打印 `constant`；
   打印成 `memory` 说明跑的是 treatment 的复本，**作废重来**。
2. **`corrected_fraction`** 要与 runbook §0 预检的数字一致（同规则同数据）。
3. **`tensors=79`**，少一个就说明历史头没载全，`M_t` 不是部署时那个。
4. **两个 `Broadcasting trainable module` 必须同时出现 `vlm_lora` 与 `system2_memory`。**
   本仓库多卡不是 DDP 包装，而是按白名单手工 all-reduce
   （`scripts/training/distributed.py`）；少一个就意味着那组参数**各卡各练各的**。

另外看一眼 `Trainable params`：冒烟实测 **6,749,696（0.08%）= vlm_lora 5,767,168 +
system2_memory 982,528**，两组之外有任何东西可训练，`strict_trainable_modules: true`
会直接抛错。

### 已经装了一道防止"静默失效"的兜底

记忆 token 曾经**完全没有到达语言模型**（台账 §5 第 16 条）：提示里哨兵齐了、`M_t` 也正常，
但替换用的 hook 挂错了模块，触发 0 次，把投影放大 50 倍 LM loss 一动不动。
现在 `src/models/qwen2_5_vl/integration.py` 里有一条兜底——**准备了替换却一次都没发生就抛错**，
所以这种失效不会再悄悄跑完两条臂。看到

```
sentinel embeddings were prepared but never substituted
```

说明注入路径又断了，**作废重来**，不要读那次的数。

---

## 评测（每臂一次，单卡即可）

命令见 [exp13-decision-layer-runbook.md](exp13-decision-layer-runbook.md) §2。
判据要的两个数：`recovery_turn_accuracy`、`normal_preservation`。

**读数之前先看 `by_kind.*.first_token_texts`。** 如果两臂都恒输出同一个 token，
差值为 0 是结构决定的，不是能力差异——EXP-12 的 D2 正是这么翻的车
（两个常数互比，Δ = −0.26pt 被误读成"未来头差一点"）。

---

## 产物

| 物 | 路径（`/mnt/afs/liwenhao/agent/370910109/` 下） |
|---|---|
| 两臂训练 run | `model/exp13_system2_memory/{exp13a,exp13b}/run_*/` |
| 冻结历史头载入记录 | 同上 `manifest/frozen_past_head_init.json` |
| 训练前验证（val_lm_loss 基线） | 同上 `manifest/pre_training_validation.json` |
| **决策评测（判据来源）** | `model/exp13_system2_memory/{exp13a,exp13b}/decisions.json` |

---

## 边界（跑之前就要知道）

- 位姿是 DAgger tar 里的 **Habitat 真值**，不是部署期的 AMB3R VO。
  **正面结果是上界**，13-C 之前必须给 DAgger 集补 VO 缓存并复验。
- 重标只覆盖"oracle 首步是转向"的状态；oracle 直行但方向仍不对的状态没有监督，
  所以 `recovery_turn_accuracy` 不是恢复能力的完整度量。
- 单训练种子起步。擦边结论必须补种子，别在判据上找补。
