# EXP-13 复现命令（决策层三段式）

判据见台账 [EXP-13](README.md#exp-13-把认知的作用点挪到-system2-的提示里)。
**开跑前把判据再读一遍，跑完只填结果，不改判据。**

三段是**递进**的，总门表在台账里。不要因为 13-A "看着还行"就并行起 13-B ——
13-A 便宜（8 卡 1.5 h），13-B 贵（8 卡 7 h），先后跑一次比并行跑两次省。

共用变量：

```bash
R=/mnt/afs/liwenhao/agent/370910109
REPO=$R/HeatmapVLN
PY=$R/envs/qwen25/bin/python
DAGGER=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
ORACLE=$R/model/exp12_recovery_gate/d1_per_state.jsonl
V2=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
```

`$V2` 的 sha256 是 `0b5a06444736ae0bbda4765d2c871d8d875989af7149c4a73470bf25288c6d69`
（2026-09-05 核对）。三段都只从它取**冻结的 79 个历史头张量**，别的一概不用。

---

## 0. 开跑前的纯 CPU 预检（几分钟，先跑这个）

它回答"重标到底改了多少状态、改在哪里"，**在烧任何卡时之前**。
不读一张 JPEG，只读 sealed `samples.jsonl`。

```bash
ssh finn_cci_c500 'bash -lc "
R=/mnt/afs/liwenhao/agent/370910109
cd \$R/HeatmapVLN && \$R/envs/qwen25/bin/python scripts/tools/audit_dagger_system2_targets.py \
  --collection-root \$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17 \
  --oracle-views \$R/model/exp12_recovery_gate/d1_per_state.jsonl \
  --output-json \$R/model/exp13_relabel_audit.json
"'
```

盯三件事：

1. `corrected_fraction` —— 被改写的状态占比。**如果它接近 0，13-B 没有可学的东西**，
   不要提交训练任务，先回来看重标规则。
2. `recovery_slice.corrected_fraction` —— `wrong_branch ∪ off_route` 上的改写比例。
   EXP-12 说这批状态 native 只有 8.3%/29.3% 猜对，所以这里**应该**远高于全局。
   如果不是，说明重标没打在靶子上。
3. `examples` 里的几条 —— **逐条肉眼看方向**。`oracle_actions_head` 以 2 开头就该出 `←`，
   以 3 开头就该出 `→`。方向搞反在聚合数字上完全看不出来，在闭环里是灾难。

这份产物**不是判据**，是预检；它不进 13-A/13-B 的任何结论。

---

## 1. 13-A 读出探针（8 卡 × 约 1.5 h，网页提交）

提交物：[exp13-feature-cache-submission.md](exp13-feature-cache-submission.md)。

一条 `bash` 跑完全部三步：8 个单卡 worker 缓存 → CPU 合并 → CPU 拟合读出头。
**这不是分布式任务**，是 8 个各自独立的进程，第 i 个只处理 `index % 8 == i`。
所以某张卡挂了只损失八分之一，单独补跑那一片即可：

```bash
CUDA_VISIBLE_DEVICES=3 $PY scripts/tools/cache_recovery_decision_features.py \
  --config configs/ppa_action_refine_v2_8gpu.yaml \
  --collection-root $DAGGER --checkpoint $V2 --per-state-jsonl $ORACLE \
  --shard-index 3 --shard-count 8 \
  --output-npz $R/model/exp13_decision_features/features_shard3.npz
```

### 读结果

```bash
$PY -c "
import json; d=json.load(open('$R/model/exp13_decision_features/readout.json'))
print(json.dumps(d['memory_minus_system2_pt'], indent=1))
for arm, v in d['arms'].items():
    print(arm, v['val']['hard_macro_accuracy'], v['val']['recovery_nonfront_recall'], v['val']['normal_false_alarm'])
print('baselines', json.dumps(d['baselines']['constant_front'], indent=1))
"
```

判据要的三个数：`memory_minus_system2_pt.recovery_nonfront_recall`、
`arms.system2_memory.val.recovery_nonfront_recall`、
`arms.system2_memory.val.normal_false_alarm`。

**读完立刻做一件事**：把 `arms.geometry` 与 `arms.system2` 的差和 `arms.system2_memory`
与 `arms.system2` 的差摆在一起。差在 3pt 以内就触发台账里那条**诚实条款**，
论文措辞必须改成"System2 缺的是几何量"，不能写成"学到的记忆表征"。

---

## 2. 13-B System2 提示注入（8 卡 × 约 3–4 h × 2 臂，网页提交）

提交物：[exp13-system2-memory-submission.md](exp13-system2-memory-submission.md)。
**只有 13-A 不是"否定"才跑。**

两臂串行，`exp13a`（`mode: memory`）在前、`exp13b`（`mode: constant`）在后。
config 只差一行，diff 已在提交里核对过。

### 启动后必须先确认的三件事

日志里**必须**出现：

```
System2 memory tokens enabled: mode=memory, tokens=8, M=256 -> 3584
✓ Loaded the deployed Past Head as a frozen input: tensors=79
DAgger System2 SFT relabelling (train): {... "corrected_fraction": ...}
```

- 第一行的 `mode` 必须与臂一致。**跑 `exp13b` 时它必须是 `constant`** ——
  否则跑的是treatment 的副本，两臂就没有对照了。
- 第二行必须正好 79。少一个说明历史头没载全，M_t 就不是部署时那个。
- 第三行的 `corrected_fraction` 必须与第 0 步预检的数字一致（同一套规则，同一批数据）。

### 评测（每臂一次）

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/tools/eval_system2_recovery_decisions.py \
  --config configs/ablation/exp13a_system2_memory_lora_8gpu.yaml \
  --checkpoint $R/model/exp13_system2_memory/exp13a/run_*/checkpoints/best.pth \
  --parent-checkpoint $V2 --collection-root $DAGGER --oracle-views $ORACLE \
  --shard-index 0 --shard-count 1 \
  --output-json $R/model/exp13_system2_memory/exp13a/decisions.json
```

判据要的两个数在 `recovery_turn_accuracy` 与 `normal_preservation`。
**同时看 `by_kind.*.first_token_texts`**：两臂都恒输出同一个 token 的话，
差值为 0 是结构决定的，不是能力差异（EXP-12 D2 就是这么翻的车）。

---

## 3. 13-C 闭环

**只有 13-A 与 13-B 都不否定才跑。** 开跑前必须先做一件 13-A/13-B 都没做的事：
**给 DAgger 集补 AMB3R VO 缓存并复验 13-B**。A/B 用的是 tar 里的 Habitat 真值位姿，
部署期是 VO；已知位姿域偏移把 pck8 从 0.88 压到 0.66（台账 §4）。
不补这一步，13-C 与 13-A/B 不同域，涨跌都无法归因。

评测复用 `scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh` 的修复栈与
`scripts/tools/paired_closed_loop_bootstrap.py`，两个协议种子（42/1337），
与 EXP-07 同一套口径。

---

## 4. 已知的坑（都在台账 §5 里，这里只列会打中本实验的）

- **§5 第 11 条**：`configs/ppa_action_refine_v2_8gpu.yaml` 有 5 个 `$VAR` 占位。
  漏导不会在启动时报错，而是在冷 AFS 载完模型 20 分钟后抛
  `HFValidationError: ... '$INTERNNAV_MODEL_PATH'`。两个启动脚本都已经在开跑前
  `ls -e` 过每一个，但手工单跑某一片时要自己导全。
- **§5 第 15 条**：配对比较之前先打印两侧的预测分布。13-A 的读出头与 13-B 的解码
  都可能塌成常数，那时差值为 0 与"没有能力差异"是两回事。
- **§0 第 7 条**：开发机最多 3 张卡。13-A/13-B 都是 8 卡，**一律走网页提交**。
  单卡补跑某一片可以在开发机上做。
- **空白容器里起任务到第一行日志要几分钟**（`CLAUDE.md` §4），不要急着杀。
