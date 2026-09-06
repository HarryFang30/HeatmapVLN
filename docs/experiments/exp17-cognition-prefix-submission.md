# EXP-17 认知前缀两臂 网站提交物

判据见台账 [EXP-17](README.md#exp-17-认知前缀慢系统在决定之前显式写出来路与进度)。
**开跑前把判据再读一遍，跑完只填结果，不改判据。** 20% 占位比例写死。

两臂串行，`exp17a`（位姿 token、无前缀）在前、`exp17b`（位姿 token + 认知前缀）在后。
配置只差注册过的行（`tests/test_exp17_configs.py`）。`geometry` 模式下冻结历史头照常载入但前向不跑它。

**为什么走网页**：8 卡（手工 all-reduce，不是 DDP），两臂串行约 8–10 小时。台账 §0 第 7 条。

---

## 提交物

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export EXP13_FJL_ROOT=$R
export EXP13_TRAIN_ROOT=$R/model/exp17_cognition_prefix
export EXP13_DAGGER_ROOT=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl
export EXP13_PARENT_CHECKPOINT=$R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export R2R_TRAIN_JSON=$R/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
export EXP13_ARMS="exp17a exp17b"
export EXP13_EPOCHS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_exp13_system2_memory_8gpu_mxc500.sh
```

启动脚本与 EXP-13/14 是同一个，只多了 `R2R_TRAIN_JSON`（进度分母）和两个新臂名。

---

## 起步很慢是正常的

同 13-B：sealed 账本要过一遍（约 15 分钟），之后才载 7B 模型；第一行训练日志约 20 分钟后出现。

## 启动后必须先确认的五件事

`run_*/logs/train.log` 里：

```
System2 memory tokens: mode=geometry, tokens=8 ...
DAgger System2 SFT relabelling (train): {... "stop_supervision": true, "cognition_prefix": true, "prefix_placeholder_fraction": 0.2, "prefix_placeholder_rows": N ...}
✓ Loaded the deployed Past Head as a frozen input: tensors=79
🔄 Broadcasting trainable module: system2_memory
```

1. `mode=geometry`，两臂都是。
2. exp17b 的 relabelling 行里 `"cognition_prefix": true` 且 `prefix_placeholder_fraction: 0.2`；exp17a 是 `false / 0.0`。
   `prefix_placeholder_rows` 应约为 train 状态数的五分之一（shard_00 预览：5318 里 1031）。
3. `tensors=79`。
4. `vlm_lora` 与 `system2_memory` 两条 Broadcasting 都在。
5. 看到 `sentinel embeddings were prepared but never substituted` → 注入断了，作废重来。

---

## 评测（每臂一次，开发机三张卡分片，约 1 小时）

```bash
R=/mnt/afs/liwenhao/agent/370910109; PY=$R/envs/qwen25/bin/python
D=$R/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17
export INTERNNAV_MODEL_PATH=$R/InternNav-Model EXP13_OUTPUT_ROOT=/tmp/exp17_eval EXP13_TENSORBOARD_ROOT=/tmp/exp17_eval/tb
export EXP13_ORACLE_VIEWS=$R/model/exp12_recovery_gate/d1_per_state.jsonl R2R_TRAIN_JSON=$R/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
export DAGGER_ROOT_00=$D/shard_00 DAGGER_ROOT_01=$D/shard_01 DAGGER_ROOT_02=$D/shard_02 DAGGER_ROOT_03=$D/shard_03
export DAGGER_POLICY_FINGERPRINT=$($PY -c "import json;print(json.load(open('$D/shard_00/collection_manifest.json'))['contract']['policy_fingerprint'])")
cd $R/HeatmapVLN
ARM=exp17b; CFG=configs/ablation/exp17b_c3_geometry_prefix_stop_lora_8gpu.yaml
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((5+i)) setsid nohup $PY scripts/tools/eval_system2_cognition_prefix.py \
    --config $CFG --checkpoint $R/model/exp17_cognition_prefix/$ARM/run_*/checkpoints/best.pth \
    --parent-checkpoint $R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth \
    --collection-root $D --oracle-views $EXP13_ORACLE_VIEWS \
    --passes natural,placeholder,no_pose --shard-index $i --shard-count 3 \
    --output-json $R/model/exp17_cognition_prefix/$ARM/decisions_generated.shard$i.json \
    > $R/model/exp17_cognition_prefix/$ARM/eval_shard$i.log 2>&1 < /dev/null &
done
# 三片跑完后合并：
$PY scripts/tools/eval_system2_cognition_prefix.py --passes natural,placeholder,no_pose \
  --merge $R/model/exp17_cognition_prefix/$ARM/decisions_generated.shard0.json \
  --merge $R/model/exp17_cognition_prefix/$ARM/decisions_generated.shard1.json \
  --merge $R/model/exp17_cognition_prefix/$ARM/decisions_generated.shard2.json \
  --output-json $R/model/exp17_cognition_prefix/$ARM/decisions_generated.json
```

exp17a 把 `--passes natural,no_pose`；exp14a/exp14b 用各自的 config 与 checkpoint、`--passes natural`，
输出到 `model/exp17_cognition_prefix/{exp14a,exp14b}/decisions_generated.json`，这样四臂是同一口径。

**读数之前先看 `passes.natural.decision_distribution`**：一个恒答同一 token 的模型什么差值都是 0。

---

## 产物

| 物 | 路径（`/mnt/afs/liwenhao/agent/370910109/` 下） |
|---|---|
| 两臂训练 run | `model/exp17_cognition_prefix/{exp17a,exp17b}/run_*/` |
| **生成式决策评测（判据来源）** | `model/exp17_cognition_prefix/{exp17a,exp17b,exp14a,exp14b}/decisions_generated.json` |

## 边界（跑之前就要知道）

- 训练输入位姿是 Habitat 真值，部署是 AMB3R；本臂无噪声增广，闭环前必须处理（回填缓存或加增广）。
- 决策级不是闭环；SR 由后续 600 集金丝雀与两种子非劣检验说话。
- 慢系统预训练见过全部 61 个 train 场景，"留出"只对本项目微调成立。
