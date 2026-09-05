# EXP-10 桥关臂 网站提交物

判据见台账 [EXP-10](README.md#exp-10-未来认知头与桥的关联是否可观测)。
**开跑前把判据再读一遍，跑完只填结果，不改判据。**

**为什么走网页提交而不是开发机**：本臂必须与桥开臂同为 **8 卡**——`--validate-only` 的
world size 会改变 val 分片与 512 对 rollout 的配对方式，缩成 3 卡跑出来的数**不能**
与既有桥开臂比较。而开发机上限是 3 卡（台账 §0 第 7 条），所以这里只能走网页。

对照臂（桥开）已完成，不需要重跑：
`model/exp09r_revalidate_512/best/run_20260904_050155/manifest/pre_training_validation.json`。

---

## 提交物

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export PPA_DATA_ROOT=$R/r2r_panoramic_data_v2/train
export PPA_AMB3R_CACHE_ROOT=$R/data/amb3r_endpoint_v3_full_r2r
export INTERNNAV_MODEL_PATH=$R/InternNav-Model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

RUNTIME=$R/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache
export HF_HOME=$RUNTIME/huggingface TORCH_HOME=$RUNTIME/torch XDG_CACHE_HOME=$RUNTIME/xdg
export MPLCONFIGDIR=$RUNTIME/matplotlib TRITON_CACHE_DIR=$RUNTIME/triton

export PPA_ACTION_REFINE_OUTPUT_ROOT=$R/model/exp10_bridge_off_512/best
export PPA_TENSORBOARD_ROOT=$R/model/exp10_bridge_off_512/best/tensorboard

$R/envs/qwen25/bin/python -m torch.distributed.run --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29761 \
  scripts/train.py \
  --config configs/ablation/exp10_bridge_off_revalidate_512_8gpu.yaml \
  --load-weights $R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth \
  --validate-only --distributed --num-workers 2 --pin-memory --prefetch-factor 2
```

要点：

- `--validate-only` 只跑一遍训练前验证就退出：**不训练、不存 checkpoint**，
  写完 `manifest/pre_training_validation.json` 即结束。
- `RUNTIME` 指向桥开臂用过的同一份缓存（已热），空白容器里能省掉一次冷加载。
- 空白容器起任务到第一行日志要几分钟，**不要急着杀**（`CLAUDE.md` §4）。
- 预期时长与桥开臂同量级（约 0.5 h）。

## 启动后必须先确认的一件事

日志里**必须**出现：

```
PPA bridge retrains from its exact-zero fresh state
```

这是"桥确实被归零"的唯一证据（`past_plan_action_reset_bridge: true` 生效）。
桥开臂的日志里**不应**有这句。**没有这句就作废重来**，因为那说明跑的其实是桥开臂的复本。

两份 config 已核对过，**剔除注释后逐行只差一处**（2026-09-05 在开发机上验证）：

```
164c164
<       past_plan_action_reset_bridge: false      # exp09r（桥开）
---
>       past_plan_action_reset_bridge: true       # exp10（桥关）
```

## 读结果

```bash
python3 - <<'PY'
import json, glob
KEYS = ('val_future_soft_iou', 'val_future_topk_support_recall', 'val_future_visibility_f1',
        'val_future_front_soft_iou', 'val_future_right_soft_iou',
        'val_future_back_soft_iou', 'val_future_left_soft_iou',
        'val_rollout_endpoint_error', 'val_rollout_endpoint_error_native',
        'val_rollout_action_agreement', 'val_rollout_pairs')
for path in sorted(glob.glob('/mnt/afs/liwenhao/agent/370910109/model/exp*_512/*/run_*/manifest/pre_training_validation.json')):
    m = json.load(open(path))['metrics']
    arm = path.split('/model/')[1].split('/run_')[0]
    print(arm, {k: round(m[k], 4) for k in KEYS if k in m})
PY
```

指标在 `manifest/pre_training_validation.json` 的 `metrics` 字段——
**不在 `logs/metrics.jsonl`**（台账 §5 第 16 条）。

## 判据读数（抄自台账，勿改）

对照的是桥开臂的 `val_future_soft_iou=0.2394` / `val_future_topk_support_recall=0.7718` /
`val_future_visibility_f1=0.9150`。

- **H1 支持**：top-k 支持召回差 > 1pt，**或** Soft-IoU 差 > 0.01。
- **H1 否定**：三项差都 < 0.2pt / 0.002。
- **H1 没测出来**：之间。

⚠️ **结论措辞已按 EXP-12 收窄**：EXP-12 D2 测到未来头的**方向输出**在恢复状态下是常数"前"
（3743/3766），所以本臂即使支持，也只能写成"注入改变了**前视未来热力图内部的空间分布**"，
不能写成"注入改变了未来预测"。判据数值不变，这是收窄结论范围，不是放宽判据。
