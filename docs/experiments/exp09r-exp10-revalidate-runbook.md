# EXP-09-R / EXP-10 runbook：在开发机上重验部署 checkpoint（512 对 rollout）

台账见 [README.md → EXP-09](README.md#exp-09-阶段三配方里哪一项真的在起作用) 与
[EXP-10](README.md#exp-10-未来认知头与桥的关联是否可观测)。这两项是**开发机直跑**的短任务
（各约 0.5 h，8 卡），不走网站提交；长任务仍按 `CLAUDE.md` §1.1 提交。

## 这跑的是什么

`scripts/train.py --validate-only` 只跑一遍"训练前验证"就退出：加载 `--load-weights` 的权重、
在 val 划分上算一遍全部指标（认知指标 + 512 对 bridged-vs-native 采样 rollout），
写 `manifest/pre_training_validation.json` 后结束，不训练、不存 checkpoint。

| 用途 | config | 桥 | 回答 |
|---|---|---|---|
| EXP-09-R 参照 | `configs/ablation/exp09r_stage3_v2_revalidate_512_8gpu.yaml` | 加载已训练的 v2 桥 | 消融臂要对照的"v2@512"基准 |
| EXP-10 桥关臂 | `configs/ablation/exp10_bridge_off_revalidate_512_8gpu.yaml` | 重置为精确零（Δ=0） | 注入是否改变未来预测 |

两份 config 只差 `past_plan_action_reset_bridge` 一行；都基于认证 v2 配方，只把
`val_rollout_batches` 从 8 提到 64（即 64 对 → 512 对），并打开 `evaluate_before_training`。

## 跑法

> ⚠️ **2026-09-05 更新：本节的"开发机 8 卡直跑"已作废。** 台账 §0 第 7 条规定
> **开发机最多 3 张卡**，4 卡及以上一律走网页提交。下面这段保留是因为 **EXP-09-R 参照臂
> 与 EXP-10 桥开臂确实是 2026-09-04 用它在开发机上跑出来的**（那是本规定生效之前），
> 是既有结果的出处记录，**不要照抄再跑**。
>
> 尚未跑的 **EXP-10 桥关臂**请用网页提交物：
> [exp10-bridge-off-submission.md](exp10-bridge-off-submission.md)。
> 它必须与桥开臂同为 8 卡——world size 会改变 val 分片与 512 对 rollout 的配对，
> 缩成 3 卡的数**不能**与桥开臂比较。

以下为历史记录（EXP-09-R / EXP-10 桥开臂的实际跑法）。
ssh 链路经中转、掉线会杀裸进程，所以用 tmux：

```bash
tmux new-session -d -s exp09r "bash -lc '/tmp/exp09r_revalidate.sh 2>&1 | tee /tmp/exp09r.log'"
```

`/tmp/exp09r_revalidate.sh` 的内容（EXP-09-R 参照臂，依次跑 epoch 3 的 `best.pth` 与 `epoch_004.pth`）：

```bash
R=/mnt/afs/liwenhao/agent/370910109
export PPA_DATA_ROOT=$R/r2r_panoramic_data_v2/train
export PPA_AMB3R_CACHE_ROOT=$R/data/amb3r_endpoint_v3_full_r2r
export INTERNNAV_MODEL_PATH=$R/InternNav-Model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
RUNTIME=$R/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache
export HF_HOME=$RUNTIME/huggingface TORCH_HOME=$RUNTIME/torch XDG_CACHE_HOME=$RUNTIME/xdg
export MPLCONFIGDIR=$RUNTIME/matplotlib TRITON_CACHE_DIR=$RUNTIME/triton

cd $R/HeatmapVLN
for ckpt in best epoch_004; do
  out=$R/model/exp09r_revalidate_512/$ckpt
  export PPA_ACTION_REFINE_OUTPUT_ROOT=$out PPA_TENSORBOARD_ROOT=$out/tensorboard
  $R/envs/qwen25/bin/python -m torch.distributed.run --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port=29751 \
    scripts/train.py --config configs/ablation/exp09r_stage3_v2_revalidate_512_8gpu.yaml \
    --load-weights $R/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/${ckpt}.pth \
    --validate-only --distributed --num-workers 2 --pin-memory --prefetch-factor 2
done
```

EXP-10 桥关臂：**改走网页提交**，见 [exp10-bridge-off-submission.md](exp10-bridge-off-submission.md)。
内容与上面同一脚本、只换 config 为 `exp10_bridge_off_revalidate_512_8gpu.yaml`、
输出根换成 `model/exp10_bridge_off_512/best`，只跑 `best.pth`（与桥开臂同一 checkpoint）。
启动日志必须出现 `PPA bridge retrains from its exact-zero fresh state`——那是桥关臂的证据；
桥开臂**不应**出现这句。

## 读结果

```bash
python3 - <<'PY'
import json, glob
for path in sorted(glob.glob('/mnt/afs/liwenhao/agent/370910109/model/exp*_512/*/run_*/manifest/pre_training_validation.json')):
    m = json.load(open(path))['metrics']
    print(path.split('/model/')[1].split('/run_')[0],
          {k: round(m[k], 4) for k in (
              'val_rollout_endpoint_error', 'val_rollout_endpoint_error_native',
              'val_rollout_endpoint_gap', 'val_rollout_action_agreement',
              'val_rollout_pairs', 'val_future_soft_iou',
              'val_future_topk_support_recall', 'val_future_visibility_f1',
              'val_heatmap_joint_pck8') if k in m})
PY
```

指标在 `manifest/pre_training_validation.json` 的 `metrics` 字段，
**不在 `logs/metrics.jsonl`**（那里只有 run/checkpoint 记账行）——台账 §5 第 16 条。

判据在台账里，不在这里；EXP-09 的消融臂一律与"v2@512"比较，不与 v2 run 自己的 64 对数字比较。

桥开臂的全部读数已经入库到台账的 EXP-10 条目（未来头 / 历史头 / rollout 三组），
引用时直接引台账，不要重新从日志里抄。
