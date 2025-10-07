# run_baseline_push.md ｜ 训练 Baseline 起跑与数据质量闭环（给 Claude 的执行清单｜无仿真）

> 目标：在 **不引入任何仿真** 的前提下，把“已可学习”的系统拉到**稳定收敛**：
> 1) 以可见性优先采样重打包并验收数据；
> 2) 完成单批过拟合复测；
> 3) 跑通 Stage‑A baseline（64×64，仅训 Head）；
> 4) 通过后再升级聚合/分辨率/骨干。

---

## 0) 前置条件
- 数据根：`./data/habitat_vln`（按 dataset.md / data_quality_push.md 标准）。
- 关键脚本：
  - `scripts/pack_dataset.py`, `scripts/report_data_quality.py`
  - `scripts/overfit_one_batch.py`, `scripts/train_multistage.py`, `scripts/eval_heatmap.py`
- 模型：`src/models/vln_heatmap_model.py`（ResNet18 帧编码 + logits 空间 CE）。

---

## 1) 重打包 & 数据体检（必须）
**命令**：
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val
```
**SLO（达标线）**：
- `K_eff ≥ 2` 占比 ≥ **80%**；
- 平均熵 `E[H(q)] ≤ 5.0`（64×64）；
- `mask==0` 通道占比 < **20%**；
- 叠加可视化肉眼合理（热点贴近目标区域）。

> 不达标 → 见 §2 的“旋钮”调参后重打包，再跑 report。

---

## 2) 数据侧“旋钮”（仅 pack 级改动）
在 `configs/dataset_pack.yaml` 的 `pack:`/`heatmap:` 小节中按需调：
```yaml
pack:
  sampler: visibility_fps        # visibility | visibility_fps | fps | uniform
  lookback: 5                    # 备选: 3 / 5 / 7
  min_visible_ratio: 0.02        # 备选: 0.01 / 0.02 / 0.03
  topk_by_visible_ratio: true
  keyframes: 4
  min_effective_k: 2
  depth_subsample: 4             # 评估可见性时的深度子采样

heatmap:
  gaussian_sigma_px: 2.0         # 备选: 1.5 / 2.0
```
**建议顺序**：
1) `lookback` 从 5 → 3；
2) `min_visible_ratio` 从 0.02 → 0.01（放宽）；
3) `gaussian_sigma_px` 从 2.0 → 1.5（更尖锐，降低熵）。

---

## 3) 单批过拟合复测（验证“能学”且更容易）
**命令**：
```bash
python scripts/overfit_one_batch.py
```
**期望**：数百步内 `loss < 3.0`（取决于标签熵）。若仍 >3.5：返回 §2 微调旋钮并重打包。

---

## 4) 跑 Stage‑A baseline（64×64，仅训 Head）
### 4.1 配置（`configs/training_config.yaml`）
```yaml
training:
  stages:
    - name: warmup_head
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [64, 64]
optim:
  head_lr: 3.0e-3
  lora_lr: 5.0e-5
  weight_decay: 1.0e-2   # 仅用于 backbone；Head/Renderer 在分组中用 1e-4/0
  grad_clip: 1.0
  amp: off               # baseline 阶段关闭 AMP，更易稳定
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 8
```
> 训练代码应已使用“参数分组”：Head/Renderer 提高 LR 并豁免 WD（bias/Norm/renderer 标量）。

### 4.2 开训
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml
```
**判定**：
- `val NLL` 稳定 **低于 8.0** 且呈下降趋势；
- 叠加可视化逐步清晰；
- 日志中 `logits_std` 不塌到 0，`grad_head_norm` 在 0.1～1+ 合理区间。

---

## 5) 评估与固化可视化
**命令**：
```bash
python scripts/eval_heatmap.py \
  --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_best.pt \
  --save-vis
```
**产物**：NLL/KL、峰值定位误差、(RGB|GT|Pred|对照) 叠加图（存 `outputs/vis/`）。

---

