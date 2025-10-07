# post_stageB_next_steps.md ｜ Stage‑B 之后怎么走（给 Claude｜可直接执行）

> 现状：Stage‑A→Stage‑B 已跑通，Val NLL 从 **8.277 → 8.036**。距离 8.0 只差 ~0.036。
> 目标：**先线性拿下 < 8.0**，再进入 **Stage‑C（128×128）**。Habitat 环境建议**稍后**再装（可同时做最小预备）。

---

## 成功标准（SLO）
- **短期**：Val NLL **< 8.0**，并有下降趋势；叠加图“贴脸”。
- **中期**：128×128 阶段稳定训练，显存/速度可接受。
- **数据**：`K_eff≥2 ≥ 85%`、`avg H(q) ≤ 5.0`（必要时小幅调参）。

---

## Path A（推荐）：不改数据，继续微调 2–3 个 epoch → 刷过 8.0
**思路**：从 `checkpoint_finetune_lastblock_epoch_6.pt` 继续，**略降 head LR**、**保持 GRU + 解冻 layer4+proj**，再训 2–3 epoch。

### 1) 配置补丁（Claude 请改）
**文件**：`configs/training_config.yaml`
```yaml
training:
  stages:
    - name: finetune_lastblock_continue
      epochs: 3              # 再训 3 个 epoch（可改 2）
      freeze_llm: false      # 仍只解冻 layer4 + proj（由代码控制）
      lora: false
      hm_size: [64, 64]
      use_gru: true
optim:
  head_lr: 2.0e-3           # 从 3e-3 略降，防过冲
  lora_lr: 1.0e-4
  weight_decay: 1.0e-2
  grad_clip: 1.0
  amp: bf16
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 8
```

### 2) 训练命令
```bash
INIT=outputs/checkpoints/checkpoint_finetune_lastblock_epoch_6.pt
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml \
  --init-ckpt $INIT \
  --tag stageB_continue
```

### 3) 评估与可视化
```bash
python scripts/eval_heatmap.py --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_finetune_lastblock_continue_epoch_3.pt \
  --save-vis
```

> **期望**：Val NLL **< 8.0**。若仍在 8.03–8.06，增加 1 个 epoch 或把 `head_lr: 2e-3 → 2.5e-3`。

---

## Path B：小幅降熵（数据轻调）→ 复训 1–2 个 epoch
**前提**：若 Path A 没过线，再动数据。把标签高斯再“尖一点”，并稍增重叠。

### 1) 修改 `configs/dataset_pack.yaml`
```yaml
pack:
  lookback: 5
  min_visible_ratio: 0.02
heatmap:
  gaussian_sigma_px: 1.6    # 从 1.8 降；注意别太小
  occlusion_check: true
```

### 2) 重新打包 & 报告
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val
```
**期望**：`K_eff≥2 ≥ 85%` 保持；`avg H(q)` 从 ~5.6 降到 **≤5.2**。

### 3) 基于 Stage‑B 最后权重，快速复训 1–2 个 epoch
```bash
INIT=outputs/checkpoints/checkpoint_finetune_lastblock_epoch_6.pt
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml --init-ckpt $INIT --max-epochs 2 \
  --tag stageB_entropy_squeeze
```
> **期望**：Val NLL 直接跌破 **8.0**。

---

## Path C：进入 Stage‑C（128×128）
> 在 A 或 B 过线后执行。关注显存与吞吐；不同分辨率的 NLL **不可直接横比**，看趋势和可视化。

### 1) 配置补丁（Claude 请改）
**文件**：`configs/training_config.yaml`
```yaml
training:
  stages:
    - name: upscale_128_head
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [128, 128]
      use_gru: true
    - name: upscale_128_lastblock
      epochs: 4
      freeze_llm: false
      lora: false
      hm_size: [128, 128]
      use_gru: true
optim:
  head_lr: 2.0e-3
  lora_lr: 1.0e-4
  batch_size: 4         # 显存保护（按需调整）
  amp: bf16
```

### 2) 训练 & 评估
```bash
INIT=outputs/checkpoints/checkpoint_finetune_lastblock_continue_epoch_3.pt
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml --init-ckpt $INIT --tag stageC_128
python scripts/eval_heatmap.py --config configs/training_config.yaml --ckpt outputs/checkpoints/checkpoint_upscale_128_lastblock_epoch_4.pt --save-vis
```

---



