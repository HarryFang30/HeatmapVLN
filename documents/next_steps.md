# next_steps.md｜从“能跑”到“能学”的下一步计划（给 Claude 的执行清单）

> 目标：基于你**已完成**的数据链路与训练脚手架，按最小成本把模型从“smoke=通了”推进到“**能学会**、指标可复现”。
>
> 特色：每一步都给出**可直接运行的命令**与**可粘贴的代码/配置片段**，不涉及任何仿真环境。

---

## 0) 前置与约定
- 数据根：`./data/habitat_vln`（已按 dataset.md 标准格式产出）。
- 训练脚本：`scripts/train_multistage.py`（Claude 已实现）。
- 评估脚本：`scripts/eval_heatmap.py`。
- 冒烟脚本：`scripts/smoke_train.py`（已通过）。
- 模型：`src/models/vln_heatmap_model.py`；Heatmap 头与 Renderer 已就绪。

> 若你的路径/文件名不同，请在执行时据实替换。

---

## 1) Sanity Check #1：单 batch 过拟合（必做）
**目的**：用 1–2 个 clip 让模型快速把训练 loss 砸到很低（最好 <1.0），验证“**能学**”。

### 1.1 新建脚本：`scripts/overfit_one_batch.py`
```python
import torch, os
from torch.utils.data import DataLoader
from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss

def main():
    ds = VLNHeatmapDataset(root='./data/habitat_vln', split='train',
                           frames_per_clip=8, heatmap_per_clip=4,
                           image_size=(384,384), hm_size=(64,64))
    # 只取前 2 个样本
    small = torch.utils.data.Subset(ds, list(range(min(2, len(ds)))))
    dl = DataLoader(small, batch_size=1, shuffle=True)

    model = VLNHeatmapModel(k_heatmaps=4, hm_size=(64,64), vision_dim=1024,
                            agg='mean', use_lora=False).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for step, batch in zip(range(800), dl):  # 最多 800 步
        preds, _ = model(batch['frames'])
        loss = kl_ce_loss(preds, batch['gt_heatmaps'], mask=batch.get('mask'))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step}: loss={float(loss):.4f}")
    print("final loss:", float(loss))

if __name__ == '__main__':
    main()
```

**运行**：
```bash
python scripts/overfit_one_batch.py
```
**判定**：loss 持续下降；数百步内明显 < 3.0，最好 < 1.0。

> 若不过拟合：先检查 Renderer 的 τ/σ/α、loss 的归一（每张热图 sum≈1）、softmax 维度是否是 `(H,W)`。

---

## 2) 准备“能学”的数据量（不依赖外部）
先用合成 demo 扩到一个能训练的规模，随后无缝替换为你的真实数据。

**生成合成数据（示例）**：
```bash
# 生成 150 个 train clip（RoomA），30 个 val clip（RoomB）
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomA --clips 150 --T 8 --W 384 --H 384
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomB --clips 30  --T 8 --W 384 --H 384

# 打包为标准训练格式（注意分 scene 切分）
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val

# 质检：随机抽 12 个 clip 叠加图
python scripts/inspect_dataset.py --root ./data/habitat_vln --split train --num 12
```

**质检指标**：
- `mask==0` 比例 < 20%；
- 叠加图肉眼合理，无大面积“飘移”。

---

## 3) 配置改动（直接可跑）
编辑 `configs/training_config.yaml`（或 Claude 已创建的同名文件），确保包含：

```yaml
seed: 42

data:
  root: ./data/habitat_vln
  frames_per_clip: 8
  heatmap_per_clip: 4
  image_size: [384, 384]
  init_hm_size: [64, 64]

training:
  stages:
    - name: warmup_head
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [64, 64]
    - name: finetune_all
      epochs: 8
      freeze_llm: false
      lora: true
      lora_rank: 16
      hm_size: [128, 128]
    - name: finetune_all_highres
      epochs: 10
      freeze_llm: false
      lora: true
      lora_rank: 16
      hm_size: [224, 224]

optim:
  optimizer: adamw
  head_lr: 1.0e-3
  lora_lr: 5.0e-5
  weight_decay: 1.0e-2
  grad_clip: 1.0
  amp: bf16
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 8
  grad_accum_steps: 1

loss:
  type: kl_ce
  focal: {enabled: false, alpha: 0.25, gamma: 2.0}

log:
  out_dir: ./outputs
  save_every_epochs: 1
  vis_every_steps: 200
  max_ckpts: 3
```

---

## 4) 跑一个占位骨干的 baseline（验证策略能收敛）
**命令**：
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml
```
**期望**：
- `train/val NLL` 明显下降（低于均匀基线 `~8.317`）；
- 叠加可视化中热点逐步清晰；
- Stage B（LoRA）相对 Stage A 进一步下降。

> 如果训练很慢或不稳，先只跑 Stage A（64×64、5～8 epoch）确认下降，再开启后续阶段。

---

## 5) 评估与可视化（固化结果）
**命令**：
```bash
python scripts/eval_heatmap.py \
  --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_best.pt \
  --save-vis
```
**产物**：
- NLL/KL、峰值定位误差；
- `outputs/vis/` 下保存（RGB | GT | Pred | 对照）叠加图。

---

## 6) Sanity Check #2：两条便宜的 baseline（可选）
用于确认“不是中心偏置/数据均值”在起作用。

### 6.1 新建：`scripts/baseline_center_gaussian.py`
```python
import numpy as np, torch, json, os
from glob import glob
from src.data.vln_heatmap_adapter import VLNHeatmapDataset

def center_gaussian(Hm, Wm, sigma=0.12):
    y,x = np.mgrid[0:Hm, 0:Wm]
    cy, cx = (Hm-1)/2, (Wm-1)/2
    d2 = (y-cy)**2 + (x-cx)**2
    hm = np.exp(-d2/(2*(sigma*max(Hm,Wm))**2)).astype('float32')
    s = hm.sum(); return torch.from_numpy(hm/s)

def main():
    ds = VLNHeatmapDataset('./data/habitat_vln','val',8,4,(384,384),(64,64))
    hm = center_gaussian(64,64)
    nll, n = 0.0, 0
    for i in range(min(len(ds), 200)):
        g = ds[i]['gt_heatmaps']  # [K,Hm,Wm]
        p = hm.expand_as(g)
        q = torch.softmax(g.view(g.size(0), -1), dim=-1)
        p = p.view(g.size(0), -1).clamp_min(1e-8)
        nll += float(-(q*torch.log(p)).sum(dim=-1).mean())
        n += 1
    print('Center-Gaussian baseline NLL:', nll/max(n,1))

if __name__=='__main__':
    main()
```

### 6.2 新建：`scripts/baseline_mean_heatmap.py`
```python
import torch
from torch.utils.data import DataLoader
from src.data.vln_heatmap_adapter import VLNHeatmapDataset

def main():
    ds = VLNHeatmapDataset('./data/habitat_vln','train',8,4,(384,384),(64,64))
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    import torch.nn.functional as F
    mean_hm = None; n = 0
    for b in dl:
        g = b['gt_heatmaps']  # [B,K,Hm,Wm]
        g = F.softmax(g.view(g.size(0), g.size(1), -1), dim=-1).view_as(g)
        mean_hm = g.mean(dim=(0,)) if mean_hm is None else mean_hm + g.mean(dim=(0,))
        n += 1
        if n > 100: break
    mean_hm = (mean_hm / n).mean(dim=0)  # [Hm,Wm]

    # 在 val 上评估
    val = VLNHeatmapDataset('./data/habitat_vln','val',8,4,(384,384),(64,64))
    import math
    nll, m = 0.0, 0
    for i in range(min(len(val), 200)):
        g = val[i]['gt_heatmaps']
        q = torch.softmax(g.view(g.size(0), -1), dim=-1)
        p = mean_hm.view(1, -1).expand(q.size(0), -1).clamp_min(1e-8)
        nll += float(-(q*torch.log(p)).sum(dim=-1).mean())
        m += 1
    print('Dataset-Mean baseline NLL:', nll/max(m,1))

if __name__=='__main__':
    main()
```

> 你的模型应 **显著优于** 这两条 baseline，说明确实学到了结构信息。

---

## 7) 替换强骨干（可在 baseline 通过后进行）
当占位 CNN 验证“策略能收敛”后，替换真正的视觉骨干（例如 Qwen2.5-VL 视觉塔或 ViT/DINO）。

### 7.1 修改点（不需仿真）
- 在 `src/models/vln_heatmap_model.py` 中实现 `_build_backbone()`：
  - 输入：`[B,T,3,H,W]`；输出：`[B,T,vision_dim]` 的时序特征；
  - 对齐 `vision_dim` 到 Heatmap 头输入；
  - 保留 `agg='mean'|'gru'` 的聚合分支。
- 在配置中新增：
```yaml
model:
  backbone: 'qwen'   # 'placeholder' | 'vit_b16' | 'qwen'
  vision_dim: 1024   # 按实际骨干更新
  agg: 'mean'        # 或 'gru'
```
- `train_multistage.py` 读取 `cfg['model']`，据此实例化模型。

**验证**：同样跑 Stage A → B，记录 NLL 与可视化；应优于占位 CNN。

---

## 8) Git 分支与推送（巩固基线）
```bash
# 新分支（如果尚未创建）
git checkout -b train

# 忽略数据/可视化产物
cat >> .gitignore <<'EOF'
data/
raw_sequences/
outputs/
__pycache__/
*.pyc
*.ipynb
EOF

# 提交本次改动（过拟合脚本 + baseline 脚本 + 配置更新）
git add -A
git commit -m "chore(exp): overfit-one-batch & baselines; update training config"

git push -u origin train
```

---

## 9) 成功判定（SLO）
- 单 batch 过拟合：loss < 1.0（或显著下降）；
- 占位 CNN baseline：val NLL 稳定低于 8.317；
- 更强骨干：val NLL 进一步下降，叠加热点更清晰；
- 质检：空热图比例 < 20%，对齐合理；
- 训练可复现：同一 seed 下 3 次 run 指标方差可接受（±0.1～0.2 NLL 以内）。

---

### 完成以上步骤后
你将拥有一个 **可复现、可评估、能收敛** 的热力图预测训练 baseline，并随时可替换更强骨干或接入真实数据源而无需改训练脚本。祝冲！🚀

