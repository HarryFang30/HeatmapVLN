# HeatmapVLN 训练实施手册（按现有项目结构重排版）

> 本手册覆盖 **放置路径、改动清单、函数签名、配置与命令**，完全贴合你当前的仓库树（`configs/`, `scripts/`, `src/data/`, `src/models/`, `src/utils/`, 顶层 `models/` 存放权重）。
>
> 目标：在 Habitat 导航数据上 **稳定预测 K 张热力图**。方法：**两阶段训练（先冻后融）+ 分辨率课程 + KL/CE 主损失**。

---

## 0) 关键约定
- **输入**：一段导航子序列 `T` 帧（RGB），可选文本指令。
- **输出**：`[K, Hm, Wm]` 的热力图概率分布（每张图 `sum=1`）。
- **GT**：由 Habitat 位姿/深度几何投影生成，训练前 **轻度高斯平滑 + 归一化**。
- **阶段**：
  - Stage A：冻结 Qwen2.5‑VL（与其他重骨干），仅训 **MLP Head + 可学习渲染**。
  - Stage B：开启 **LoRA**（或部分层）端到端细调。
- **课程**：热力图尺寸 `64 → 128 → 224/256`。

---

## 1) 放置位置与改动清单（按你当前结构）
```
configs/
  └─ training_config.yaml          # ✨ 新/改：三阶段 + 课程 + 优化器/损失

scripts/
  ├─ train.py                      # ✨ 改：统一调度多阶段；仅主损失
  ├─ evaluate.py                   # ✨ 改：NLL/KL 验证 + 可视化调用
  ├─ pretrain.py                   # （可保留，不强依赖）
  └─ train_finetune.py             # （可保留，不强依赖）

src/
  ├─ data/
  │  └─ vln_dataset.py             # ✨ 改：新增 `VLNHeatmapDataset`（返回 frames+gt）
  │                                #      支持 hm_size 课程、通道 mask
  ├─ models/
  │  ├─ vln_heatmap_model.py       # ✨ 新：组装骨架（backbone→聚合→Head→Renderer）
  │  ├─ heatmap/
  │  │  ├─ multi_head.py           # ✨ 新：`MultiHeatmapHead`（输出 K×Hm×Wm logits）
  │  │  └─ renderer.py             # ✨ 新：`GaussianRenderer`（softmax(τ)+高斯平滑(σ,α)`）
  │  ├─ mlp/mlp.py                 # （复用；如需可在此放置简单 MLP）
  │  ├─ qwen2_5_vl/…               # （已存在：Qwen 配置/封装）
  │  ├─ heatmap/generator.py       # （保持不动；若你更愿意，也可把 Head/Renderer 合并到这里）
  │  └─ spatial_mllm_enhanced.py   # （可选：若已有统一前向，可在此直接调用 Head/Renderer）
  └─ utils/
     ├─ losses.py                  # ✨ 新：`kl_ce_loss`（支持 mask），可选 Focal 开关
     ├─ metrics.py                 # （已有：补充 NLL/峰值误差）
     └─ visualization.py           # （已有：用于叠加图保存）

models/                              # ✅ 保持：HF 权重与 VGGT/DINOv3 权重
```

> 说明：为**最小侵入式**改动，Head/Renderer 作为独立小模块加入 `src/models/heatmap/`，主模型在 `src/models/vln_heatmap_model.py` 里拼装；数据仍走 `src/data/vln_dataset.py`；训练主入口统一用 `scripts/train.py`。

---

## 2) 配置（`configs/training_config.yaml`）
```yaml
seed: 42

# 数据
data:
  root: /path/to/habitat_dataset
  frames_per_clip: 8             # T
  heatmap_per_clip: 4            # K
  image_size: [384, 384]
  init_hm_size: [64, 64]         # 课程起始
  num_workers: 8
  pin_memory: true

# 三阶段 + 课程
training:
  stages:
    - name: warmup_head
      epochs: 2
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

# 优化与调度
optim:
  optimizer: adamw
  head_lr: 1.0e-3
  lora_lr: 5.0e-5
  weight_decay: 1.0e-2
  grad_clip: 1.0
  amp: bf16
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 16
  grad_accum_steps: 1

# 损失（仅主损失）
loss:
  type: kl_ce                   # kl_ce / mse 可切换
  focal:
    enabled: false
    alpha: 0.25
    gamma: 2.0

# 日志/保存
log:
  out_dir: ./outputs
  save_every_epochs: 1
  vis_every_steps: 200
  max_ckpts: 3
```

---

## 3) 数据集（`src/data/vln_dataset.py`）
**新增/改动点**：实现 `VLNHeatmapDataset`，返回 `frames[T,3,H,W]` 与 `gt_heatmaps[K,Hm,Wm]`（通道各自归一化）。GT 若全零，置 `mask[k]=0` 以在损失中跳过。

**建议接口**：
```python
class VLNHeatmapDataset(Dataset):
    def __init__(self, root, split, frames_per_clip, heatmap_per_clip,
                 image_size=(384,384), hm_size=(64,64)):
        # 1) 索引样本；2) 定义图像/热图 transform；3) 读入几何投影生成的专家热力图
        ...
    def __len__(self): ...
    def __getitem__(self, idx):
        # 返回 dict：{"frames": Tensor[T,3,H,W], "text": Optional[str],
        #             "gt_heatmaps": Tensor[K,Hm,Wm], "mask": Tensor[K], "meta": {...}}
        ...
```

> 注：`hm_size` 将被训练脚本按阶段动态更新（分辨率课程）。

---

## 4) 模型拼装（`src/models/vln_heatmap_model.py`）
**职责**：统一前向：Backbone（Qwen 视觉塔或你现有封装）→ 时序聚合（mean/GRU）→ `MultiHeatmapHead`（K×Hm×Wm logits）→ `GaussianRenderer`（softmax(τ)+平滑）。

**建议最小实现**：
```python
class VLNHeatmapModel(nn.Module):
    def __init__(self, k_heatmaps, hm_size=(64,64), vision_dim=1024,
                 agg='mean', use_lora=False, lora_rank=16):
        super().__init__()
        self.backbone = build_qwen_vision_backbone(models_root='./models', ...)
        if use_lora:
            inject_lora(self.backbone, rank=lora_rank, target_modules=["q_proj","v_proj","k_proj","o_proj"])  # 视可用性
        self.temporal = nn.GRU(vision_dim, vision_dim//2, num_layers=1,
                               batch_first=True, bidirectional=True) if agg=='gru' else None
        fused_dim = vision_dim
        from src.models.heatmap.multi_head import MultiHeatmapHead
        from src.models.heatmap.renderer import GaussianRenderer
        self.head = MultiHeatmapHead(in_dim=fused_dim, k_heatmaps=k_heatmaps, hm_size=hm_size)
        self.renderer = GaussianRenderer(hm_size=hm_size)

    def forward(self, frames, text=None):  # frames: [B,T,3,H,W]
        feats = self.backbone(frames)      # 期望 [B,T,vision_dim]
        if self.temporal is not None:
            feats, _ = self.temporal(feats)
        fused = feats.mean(dim=1)
        logits = self.head(fused)          # [B,K,Hm,Wm]
        probs  = self.renderer(logits)     # [B,K,Hm,Wm]，每张概率分布
        return probs, {"logits": logits}
```

> 若你已有 `spatial_mllm_enhanced.py` 的统一管线，也可在其中直接调用 `MultiHeatmapHead` 与 `GaussianRenderer`，减少新文件数量。

---

## 5) 头与渲染（`src/models/heatmap/`）
**(a) `multi_head.py`**
```python
class MultiHeatmapHead(nn.Module):
    def __init__(self, in_dim, k_heatmaps, hm_size):
        super().__init__()
        Hm, Wm = hm_size
        out_dim = k_heatmaps * Hm * Wm
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
        self.k, self.hm_size = k_heatmaps, hm_size
    def forward(self, x):
        B = x.size(0)
        y = self.mlp(x).view(B, self.k, *self.hm_size)
        return y  # logits
```

**(b) `renderer.py`**（softmax 温度 τ + 可分离高斯平滑）：
```python
class GaussianRenderer(nn.Module):
    def __init__(self, hm_size):
        super().__init__()
        self.hm_size = hm_size
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(1.5)))
        self.alpha = nn.Parameter(torch.tensor(0.6))
    def forward(self, logits):
        tau = torch.exp(self.log_tau).clamp(1e-2, 10.)
        probs = torch.softmax(logits / tau, dim=(-1, -2))
        sigma = torch.exp(self.log_sigma)
        probs = gaussian_separable_blur(probs, sigma)  # 实现为两次1D卷积
        return self.alpha * probs + (1 - self.alpha) * torch.softmax(logits / tau, dim=(-1,-2))
```

---

## 6) 损失（`src/utils/losses.py`）
仅保留**主损失**（便于稳定起步）。对 GT 先做 `softmax`（若上游已归一则可直用），对 **mask==0** 的通道跳过。
```python
def kl_ce_loss(pred_probs, target_maps, mask=None, eps=1e-8):
    B,K,Hm,Wm = pred_probs.shape
    q = torch.softmax(target_maps.view(B*K, -1), dim=-1)
    p = pred_probs.view(B*K, -1).clamp_min(eps)
    loss = -(q * torch.log(p)).sum(dim=-1)
    if mask is not None:
        m = mask.view(B*K).float()
        return (loss * m).sum() / m.sum().clamp_min(1.)
    return loss.mean()
```
> 若后期需要，可在此处添加 Focal（默认关闭）。

---

## 7) 训练脚本（`scripts/train.py`）
统一调度 **Stage A/B** 与 **分辨率课程**：
```python
cfg = load_yaml('configs/training_config.yaml'); set_seed(cfg['seed'])
# 1) 初始 hm_size
hm0 = tuple(cfg['training']['stages'][0]['hm_size'])
train_set = VLNHeatmapDataset(cfg['data']['root'], 'train', ..., hm_size=hm0)
val_set   = VLNHeatmapDataset(cfg['data']['root'], 'val',   ..., hm_size=hm0)
train_loader = DataLoader(train_set, batch_size=cfg['optim']['batch_size'], shuffle=True,
                          num_workers=cfg['data']['num_workers'], pin_memory=cfg['data']['pin_memory'])
val_loader   = DataLoader(val_set,   batch_size=cfg['optim']['batch_size'], shuffle=False,
                          num_workers=cfg['data']['num_workers'], pin_memory=cfg['data']['pin_memory'])

# 2) 模型
model = VLNHeatmapModel(k_heatmaps=cfg['data']['heatmap_per_clip'], hm_size=hm0,
                        vision_dim=1024, agg='mean', use_lora=cfg['training']['stages'][0]['lora'],
                        lora_rank=cfg['training']['stages'][0].get('lora_rank',16)).to(device)

# 3) 分阶段循环
for stage in cfg['training']['stages']:
    # 3.1 冻结/解冻
    set_freeze(model.backbone, freeze=stage['freeze_llm'])
    # 3.2 变更 hm_size（课程）
    hm_size = tuple(stage['hm_size'])
    train_set.hm_size = hm_size; val_set.hm_size = hm_size
    model.head.hm_size = hm_size  # 如需，重建 head/renderer 或提供 reset()
    # 3.3 构建优化器与调度（参数分组：head/renderer 用 head_lr；LoRA/backbone 用 lora_lr）
    groups = build_param_groups(model, head_lr=cfg['optim']['head_lr'], lora_lr=cfg['optim']['lora_lr'], wd=cfg['optim']['weight_decay'])
    optimizer = torch.optim.AdamW(groups)
    scheduler = build_scheduler(optimizer, cfg['optim'])

    for epoch in range(stage['epochs']):
        model.train()
        for batch in train_loader:
            frames = batch['frames'].to(device)
            targets= batch['gt_heatmaps'].to(device)
            mask   = batch.get('mask')
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if cfg['optim']['amp']=='bf16' else torch.float16):
                preds, _ = model(frames)
                loss = kl_ce_loss(preds, targets, mask=mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
            optimizer.step(); scheduler.step()
        # 评估 + 保存
        eval_and_save(model, val_loader, cfg, stage_name=stage['name'], epoch=epoch)
```

> 可直接复用你现有的 `src/utils/logger.py` 做日志、`src/utils/visualization.py` 做叠加图。

---

## 8) 评估（`scripts/evaluate.py`）
主指标：**NLL/KL**（与训练一致）。附加：峰值定位误差（argmax 像素距离）。
```python
def eval_and_save(model, val_loader, cfg, stage_name, epoch):
    model.eval(); nll=0; n=0
    with torch.no_grad():
        for batch in val_loader:
            preds,_ = model(batch['frames'].to(device))
            nll += kl_ce_loss(preds, batch['gt_heatmaps'].to(device)).item(); n+=1
    score = nll/max(n,1)
    save_ckpt(model, cfg['log']['out_dir'], stage_name, epoch, score)
    return score
```

---

## 9) 运行命令
```bash
# 多卡（推荐）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/training_config.yaml

# 仅评估
python scripts/evaluate.py --config configs/training_config.yaml
```

---

## 10) 验收清单
- [ ] `VLNHeatmapDataset` 返回 `frames[T,3,H,W]` + `gt_heatmaps[K,Hm,Wm]` + `mask[K]`。
- [ ] `VLNHeatmapModel` 前向输出 `probs` 概率图（每张 sum=1）。
- [ ] `MultiHeatmapHead` 输出 `K×Hm×Wm logits`；`GaussianRenderer` 含 `τ, σ, α` 可学习参数。
- [ ] 主损失使用 `kl_ce_loss`（支持 mask）；训练脚本支持 **Stage A/B + 分辨率课程**。
- [ ] 评估记录 NLL，保存可视化与 checkpoint。

---

## 11) 故障排查
- **过平**：减小 τ 或减弱平滑（α↓/σ↓）；必要时打开 Focal。
- **过尖不稳**：增大 τ 或增强平滑（α↑/σ↑）；放慢升分辨率节奏。
- **坐标异常**：在 64×64 检查 `logits`/`probs`/GT 的像素对齐，再逐级放大。

> 到此，改动最小、充分复用你现有 `scripts/`、`src/` 与顶层 `models/` 权重布局；直接 `torchrun` 可训练与复现。

