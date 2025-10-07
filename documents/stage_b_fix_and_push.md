# stageB_fix_and_push.md ｜ 修好 Stage‑B 并推进（给 Claude｜可直接执行）

> 现状：Stage‑A 已跑完 3 个 epoch（Val ≈ 8.27，稳步下降），进入 Stage‑B 时出现**模型属性名不匹配**（`model.backbone` vs `model.encoder`）。
> 目标：① 修 trainer 的参数分组/解冻逻辑；② 开启 GRU 聚合 + 局部解冻（`encoder.layer4` + `encoder.proj`）；③ 继续训练并验证收益。
>
> 备注：如需把 Val NLL 再压一点，可在数据侧**可选**微调（见文末 Optional）。

---

## 0) 快速修复清单（改哪里）
- **`scripts/train_multistage.py`**：
  - A) 参数分组里，把 `backbone` 统一改为 **`encoder`**；
  - B) 新增对 **`temporal`**（GRU）与 **`fuse_proj`** 的分组（Stage‑B 时需训练）；
  - C) 新增 `--init-ckpt` 选项：从 Stage‑A 最优/最新权重启动；
  - D) 按 Stage 配置控制 `requires_grad`：Stage‑A 冻结 backbone，Stage‑B 只解冻 `encoder.layer4` + `encoder.proj`；
  - E) 允许 `use_gru` 通过配置打开。

- **`src/models/vln_heatmap_model.py`**（若尚未加 GRU）：
  - A) 增加 `self.temporal: GRU` 与 `self.fuse_proj: Linear`；
  - B) `forward` 中在 `use_gru` 时走 GRU → mean → fuse → head。

- **`configs/training_config.yaml`**：
  - A) 在 Stage‑A 后追加 Stage‑B 两个 stage；
  - B) `optim.amp: bf16` 打开。

---

## 1) 代码补丁（最小可运行）

### 1.1 修改：`scripts/train_multistage.py`
**要点**：
- 统一按 `encoder` 命名抓 backbone；
- 明确三个参数组：`head/renderer`、`temporal/fuse_proj`、`encoder.layer4+proj`；
- 新增 `--init-ckpt`；
- 根据 stage 配置切 `requires_grad`。

```python
# --- 顶部导入 ---
import re, torch, argparse, os

# --- argparse 追加 ---
parser.add_argument('--init-ckpt', type=str, default=None,
                    help='Initialize model weights from this checkpoint (e.g., Stage-A best)')

# --- 加载初始化权重 ---
if args.init_ckpt and os.path.exists(args.init_ckpt):
    sd = torch.load(args.init_ckpt, map_location='cpu')
    key = 'model' if 'model' in sd else 'state_dict'
    model.load_state_dict(sd.get(key, sd), strict=False)
    logger.info(f"Loaded init ckpt: {args.init_ckpt}")

# --- 每个 stage 进入前，按配置设置 requires_grad 并构造优化器参数组 ---
def build_param_groups(model, stage_cfg):
    # 1) 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    use_gru = bool(stage_cfg.get('use_gru', False))
    # 2) Head & Renderer：总是可训
    head_params = []
    renderer_params = []
    temporal_params = []
    backbone_params = []  # 仅 encoder.layer4 + encoder.proj

    for n,p in model.named_parameters():
        if n.startswith('head'):
            p.requires_grad = True; head_params.append(p)
        elif 'renderer' in n:
            p.requires_grad = True; renderer_params.append(p)
        elif use_gru and (n.startswith('temporal') or n.startswith('fuse_proj')):
            p.requires_grad = True; temporal_params.append(p)
        elif any(x in n for x in ['encoder.layer4', 'encoder.proj']) and not stage_cfg.get('freeze_llm', True):
            p.requires_grad = True; backbone_params.append(p)
        # 其余 encoder 层保持冻结

    # 3) 组装优化器参数组（可按需调 LR/WD）
    pg = [
        {"params": head_params, "lr": cfg.optim.get('head_lr', 3e-3), "weight_decay": 1e-4},
        {"params": renderer_params, "lr": 1e-3, "weight_decay": 0.0},
    ]
    if temporal_params:
        pg.append({"params": temporal_params, "lr": 1e-3, "weight_decay": 1e-4})
    if backbone_params:
        pg.append({"params": backbone_params, "lr": cfg.optim.get('lora_lr', 1e-4), "weight_decay": 1e-4})
    return pg

# --- 在每个 stage 开始时调用 ---
param_groups = build_param_groups(model, stage_cfg)
optimizer = torch.optim.AdamW(param_groups, betas=(0.9,0.999))
```

> 如果你原脚本里已有 param grouping，请**按上述逻辑替换**：关键是名字从 `backbone`→`encoder`，以及 GRU 的参数需加入训练。

---

### 1.2（如需）在 `src/models/vln_heatmap_model.py` 加 GRU
```python
# __init__ 中
self.use_gru = False  # 默认关，按 stage 打开
self.temporal = torch.nn.GRU(self.vision_dim, self.vision_dim//2, batch_first=True, bidirectional=True)
self.fuse_proj = torch.nn.Linear(self.vision_dim, self.vision_dim)

# forward 核心片段
feat_bt = self.encoder(frames.view(B*T,3,H,W)).view(B,T,-1)  # [B,T,C]
if getattr(self, 'use_gru', False):
    feat_bt, _ = self.temporal(feat_bt)
feat = feat_bt.mean(dim=1)
if getattr(self, 'use_gru', False):
    feat = self.fuse_proj(feat)
logits = self.head(feat)
```
> 若你已实现 `use_gru` 开关，请确保属性名为 `temporal` 与 `fuse_proj`，与上面的 param grouping 对齐。

---

## 2) 训练配置：追加 Stage‑B 两段
**文件**：`configs/training_config.yaml`
```yaml
training:
  stages:
    - name: warmup_head         # Stage‑A（已完成，可保留或再加2个epoch）
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [64, 64]
      use_gru: false

    - name: warmup_head_gru     # Stage‑B(1)：开 GRU，仅训 head+temporal
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [64, 64]
      use_gru: true

    - name: finetune_lastblock  # Stage‑B(2)：解冻 encoder.layer4+proj
      epochs: 6
      freeze_llm: false         # 仅 last block + proj 由代码层控制解冻
      lora: false
      hm_size: [64, 64]
      use_gru: true

optim:
  head_lr: 3.0e-3
  lora_lr: 1.0e-4
  weight_decay: 1.0e-2
  grad_clip: 1.0
  amp: bf16                 # Stage‑B 可开 AMP
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 8
```

> 如果你希望先把 Val < 8.0 再进 Stage‑B，可以把 `warmup_head.epochs` 临时改为 **5**，跑完再进 Stage‑B。

---

## 3) 一键执行（修完 → 接着训练）
```bash
# A. 以新分支提交修复
git checkout -b fix/stageB-encoder-gru
git add scripts/train_multistage.py src/models/vln_heatmap_model.py configs/training_config.yaml
git commit -m "fix(train): stageB param groups switch to encoder; add GRU groups; init-ckpt support"

# B. 从 Stage‑A 的 checkpoint 启动 Stage‑B（或先把 Stage‑A 补到5个epoch）
INIT=outputs/checkpoints/checkpoint_warmup_head_epoch_3.pt
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml \
  --init-ckpt $INIT

# C. 评估与可视化（对比 Stage‑A）
python scripts/eval_heatmap.py --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_finetune_lastblock_epoch_6.pt --save-vis
```

**判定**：相对 Stage‑A，`val NLL` 再下降 ≥ 0.2–0.5；(RGB|GT|Pred) 叠加图更“贴脸”。

---

## 4) 验证防回归（小脚本，可选）
新建 `scripts/lint_model_names.py`，在 CI 或本地先跑一遍，避免后续再出现命名不匹配：
```python
import torch
from src.models.vln_heatmap_model import VLNHeatmapModel
m = VLNHeatmapModel(...)
names = [n for n,_ in m.named_parameters()]
assert any('encoder.layer4' in n for n in names), 'missing encoder.layer4'
assert any(n.startswith('head') for n in names), 'missing head params'
assert any('renderer' in n for n in names), 'missing renderer params'
print('OK: param name layout as expected.')
```

---

## Optional｜想要更稳地把 Val 压到 < 8.0
- **数据侧（优先级高）**：
  - `configs/dataset_pack.yaml`: `gaussian_sigma_px: 1.6`（从 1.8 降一点，降低 H(q)）；
  - 生成参数：`arc_deg: 15`、`radius: 1.6`（加重叠）；
  - 重新 pack+report，确认 `K_eff≥2 ≥85%` 仍满足。
- **训练侧**：
  - Stage‑A 再加 **2 个 epoch**（总 5）；
  - 或把 `head_lr: 3e-3 → 5e-3`（短跑更快下降，注意观测 `logits_std` 不塌）。

---

## 提交与 PR
```bash
git push -u origin fix/stageB-encoder-gru
```
PR 里附：
- Stage‑A vs Stage‑B 的 `val NLL` 曲线对比；
- 3–5 张叠加可视化；
- 若有数据侧微调，贴上新的 `data_quality.json` 摘要。

---

### 成功标准复述
- 修复后 **训练脚本不再引用 model.backbone**，统一用 `encoder`；
- GRU + 局部解冻能正常训练；
- 相对 Stage‑A，`val NLL` 明显下降，且可视化更贴脸；
- 代码进入 PR，含命名防回归检查。

