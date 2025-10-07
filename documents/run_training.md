# 训练接入与起跑手册（run\_training.md｜不涉及仿真）

> 你当前的状态：\*\*数据侧全链路已就绪\*\*（✅ 合成 demo、✅ pack、✅ 几何投影与热力图、✅ 读取与质检）。下一步就是把数据\*\*接入已有训练代码\*\*，做一次 \*\*smoke 前向+主损失\*\* 验证，然后开 \*\*Stage A/B\*\* 正式训练。
>
> 下面是给 Claude 的一步一步任务清单：\*\*不包含任何仿真依赖\*\*，只对接你已经产出的标准数据集。

---

## 0\) 目标

* 用你现有的数据根 `./data/habitat\_vln` 跑通：

  1. **冒烟测试**：前向 + KL/CE 主损失无报错；
  2. **Stage A**（冻结骨干，仅训 Head+Renderer）能稳定下降；
  3. **Stage B**（LoRA/部分层细调）继续下降并出可视化；
  4. 产出 checkpoint、可视化叠加与评估数值（NLL / 峰值误差）。

---

## 1\) 配置对接（configs/training\_config.yaml）

> 若已有训练配置文件，请在原文件基础上对齐下面关键字段；若没有，就按此新建。

```yaml
seed: 42

data:
  root: ./data/habitat\_vln
  frames\_per\_clip: 8
  heatmap\_per\_clip: 4
  image\_size: \[384, 384]
  init\_hm\_size: \[64, 64]

training:
  stages:
    - name: warmup\_head
      epochs: 2
      freeze\_llm: true
      lora: false
      hm\_size: \[64, 64]
    - name: finetune\_all
      epochs: 8
      freeze\_llm: false
      lora: true
      lora\_rank: 16
      hm\_size: \[128, 128]
    - name: finetune\_all\_highres
      epochs: 10
      freeze\_llm: false
      lora: true
      lora\_rank: 16
      hm\_size: \[224, 224]

optim:
  optimizer: adamw
  head\_lr: 1.0e-3
  lora\_lr: 5.0e-5
  weight\_decay: 1.0e-2
  grad\_clip: 1.0
  amp: bf16
  scheduler: cosine
  warmup\_ratio: 0.05
  batch\_size: 8
  grad\_accum\_steps: 1

loss:
  type: kl\_ce
  focal: {enabled: false, alpha: 0.25, gamma: 2.0}

log:
  out\_dir: ./outputs
  save\_every\_epochs: 1
  vis\_every\_steps: 200
  max\_ckpts: 3
```

> 小贴士：先用小 batch（如 8）稳起来；过平就调小 Renderer 的 `τ` 或 `sigma`。

---

## 2\) 冒烟测试（scripts/smoke\_train.py）

> 目的：不跑全程，\*\*验证 batch→model→loss\*\* 链路 OK。

**Claude：在 `scripts/` 新建 `smoke\_train.py`**（若已存在可复用）：

```python
import torch
from torch.utils.data import DataLoader
from src.data.vln\_heatmap\_adapter import VLNHeatmapDataset
from src.models.vln\_heatmap\_model import VLNHeatmapModel
from src.utils.losses import kl\_ce\_loss

def main():
    ds = VLNHeatmapDataset(
        root="./data/habitat\_vln", split="train",
        frames\_per\_clip=8, heatmap\_per\_clip=4,
        image\_size=(384,384), hm\_size=(64,64)
    )
    dl = DataLoader(ds, batch\_size=2, shuffle=True, num\_workers=4, pin\_memory=True)

    batch = next(iter(dl))
    frames = batch\["frames"]          # \[B,T,3,H,W]
    targets = batch\["gt\_heatmaps"]    # \[B,K,Hm,Wm]
    mask = batch\["mask"]              # \[B,K]

    model = VLNHeatmapModel(
        k\_heatmaps=targets.shape\[1], hm\_size=(64,64),
        vision\_dim=1024, agg="mean", use\_lora=False
    ).eval()

    with torch.no\_grad():
        preds, \_ = model(frames)
        loss = kl\_ce\_loss(preds, targets, mask=mask)
    print("preds:", preds.shape, "loss:", float(loss))

if \_\_name\_\_ == "\_\_main\_\_":
    main()
```

**命令**：

```bash
python scripts/smoke\_train.py
```

**期望**：打印 `preds: \[B,K,64,64] loss: <正数>`，无报错。

---

## 3\) 开训（scripts/train.py）

> 你之前的训练脚本若已支持“阶段切换 + 分辨率课程”，直接用；否则按下面关键点补齐。

### 3.1 批次适配（同时兼容旧/新字段）

```python
def unify\_batch(batch):
    if "frames" in batch and "gt\_heatmaps" in batch:
        return batch\["frames"], batch\["gt\_heatmaps"], batch.get("mask")
    # 兼容历史格式：video\_frames/target\_heatmaps...
    frames = batch\["video\_frames"]; targets = batch.get("target\_heatmaps")
    if targets is None: targets = batch\["target\_heatmap"].unsqueeze(1)
    return frames, targets, None
```

### 3.2 阶段循环（冻结/解冻 + 课程）

* Stage A：`freeze\_llm=True`，只训 **Head + Renderer**；
* Stage B：`freeze\_llm=False`，开 **LoRA**（小 rank）与 Head 一起训；
* 每阶段根据 `hm\_size` 重建/调整 DataLoader 与 Head/Renderer 的目标分辨率。

### 3.3 主损失与日志

* 主损失：`kl\_ce\_loss(preds, targets, mask)`；
* 训练中每 `vis\_every\_steps` 存一张叠加图（复用你的 `inspect\_dataset` 可视化函数）。

**命令**：

```bash
CUDA\_VISIBLE\_DEVICES=0 torchrun --nproc\_per\_node=1 scripts/train.py \\
  --config configs/training\_config.yaml
```

---

## 4\) 评估与可视化（scripts/evaluate.py）

> 若已存在评估脚本，确保输出以下指标；没有就新建一个轻量版。

* **主指标**：NLL/KL（与训练一致）；
* **峰值定位误差**：每张热图 `argmax` 与 GT 峰点像素距离；
* **叠加图**：保存到 `outputs/vis/`，便于人工巡检。

**命令**：

```bash
python scripts/evaluate.py --config configs/training\_config.yaml --ckpt path/to/ckpt.pt
```

---

## 5\) 成功判定（SLO）

* Smoke 前向无报错；
* Stage A 前 1–2 个 epoch 内 NLL 稳定下降；
* Stage B 相比 A 有进一步下降（或更清晰的热区）；
* 叠加可视化能看见热点贴合参考帧；
* `mask==0` 的通道比例 < 20%。

---

## 6\) 常见问题 \& 快速解法

* **输出过平**：Renderer 降低 `τ` 或 `σ`；必要时开启 Focal（γ=1–2）。
* **输出过尖/不稳**：增大 `τ` 或 `σ`；把分辨率课程放慢（64→128→224）。
* **热区漂移**：检查外参方向（`T\_world\_cam` 与逆矩阵用法）。
* **OOM**：先降 `batch\_size` 或 `frames\_per\_clip`，再升。
