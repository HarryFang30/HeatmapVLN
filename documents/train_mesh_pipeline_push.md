# train_mesh_pipeline_push.md ｜ Mesh 数据→Baseline→Stage‑B 训练推进（给 Claude｜可直接执行）

> 现状速记：mesh 渲染链路已就位，数据已能达成 `K_eff≈3`，但样本量偏小（7/3），且 `avg H(q)≈5.6` 略高。目标是：**放大数据规模**、**压低标签熵**、**跑通 Baseline（Stage‑A）**，再进 **Stage‑B（GRU+局部解冻）**。

---

## 0) 成功标准（SLO）
- **数据**：`K_eff≥2` ≥ **85%**（train/val）；平均熵 `E[H(q)] ≤ 5.0`（64×64）。
- **过拟合**：`final_loss ≤ median(H(q))+0.3`（相对熵阈值 PASS）。
- **Stage‑A**：3 个 epoch 后 `val NLL < 8.0` 且**明显下降**；叠加图热点“贴脸”。

---

## 1) 放大数据规模（mesh 渲染）
> 目标：train=**120** clips / val=**30** clips；若显卡/时间紧，先跑半量（60/15），验收通过再补满。

**命令**：
```bash
# 清理旧数据
rm -rf ./data/habitat_vln/{train,val} ./raw_sequences/{train,val}
mkdir -p ./raw_sequences/train/RoomA ./raw_sequences/val/RoomB

# 训练集（建议 120 clips）
python scripts/gen_synth_demo.py \
  --root ./raw_sequences --scene RoomA --clips 120 \
  --T 8 --W 384 --H 384 \
  --pose_mode face_target --path_mode short_arc \
  --arc_deg 20 --radius 1.8 \
  --render_mode mesh --mesh_grid 28 --split train

# 验证集（建议 30 clips）
python scripts/gen_synth_demo.py \
  --root ./raw_sequences --scene RoomB --clips 30 \
  --T 8 --W 384 --H 384 \
  --pose_mode face_target --path_mode short_arc \
  --arc_deg 20 --radius 1.8 \
  --render_mode mesh --mesh_grid 28 --split val
```
> 说明：相较 grid=32，`mesh_grid=28` 可显著提速；`radius 1.8` + `arc 20°` 提升重叠，利于 K.

---

## 2) 压低标签熵 & 收紧可见性阈值
**修改** `configs/dataset_pack.yaml`（仅列出关键变动）：
```yaml
pack:
  lookback: 5               # 从 7 收回到 5（更近邻，减少跨越导致的漂移）
  min_visible_ratio: 0.02   # mesh 稠密→可收紧；若达不到 K_eff≥2，可暂回 0.015

heatmap:
  gaussian_sigma_px: 1.8    # 从 2.0 → 1.8，降低 H(q)（目标 ≤5.0）
  occlusion_check: true

export:
  drop_if_effective_k_below: 2
  mark_low_quality: true
```

---

## 3) 打包 & 质量报告
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val
```
**达标**：`K_eff≥2 ≥85%` 且 `avg H(q) ≤5.0`。若不达标，按顺序调：
1) `arc_deg: 20 → 15`
2) `radius: 1.8 → 1.6`
3) `mesh_grid: 28 → 32`
4) `min_visible_ratio: 0.02 → 0.015`

---

## 4) 过拟合（改为相对熵阈值）
> 代码已更新为：`final_loss ≤ median(H(q))+0.3` 即 PASS。
```bash
python scripts/overfit_one_batch.py
```
**预期**：PASS；且 `logits_std` 不塌、`grad_head_norm` 在 0.1～1.5 区间。

---

## 5) Baseline 训练（Stage‑A）
**确认/更新** `configs/training_config.yaml`（关键片段）：
```yaml
training:
  stages:
    - name: warmup_head
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [64, 64]
      use_gru: false
optim:
  head_lr: 3.0e-3
  lora_lr: 5.0e-5
  weight_decay: 1.0e-2  # 仅作用 backbone 组；Head/Renderer 在 param groups 用 1e-4/0
  grad_clip: 1.0
  amp: off              # 基线稳优先
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 8
```
**开训**：
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml
```
**判定**：`val NLL < 8.0` 且下降明显；可视化贴脸；诊断里 `logits_std`>0、`grad_head_norm` 稳定。

---

## 6) 评估与可视化
```bash
python scripts/eval_heatmap.py --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_warmup_head_epoch_3.pt --save-vis
```
产出：NLL/KL 指标、峰值定位误差、(RGB|GT|Pred) 叠加图到 `outputs/vis/`。

---

## 7) Stage‑B 推进（GRU + 局部解冻）
> 在 Stage‑A 达标后执行。

### 7.1 模型：开启 GRU 聚合
- 若已实现 `use_gru` 开关：把 Stage‑B 配置 `use_gru: true` 即可。
- 若未实现，请在 `VLNHeatmapModel` 中加入：
```python
# __init__
self.temporal = torch.nn.GRU(self.vision_dim, self.vision_dim//2, batch_first=True, bidirectional=True)
self.fuse_proj = torch.nn.Linear(self.vision_dim, self.vision_dim)

# forward（核心片段）
feat_bt = self.encoder(frames.view(B*T,3,H,W)).view(B,T,-1)
feat_bt, _ = self.temporal(feat_bt)
feat = self.fuse_proj(feat_bt.mean(dim=1))
logits = self.head(feat)
```

### 7.2 选择性解冻 + 参数分组
在 `train_multistage.py` 的优化器构造处：
```python
backbone_trainable = []
for n,p in model.named_parameters():
    if any(x in n for x in ['encoder.layer4', 'encoder.proj']):
        p.requires_grad = True; backbone_trainable.append(p)
    elif n.startswith('head') or 'renderer' in n:
        p.requires_grad = True
    else:
        p.requires_grad = False

optim = torch.optim.AdamW([
  {"params": [p for n,p in model.named_parameters() if n.startswith('head')], "lr": 3e-3, "weight_decay": 1e-4},
  {"params": [p for n,p in model.named_parameters() if 'renderer' in n], "lr": 1e-3, "weight_decay": 0.0},
  {"params": backbone_trainable, "lr": 1e-4, "weight_decay": 1e-4},
])
```

### 7.3 Stage‑B 配置补丁
```yaml
training:
  stages:
    - name: warmup_head_gru
      epochs: 3
      freeze_llm: true
      lora: false
      hm_size: [64, 64]
      use_gru: true
    - name: finetune_lastblock
      epochs: 6
      freeze_llm: false   # 仅解冻 layer4 + proj（由代码控制）
      lora: false
      hm_size: [64, 64]
optim:
  amp: bf16               # Stage‑B 可开启 AMP 提速
```
**判定**：相对 Stage‑A，`val NLL` 再下降 ≥0.2–0.5；可视化更清晰。

---

## 8) 性能与小贴士
- 生成很慢时：`mesh_grid 28→24` 或 `W/H 384→320`，但覆盖率需>70%。
- K_eff 偏低：先降 `arc_deg`（20→15），再降 `radius`（1.8→1.6）。
- 熵偏高：`gaussian_sigma_px` 继续降至 `1.6`（但过小会让监督过于尖锐）。

---

## 9) 提交与 PR
```bash
git checkout -b train/mesh-pipeline-push
printf "data/\nraw_sequences/\noutputs/\n__pycache__/\n*.pyc\n" >> .gitignore
git add -A
git commit -m "feat(train): mesh data scale-up, entropy control, baseline run, stage-B hooks"
git push -u origin train/mesh-pipeline-push
```

> PR 附上：train/val 的 `data_quality.json` 关键指标 + 过拟合曲线 + Stage‑A 可视化（3–5 张）。

