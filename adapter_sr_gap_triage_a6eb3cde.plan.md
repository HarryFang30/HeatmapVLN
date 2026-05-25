---
name: Adapter SR Gap Triage
overview: 系统排查 Adapter 闭环 SR 25% vs InternNav 原生 baseline 63% 的 38 个百分点 gap。先跑 3 步轻量诊断（10-20 分钟）定位根因，命中 H2/H3/H5 单文件级 bug 时立即修复，命中 H1（VLM 输出退化）时只报告不修。
todos:
  - id: step1_pixel_goal
    content: "Step 1: grep eval.log 看 predicted pixel_goal 分布, 判定 H1"
    status: completed
  - id: step2_parity
    content: "Step 2: 跑 latent_parity_train_vs_eval.py 20 samples, 看 A/B/C VERDICT, 判定 H2 / H5"
    status: completed
  - id: step3_norm
    content: "Step 3: 临时加一行 traj_hs norm 日志, 跑 2 条 episode, 判定 H3"
    status: completed
  - id: decide_branch
    content: "根据 Step 1-3 结果分支: H1 报告停手 / H2 修 integration.py / H3 修 padding / H5 修 prompt"
    status: completed
  - id: fix_if_hit
    content: 如果命中 H2 / H3 / H5, 按 Section 6-8 详细修复 (单文件改动 < 50 行)
    status: cancelled
  - id: verify_parity
    content: 修复后重跑 Step 2 确认 latent parity 达 cosine >= 0.99
    status: cancelled
  - id: rerun_20ep
    content: 重跑 20 episode adapter eval, 报告新 SR 是否回升
    status: cancelled
  - id: cleanup_debug
    content: 清理 Step 3 临时日志, lint 检查, git 提交修复
    status: completed
isProject: false
---

# Adapter SR Gap Triage

## 1. 问题陈述

- 真基线: InternNav 原生 (front-view + InternNav eval) = SR 63%
- 当前路径 D: Stage1-S2 (panorama) + adapter + 冻结 NextDiT = SR 25%
- Adapter offline 指标良好 (traj_cosine=0.97, first_action_match=0.83)
- 38 个百分点 gap 不能用"adapter 丢失全景信息"解释，必须有具体技术原因

## 2. 假设清单 (按优先级)

- **H1** Stage1-S2 VLM 在闭环 Habitat panorama 输入上输出的 pixel_goal 退化 (最可能, offline 用 teacher coord 测的, 没暴露这里)
- **H2** 训练 forward (`_forward_batch_panorama_tokenized`) 与闭环 `generate_latents` 输出不同 latent → adapter OOD
- **H3** Commit `17663f9` attention_mask padding fix 引入数值副作用
- **H5** Eval `construct_input` prompt 与训练 `PanoramicTokenizedCollator` prompt 文本结构不一致
- **H4** (跳过, 走 Step 5 不在本计划内) `r2r_val_unseen.py` 状态机偏差
- **H6** (不重要, 不在本计划内) traj_images pix_goal 取当前 lookdown vs goal lookdown

## 3. Step 1 [5 分钟, 只读]: VLM pixel_goal 质量检视

**目标**: 判定 H1 是否命中。

**操作**:

```bash
LOG=/workspace/HeatmapVLN/logs/habitat_smoke_adapter_20ep/adapter_20ep/eval.log
# 替换为你 adapter 路径实际 eval.log

grep "predicted pixel_goal" "$LOG"
grep "predicted pixel_goal" "$LOG" | awk -F'[][]' '{print $2}' \
  | sort | uniq -c | sort -rn | head -30
grep "VLM output:" "$LOG" | head -30
```

**判读**:

- 坐标多样, 覆盖图像各区域, 跟 instruction 大致 align → H1 排除, 走 Step 2
- 坐标退化到中心 / 边缘 / 频繁 STOP / 文本乱码 → **H1 命中**, Stage1-S2 VLM 头是瓶颈, 报告后停止本计划 (修复 VLM 是大动作, 不在本排查 scope 内)

## 4. Step 2 [5 分钟, 只读]: Latent parity 三路对比

**目标**: 判定 H2 / H5 是否命中。

**操作**:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluation/latent_parity_train_vs_eval.py \
  --config configs/train_config_internnav_8gpu_stage2_wider.yaml \
  --base-checkpoint $BASE_CKPT \
  --num-samples 20 \
  2>&1 | tee /tmp/parity.log
```

脚本现成: [scripts/evaluation/latent_parity_train_vs_eval.py](scripts/evaluation/latent_parity_train_vs_eval.py)

三条路径:

- A: eval pipeline (model.generate + condition + generate_latents)
- B: training collator forward
- C: 同 B input 但走 generate_latents

脚本自动打 VERDICT。判读分支:

- C vs B cosine >= 0.99 且 A vs B cosine >= 0.99 → H2 / H5 排除, 走 Step 3
- C vs B cosine >= 0.99 但 A vs B cosine < 0.99 → **H5 命中**, 修复 (见 Section 7)
- C vs B cosine < 0.99 → **H2 命中**, 修复 (见 Section 6)

## 5. Step 3 [10 分钟, 加一行日志]: traj_hs norm 检查

**目标**: 判定 H3 是否命中。

**操作**: 在 [scripts/evaluation/r2r_val_unseen.py](scripts/evaluation/r2r_val_unseen.py) 的 `_run_eval_panoramic_vlm` 中, `generate_latents` 返回 `_last_traj_hs` 之后, adapter 应用之前, 加临时日志:

```python
import torch as _torch
_per_q = [float(_last_traj_hs[0, i].float().norm().item()) 
          for i in range(_last_traj_hs.shape[1])]
print(f"  [debug] traj_hs total_norm="
      f"{float(_last_traj_hs.float().norm().item()):.3f} "
      f"per_query={_per_q}", flush=True)
```

跑 2 条 episode (`--max_episodes 2`), grep `traj_hs total_norm`。

**判读** (teacher 训练时 norm 范围 60-80):

- norm 落在 [50, 100] → H3 排除, 综合 Step 1-3 给出最终结论
- norm 远高 (> 200) 或远低 (< 10) → **H3 命中**, attention_mask padding 改坏了 latent 数值

**清理**: 诊断完后删掉这一行临时日志, 不留 in-tree。

## 6. H2 命中的修复方案

**根因**: [src/models/qwen2_5_vl/integration.py](src/models/qwen2_5_vl/integration.py) 中 `extract_traj_hidden_states` 走的 `_forward_model_inputs` 路径与训练用的 `_forward_batch_panorama_tokenized` 在 panoramic 多视角处理上发散。

**修复方向** (确认 H2 命中后细化):

- 让 `generate_latents` 在 panoramic 模式下也走 `_forward_batch_panorama_tokenized` 路径
- 或者在 `r2r_val_unseen.py` 闭环里直接用 collator 重新打包输入再调 train forward

具体改动 < 50 行, 修复后必须重跑 Step 2 verify C vs B cosine >= 0.99。

## 7. H5 命中的修复方案

**根因**: [scripts/evaluation/r2r_val_unseen.py](scripts/evaluation/r2r_val_unseen.py) 调 `construct_input` 构造 prompt, 训练用 [src/data/panoramic_tokenized_collator.py](src/data/panoramic_tokenized_collator.py) 构造, 两者 chat template / instruction 文本 / image token 顺序 / system message 可能不一致。

**修复方向** (确认 H5 命中后细化):

- diff 出 A 和 B 路径的 prompt 文本差异
- 改 `construct_input` 输出对齐到 collator 的 prompt format
- 或者在闭环里直接调用 collator 来构造 inputs

修复后必须重跑 Step 2 verify A vs B cosine >= 0.99。

## 8. H3 命中的修复方案

**根因**: [src/models/qwen2_5_vl/integration.py](src/models/qwen2_5_vl/integration.py) commit `17663f9` 的 attention_mask padding 让 generated text tokens 参与了 RoPE 位置计算, 数值上可能扰动了 latent。

**修复方向** (确认 H3 命中后细化):

- 选项 A: 把 generated text tokens 的 attention_mask 改成 0 (不 attend), 看 norm 是否回归正常
- 选项 B: 在 `extract_traj_hidden_states` 里删掉 prompt 后面的 generated text tokens 再加 TRAJ suffix, 只保留 prompt + TRAJ
- 选项 C: 训练侧用同样的 padding pattern 重新提 latent 再训 adapter (代价大, 兜底)

## 9. 终态判定

- 若 Step 1-3 全排除 → adapter latent 质量是真瓶颈, 走"重训 adapter / 增数据 / 调容量"路线 (不在本计划)
- 若命中 H2 / H3 / H5 之一并修复 → **重跑 20 episode adapter eval**, 报告新 SR
- 若命中 H1 → 报告 VLM 头需重训, 不在本排查继续

## 10. 时间预算

- Step 1 + 2: 10 分钟 (高概率定位)
- Step 3: 10 分钟 (再覆盖一小部分)
- 修复 (若命中 H2/H3/H5): 30-90 分钟
- 重跑 20 episode: 70 分钟

**全流程上限**: 4 小时。诊断完, 修复完, 重跑验证完。

## 11. 排查结果（2026-05-25 执行）

### Step 1 — H1：**排除**

- 日志：`logs/habitat_smoke_adapter_20ep/smoke.log`（390 条 `predicted pixel_goal`）
- x 坐标分散（1–255），非塌缩到中心；VLM 输出为合法 `"u v"` 或 `↓`
- y 恒为 202：与 InternNav lookdown / `front_down` 协议一致（图像底行），**不是**退化信号

### Step 2 — H2：**排除**；H5：**弱信号，不修**

命令（`INTERNNAV_MODEL_PATH=/workspace/InternNav_Model`）：

```bash
python -u scripts/evaluation/latent_parity_train_vs_eval.py \
  --config configs/train_config_internnav_8gpu_stage2_wider.yaml \
  --base-checkpoint checkpoints/stage1-s2_latest.pth \
  --stage2-checkpoint checkpoints/stage2_wider_latest.pth \
  --num-samples 20 --output logs/latent_parity_triage_20.jsonl
```

| 路径 | 含义 | cosine vs B |
|------|------|-------------|
| C | 同 collator 输入走 `generate_latents` | **1.0000**（20/20） |
| A | eval 风格 prompt + gold coord 条件 | avg **0.9886**（12/20 &lt; 0.99，min 0.974） |

**VERDICT [C≈B]**：`generate_latents` 与训练 forward 一致（`integration.py` 中 `a970a36` / `17663f9` 修复已生效）。  
A vs B 小差距来自 gold-coord 注入与 collator 序列差 3 token 量级，**不足以解释 38pt SR gap**。

### Step 3 — H3：**不按 padding bug 修**

- 离线 path C：`per_query` norm ≈ **183–235**，`total` ≈ **388–444**
- 闭环 Habitat（adapter 开）：`per_query` ≈ **230–255**，`total` ≈ **444–490**（`logs/habitat_triage_step3_2ep/run.log`，294 次采样）
- 计划中的「teacher 60–80」指 **InternNav 原生 traj_latents**；全景 Stage2 的 `traj_hidden_states` 量级本就更大，且 **C≡B**，padding 未造成 train/eval 分叉

已在 `r2r_val_unseen.py` 将 norm 打印挂在 `_debug_input_trace_enabled` 下（非一次性临时行）。

### 终态结论（Section 9）

| 假设 | 结论 |
|------|------|
| H1 VLM 头 | 排除（闭环坐标未退化） |
| H2 latent 提取 | 排除（C vs B = 1.0） |
| H3 padding 数值 | 排除为 SR 根因（高 norm 但 train=eval） |
| H5 prompt | 弱信号；无单文件 &lt;50 行修复 worth 重跑 20ep |

**SR 对照（同 20ep 列表）**

| 配置 | SR |
|------|-----|
| + adapter | **25%**（`habitat_smoke_adapter_20ep`） |
| 无 adapter | 20% |
| InternNav 原生 baseline | 63%（计划引用） |

Adapter 仅带来约 **+5pt**；pipeline wiring 基本正确，**主要瓶颈是 adapter/bridge 对 InternNav teacher 的逼近度与闭环误差累积**，应走「重训 adapter / 增数据 / 调容量」而非再改 `generate_latents`。

**未执行**：修复后 20ep 重跑（无命中 H2/H3/H5 的单点修复）。

---

## 12. 关键文件

- 只读
  - [scripts/evaluation/r2r_val_unseen.py](scripts/evaluation/r2r_val_unseen.py) L1879-L1962 (panoramic eval generate_latents 注入点)
  - [scripts/evaluation/latent_parity_train_vs_eval.py](scripts/evaluation/latent_parity_train_vs_eval.py) (现成诊断工具)
  - [scripts/evaluation/eval_pano_latent_adapter.py](scripts/evaluation/eval_pano_latent_adapter.py) (offline sanity reference)
  - [src/models/qwen2_5_vl/integration.py](src/models/qwen2_5_vl/integration.py) L1317-L1407 (`extract_traj_hidden_states`)
  - [src/data/panoramic_tokenized_collator.py](src/data/panoramic_tokenized_collator.py) (训练 prompt 来源)
  - [src/models/heatmap/input_constructor.py](src/models/heatmap/input_constructor.py) (eval prompt 来源 `construct_input`)
- 可能改动
  - [src/models/qwen2_5_vl/integration.py](src/models/qwen2_5_vl/integration.py) (H2 / H3)
  - [scripts/evaluation/r2r_val_unseen.py](scripts/evaluation/r2r_val_unseen.py) (H5)
  - [src/models/heatmap/input_constructor.py](src/models/heatmap/input_constructor.py) (H5)