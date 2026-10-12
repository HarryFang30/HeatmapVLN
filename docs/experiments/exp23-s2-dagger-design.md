# EXP-23 设计：用 DAgger 微调 System2，把主模型的 R2R-CE SR 刷上去

判据与结论写在 [`README.md` 的 EXP-23 节](README.md#exp-23-system2-dagger-微调主模型在-r2r-ce-val_unseen-上的-sr-能涨多少)。
这份写**怎么做**：数据契约、四个部件的接口、以及每个部件必须守住的不变量。

## 0. 为什么是这条路（只列决定性的数）

- 主模型 = InternVLA-N1-DualVLN（HF `a698a9e8`，**已经过一轮 DAgger**）+ 本仓库的历史头与 v2 信赖域桥。
  决策（像素目标 / 转向 / STOP）全由 System2 出，桥只改"怎么走"（台账 §1「方法现在站在哪里」①）。
  所以要涨 SR，**只能动 System2 的决策**。
- 4090 上 A0 两种子逐集分解（`scratchpad` 分析，事后、无判据）：
  到了不停（进过 3 m 圈又走出去 O 6.2–7.2%，在圈里转到步数上限 L1 1.9–2.4%）合计 **8.1–9.6pt**；
  自信地停错地方（E2+E3）约 20pt；停早（E1）约 4.8pt；没进过圈就超时（L0）约 4.7pt。
- 同类工作第二轮 DAgger 的增益：JOP-VLN 64.4→68.1（+3.7）。官方第一轮约 +6。**现实预期 +1.5~4pt**。
- EXP-14 的教训：System2 的 LoRA 微调把 normal 状态的逐 token 复现打到 58.4%（像素坐标被改写）。
  已查明至少一个根因：**训练 prompt 不是部署 prompt**（`construct_input_stage2` 113 vs 119 token，多一个句号、
  每张历史图后多 `"\n"`，再加 8 个哨兵）。这一轮的训练输入必须与服务端**逐字节相同**。

用户定（2026-10-12）：不改架构；System2 用 LoRA；**决策用带适配器的权重，交给 System1 的 Z0 用关掉适配器的原权重算**
（System1、PPA 桥、历史头的输入与现在完全一致，不用重训）；算力 = nb2 的 8 张 910B + 4090 空闲卡；
官方数据 InternData-N1 可用（用户自己登录 HF）；两轮；消融不做。

## 1. 数据契约：`heatmapvln-s2-record-v1`（`src/data/s2_records.py`）

一集一个不压缩 tar：`<root>/<scene>/<scene>_<episode:06d>.s2rec.tar` + `.sha256` 提交标记。

| 成员 | 内容 | 不变量 |
|---|---|---|
| `episode.json` | 场景、集号、指令、目标、参考路径、起点、split、种子、服务端身份、`vlm_image_size`、结局指标 | — |
| `steps.jsonl` | 每个基元步之后的真值位姿 `position` / `rotation_wxyz` 和执行的动作 | 不含 lookdown 的俯仰步 |
| `calls.jsonl` | 每次 System2 调用一行：`call_index`、`step_id`（当前帧的采集步）、`history_step_ids`（提示里历史帧的采集步，**按提示顺序**）、服务端回的 `llm_output` / `native_first_output` / `native_lookdown_turns` / `kind` / `pixel_goal` / `actions` / `terminal`，以及调用时的真值位姿和 `distance_to_goal` | 每次调用都记，**不按 kind 过滤**（旧采集器只留 `trajectory`，台账 §5 第 22 条） |
| `front/{step:05d}.jpg` | 该步前视图**发到线上的原字节** | 同一帧每次重发字节必须相同（写入器校验） |
| `lookdown/{call:04d}.jpg` | 该次调用 640×480 lookdown **原字节** | — |
| `lookdown_depth/{call:04d}.png` | 同一 lookdown 的米制深度，uint16 毫米 | 深度传感器 `NORMALIZE_DEPTH=True`、`MAX_DEPTH=10`，存之前乘回 10 |

标签不进 tar：标注器写 `<label_root>/<scene>/<scene>_<episode:06d>.s2labels.jsonl`（`heatmapvln-s2-labels-v1`），可以反复重算。

相机（RGB 与深度共用，`r2r_val_unseen.py` 的 Habitat 配置）：640×480，HFOV 79°，高 1.25 m；
lookdown = 两次 15° 的 `LOOKDOWN`，俯 30°。转向 15°/次，前进 0.25 m/次。

## 2. 四个部件

### A 采集器（4090 客户端）

- `r2r_val_unseen.py` 加 `--s2_record_dir`：只记录，**不改变任何动作**（开关前后同种子逐动作相同，有测试）。
- 每次 System2 调用：用 `_rpc_blob_from_pil` 产出的同一份字节写 `front/`、`lookdown/`；
  `capture_lookdown_view` 在同一个 `env.step` 的观测里顺手取深度。
- 新启动脚本 `scripts/run_s2_dagger_collect_cuda.sh`（从 `run_ppa_r2r_val_unseen_cuda.sh` 派生）：
  `--dataset_split train`、train 分片、本地 CUDA 服务端或 910B 外部服务端（`PPA_EVAL_EXTERNAL_SERVERS=1`）两种都支持，
  500 步上限与评测相同（C500 时代的 120 步上限会截掉正好要的徘徊状态）。
- train 分片：从 `R2R_VLNCE_v1-3_preprocessed/train/train.json.gz` 按路线不相交、打乱，写成评测同格式的 cohort。

### B 标注器（4090，Habitat 寻路，不渲染）

纯几何在 `src/data/s2_oracle_labels.py`，Habitat 部分在 `scripts/tools/label_s2_records.py`。每次调用：

1. `d = geodesic(当前位置, 目标)`。
2. **oracle 路径**：参考路径航点 `w_0..w_n`；`w_k` 记为"已到"当此前任一步的位置与它的测地距离 ≤ 1 m；
   `k* = argmin_{k > 最后一个已到} [geo(p, w_k) + 参考路径从 w_k 到终点的长度]`；
   oracle 路径 = `find_path(p, w_k*)` ⊕ 之后各段的 `find_path`。
3. 标签（只取一种）：
   - `d ≤ R_stop`（默认 2.0 m）→ `STOP`。
   - 否则把 oracle 路径按 0.1 m 采样、投影进 lookdown 相机，用存下的深度剔除遮挡（容差 0.3 m），
     取沿路径**最远**的可见点（且沿路径 ≥ 0.75 m，对应 InternNav 跳过 3 步内目标）→ `↓` + `"v u"`
     （行、列；格式以录到的 native 输出为准）。
   - 一个可见点都没有 → 朝路径上第一个 ≥ 0.5 m 处的点转：`k = clamp(round(|Δyaw|/15°), 1, 4)` 个 `←` 或 `→`；
     `|Δyaw| < 7.5°` 且不可见 → 这个状态不出标签（`skip`）。
4. **一致规则**（防漂移的锚）：模型自己的答案"够好"就保留原文本当目标（自蒸馏），否则换 oracle 文本：
   - 模型 STOP：`d ≤ 3 m` 一致；`d > 3 m` 不一致。
   - 模型不停而 `d ≤ R_stop`：不一致 → STOP。
   - 模型像素：用深度反投影成地面点 q；q 到 oracle 路径 ≤ 1 m 且沿路径前进 ≥ 0.5 m → 一致。
   - 模型转向、oracle 也是转向且同向 → 一致；oracle 是像素（路就在眼前还在转）→ 不一致。
5. 校验：投影约定对照 `scripts/exp19/gt.py`；拿到 InternData-N1 r2r 的官方像素目标标注后，在同样的位姿上逐条比对。

### C 训练器（910B）

- `src/data/s2_sft_dataset.py` + `scripts/train_system2_lora.py` + `scripts/ascend/run_s2_lora_train_npu.sh`。
- 输入构造**只用部署代码**：`build_native_messages` / `append_native_lookdown_turn`（`native_internnav_exact.py`），
  图像解码用服务端同一函数（`decode_jpeg_to_rgb` → PIL → `resize(vlm_image_size)`，lookdown 不缩放），
  `processor.apply_chat_template`。监督只落在助手回合的内容和 `<|im_end|>` 上。
  每个样本的提示前缀必须与推理时 `add_generation_prompt=True` 的 token 逐个相同（运行期断言）。
- 模型按服务端同一路径加载（`InternVLAN1ForCausalLM` → `Qwen2_5_VLForConditionalGeneration` 的映射），
  PEFT LoRA 只挂语言模型：r=32、α=64、dropout 0.05，q/k/v/o/gate/up/down，28 层；视觉塔与 System1 冻结。
- torchrun；NPU 用 hccl（无 NPU 时 cuda/nccl，再无则 cpu/gloo）；bf16 底座 + fp32 LoRA；
  非重入梯度检查点；AdamW lr 5e-5、warmup 3%、cosine；全局 batch 64；
  采样权重：不一致 ×3、一致 ×1；每集最多 48 次调用（防转圈集占满）；留出场景不进训练。
- `--parity-check N`：关适配器，按服务端的贪心解码重放 N 次调用，与录到的输出比对，报逐次完全一致率（门 G0）。
- 输出 PEFT 目录（`adapter_config.json` + `adapter_model.safetensors`）。

### D 服务端（910B 与 4090 都要）

- `rpc_model_server.py` 加 `--system2_lora_adapter DIR`：native 两回合生成**开**适配器；`generate_latents` 在
  `adapters_disabled()` 里算。不开这个参数时逐字节同现在。
- 响应和服务端身份带适配器 sha256，客户端汇总里能看出这次评测用的是哪个适配器。
- 启动脚本透传：`scripts/ascend/run_ppa_servers_npu.sh`、`scripts/deploy/start_nav_servers_cuda.sh`、
  `scripts/run_ppa_r2r_val_unseen_cuda.sh` 都认 `PPA_EVAL_SYSTEM2_LORA`。
- 测试：B=0 的适配器（PEFT 初始化）与无适配器逐字节相同；`generate_latents` 确实在关适配器的上下文里。

## 3. 门（进闭环之前）

在留出场景（train 的 61 个场景里按场景哈希留 6 个）上：

| 门 | 量 | 线 |
|---|---|---|
| G0 解码复现 | 关适配器重放录到的调用，逐次完全一致率 | ≥ 0.98（低于此说明训练输入没对齐，停下查） |
| G1 保持 | 一致状态上，开适配器的输出与模型原输出完全一致率 | ≥ 0.90 |
| G2 误停 | `d > 3 m` 的调用里预测 STOP 的比例 | ≤ 0.002（约 31 次调用/集 → 集级 ≤ 6%） |
| G3 停召回 | `d ≤ R_stop` 的调用里预测 STOP 的比例 | 只报数 |
| G4 纠正 | 不一致状态上预测与 oracle 同类、像素反投影落在 oracle 路径 1 m 内 | 只报数 |

G0/G1/G2 任一不过就不进闭环。

## 4. 存储：910B 只放模型，数据全在 4090（用户 2026-10-12 定）

| 机器 | 放什么 |
|---|---|
| 910B `zhr_1` | 底座权重、环境、AMB3R（已有）；LoRA 适配器与日志（每个 < 1.5 GB）；**训练期的流式环形缓冲，硬上限 10 GB，读完即删** |
| 4090 `/workspace/exp23` | DAgger 记录与标签、InternData-N1 回放数据、打包好的训练流 |

- **训练流**：4090 上按最终的训练顺序（混合比例、采样权重、每集上限、留出场景都在这一步做完，固定种子全局打乱）
  把样本打成分片，每个样本自带它需要的全部字节（历史帧、当前帧、lookdown 的 JPEG 原字节与目标文本）。
  4090 经隧道把分片推进 910B 的环形缓冲；训练器按序读、各 rank 读完后删。训练每秒只吃约 1–2 MB，一条隧道（实测 4.3 MB/s）够用。
- **隧道方向**：910B 不能主动连 4090；所有连接都由容器经既有的 `-L` 隧道发起（nb2 用 52410–52417 / 52510–52517），
  数据双向都走这些连接。只在这些端口没有服务端占用时（采集与训练不同时进行）使用。
- InternData-N1 r2r 在 910B 上下了 27 个场景（剪到只剩前视与 30° RGB），回放够用（按 InternNav 规则约 20 万样本，
  我们最多混约 10 万），搬到 4090 后从 910B 删除；不再继续下载。4090 访问 HF 只能走 hf-mirror（实测单连接约 3.9 MB/s）。

## 5. 闭环与流程

一轮 = 采集（910B 8 槽 + 4090 空卡本地栈，约 170 集/小时）→ 标注 → 训练 → 门 → 4090 上 subset500 金丝雀 →
4090 全量两种子，与 EXP-20 的 A0 同平台逐集配对。第二轮用第一轮的模型重新采集。

数据在两台机器之间只走隧道（本机中转实测 < 0.7 MB/s，不可用）。910B 上一切只写 `~/work/zhr/zhr_1` 之内。
