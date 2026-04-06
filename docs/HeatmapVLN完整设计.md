# HeatmapVLN 当前实现说明

本文件描述仓库里**当前真实运行的 HeatmapVLN 主链路**，并与 `assets/achitecture.svg` 保持一致。

适用范围：

- 默认参考配置：`configs/train_config_internnav.yaml`
- 默认 backbone：`models/internnav_backbone`
- 默认热力图实现：`TrajectoryGuidedAttention + spatial_out + FineLocalization`

这份文档不再保留旧版提案中的 `FiLM`、`query_proj` 精定位头、或“主干完全冻结且无 LoRA”的描述。

---

## 1. 当前默认配置

| 项目 | 当前默认值 | 说明 |
| --- | --- | --- |
| 主配置 | `configs/train_config_internnav.yaml` | 当前推荐默认训练路径 |
| backbone 类型 | `qwen2_5_vl` | 来自 `models/internnav_backbone/config.json` |
| ViT hidden dim | `1280` | `vision_config.hidden_size` |
| ViT depth | `32` | `vision_config.depth` |
| LLM hidden dim | `3584` | `hidden_size` |
| LLM depth | `28` | `num_hidden_layers` |
| LoRA | `enabled` | 默认配置里开启 |
| ViT hook 层 | `[7, 15, 23, 31]` | 4 层 ViT 特征做 DPT-Lite 融合 |
| LLM hook 层 | `[6, 13, 20]` | 3 层 LLM 特征做 DPT-Lite 融合 |
| `history_query` 来源 | `max(llm_layer_indices)` | 默认即第 `20` 层 hook hidden state |
| trajectory attention | `enabled` | 默认用 `TrajectoryGuidedAttention` |
| `c_fused` / `d_attn` | `256` | DPT 融合与注意力统一维度 |
| heatmap 输出 | `[N, 4, 64, 64]` | 每个历史位置、4 个视角各一张热力图 |

补充说明：

- `Qwen2.5-VL (frozen + LoRA)` 的含义是：**基础权重冻结**，但默认配置会训练一部分 LoRA 参数。
- 如果手动关闭 `model.heatmap.trajectory.enable`，代码会回退到旧的 `CoarseLocalization`；但**当前默认真实主链路是开启状态**。

---

## 2. 输入与输出

### 2.1 输入

模型热力图分支当前实际使用的输入包括：

- 当前全景 `current_views`：`[4, 3, 256, 256]`
- 历史全景 `history_panoramas`：`[N, 4, 3, 256, 256]`
- 文本指令 `instruction`
- 历史相对位姿 `history_rel_poses`：`[N, 4]`

其中 `history_rel_poses` 的 4 维定义为：

- `dx`
- `dy`
- `cos_yaw`
- `sin_yaw`

这一路径由 `src/data/vln_sliding_window_dataset.py` 生成，并在 collator / train loop / pipeline 中一路传到 `TrajectoryGuidedAttention`。

### 2.2 输出

默认热力图分支输出：

- `visibility`: `[N, 4]`
- `heatmaps`: `[N, 4, 64, 64]`

在 `eval` / `inference` 阶段还会额外给出：

- `heatmaps_gated`: 对 `heatmaps` 做空间 softmax 后，再乘 `sigmoid(visibility)` 的结果

---

## 3. 当前真实数据链路

当前仓库里，热力图训练主链路不是“主线程现组 prompt 再现 tokenizer”，而是：

1. `VLNSlidingWindowDataset` 产出：
   - `current_views`
   - `history_panoramas`
   - `heatmap`
   - `gt_visibility`
   - `history_rel_poses`
2. `PanoramicTokenizedCollator` 在 DataLoader worker 中：
   - 调用 `construct_input()`
   - 调用 `processor.apply_chat_template(...)`
   - 产出 `pano_inputs`
   - 记录 `pano_num_histories`
   - 记录 `pano_text_anchor_positions`
3. `VLNPipeline.forward()` 直接消费 tokenized 的 `pano_inputs`
4. `Qwen3_5Integration` 做一次共享的多模态前向
5. `HeatmapVLN.decode_from_inputs_batch()` 立刻消费 hook 到的中间特征

这意味着当前真实实现是：

- **单次 Qwen 多模态前向**
- **同一批 token 上直接完成热力图 decode**

而不是：

- 先单独生成文本
- 再起另一条热力图分支

---

## 4. 当前真实模型结构

### 4.1 Backbone

当前默认 backbone 是 `Qwen2.5-VL`，其规格来自：

- `models/internnav_backbone/config.json`

关键配置：

- ViT hidden dim: `1280`
- ViT depth: `32`
- LLM hidden dim: `3584`
- LLM depth: `28`

训练状态：

- backbone 基础权重冻结
- 默认配置启用 LoRA

因此图中的主干应理解为：

- **base frozen**
- **LoRA trainable**

### 4.2 Prompt 组织

Prompt 由 `src/models/heatmap/input_constructor.py` 组织为：

1. 场景说明文本
2. 可选导航指令
3. 当前全景 4 张图
4. 每个历史位置对应的一段中文锚点文本
5. 该历史位置对应的 4 张图
6. 任务文本

历史位置不是靠额外 ID embedding 标记，而是靠这段中文锚点文本分组。

### 4.3 Hook 特征提取

`FeatureExtractor` 当前会提取三类特征：

- ViT 多层特征：`16x16`
- LLM 多层视觉特征：`8x8`
- 文本锚点 token 的 hidden state

这里最重要的一点是：

- `history_query` 取自 **最深的被 hook 到的 LLM 层**
- 不是“整个 LLM 的最终层 hidden state”

默认配置下：

- `llm_layer_indices = [6, 13, 20]`
- 因此 `history_query` 实际来自 **第 20 层**

### 4.4 DPT-Lite 融合

当前实现有两套 `DPTLiteFusion`：

- `vit_dpt_fusion`: 4 层 ViT 特征 -> `[4, 16, 16, 256]`
- `llm_dpt_fusion`: 3 层 LLM 特征 -> `[4, 8, 8, 256]`

这两条支路都是真实代码路径，不是文档草图。

### 4.5 Coarse：TrajectoryGuidedAttention

默认 coarse 头不是旧版 `CoarseLocalization`，而是：

- `src/models/heatmap/trajectory_attention.py`

输入由三部分组成：

- `history_query`: `[N, 3584]`
- `history_rel_poses`: `[N, 4]`
- `fused_llm`: `[4, 8, 8, 256]`

具体计算为：

1. `history_query` 线性投影：`3584 -> 256`
2. `history_rel_poses` 做正弦位置编码：`4 -> 132`
3. 再投影：`132 -> 256`
4. 当前 4 视角 LLM 空间特征展平：`[4, 8, 8, 256] -> [256, 256]`
5. 构造 token：
   - `hist_token(1)`
   - `traj_token(1)`
   - `spatial_tokens(256)`
6. 加可学习位置编码 `pos_embed`
7. 过 `TransformerEncoder`

默认配置下：

- `num_heads = 4`
- `num_layers = 2`
- `dim_feedforward = 1024`
- `activation = GELU`

输出为：

- `visibility`: `[N, 4]`
- `coarse_heatmap`: `[N, 4, 8, 8]`
- `spatial_out`: `[N, 256, 256]`

这里的 `spatial_out` 是当前精定位分支的关键输入。

### 4.6 Fine：spatial_out 条件化精定位

当前 `FineLocalization` 已经不是旧版 `FiLM + query_proj` 设计。

真实实现位于：

- `src/models/heatmap/fine_localization.py`

当前步骤是：

1. 将 `spatial_out` 从 `8x8` 上采样到 `16x16`
2. 将 `coarse_heatmap` 从 `8x8` 上采样到 `16x16`
3. 对 coarse map 取 `sigmoid`，作为单通道 attention prior
4. 拼接：
   - `vit_fused`: `256 ch`
   - `spatial_out_up`: `256 ch`
   - `attn`: `1 ch`
5. 得到 `513` 通道输入
6. 用两层 `ConvTranspose2d` + 一层 `Conv2d` 解码到 `64x64`

因此当前真实精定位结构是：

- **不使用 FiLM**
- **不使用 fine-stage query_proj**
- **使用 `spatial_out` 做位置相关条件化**

---

## 5. 与 `assets/achitecture.svg` 的对应关系

下表对应的是**当前已同步后的 SVG**：

| SVG 模块 | 当前真实实现 |
| --- | --- |
| `Qwen2.5-VL (frozen + LoRA)` | 基础权重冻结，默认配置启用 LoRA |
| `ViT DPT-Lite` | `DPTLiteFusion(c_vit=1280, c_fused=256, n_layers=4)` |
| `LLM DPT-Lite` | `DPTLiteFusion(c_llm=3584, c_fused=256, n_layers=3)` |
| `history_query [N, 3584]` | 文本锚点 token 在最深 hook 层上的 hidden state |
| `rel_poses [N, 4]` | `dx, dy, cos_yaw, sin_yaw` |
| `sin_pe 4->132` | `num_freqs=16` 的正弦编码 |
| `proj_h 3584->256` | `TrajectoryGuidedAttention.proj_history` |
| `proj_t 132->256` | `TrajectoryGuidedAttention.proj_traj` |
| `token concat + pos_embed [258, 256]` | `hist + traj + 256 spatial tokens + learnable pos` |
| `TransformerEncoder x2` | 默认 `trajectory.num_layers = 2` |
| `vis_head 256->128->4` | 当前真实实现 |
| `hm_head 256->128->1` | 当前真实实现 |
| `spatial_out -> Fine` | 当前精定位主条件信号 |
| `Fine localization (no FiLM, no query_proj)` | 当前真实实现 |
| `concat [ViT_fused, spatial_out, attn]` | `256 + 256 + 1 = 513` 通道 |
| `ConvTranspose decoder` | `16->32->64` 解码 |

---

## 6. 与旧版设计的差异

下面这些内容如果你在仓库旧文档、旧注释或旧讨论里看到，默认都不应再当作当前实现：

| 旧描述 | 当前真实实现 |
| --- | --- |
| `Qwen3.5-9B` 默认主干 | 当前默认是 `Qwen2.5-VL / InternNav backbone` |
| 主干“完全冻结” | 当前默认是“基础权重冻结 + LoRA 可训练” |
| `history_query` 来自最终层 hidden state | 当前来自**最深 hook 层**，默认第 `20` 层 |
| coarse 用 `CoarseLocalization` | 当前默认用 `TrajectoryGuidedAttention` |
| fine 用 `FiLM` | 当前不用 `FiLM` |
| fine 用 `query_proj` | 当前不用 fine-stage `query_proj` |
| fine 输入是 `ViT + coarse + query` | 当前是 `ViT + spatial_out + coarse_attn` |
| tokenization 在主线程做 | 当前默认由 collator 在 worker 中预处理 |

---

## 7. 代码映射

| 功能 | 文件 |
| --- | --- |
| 整体组装 | `src/models/pipeline.py` |
| Qwen 集成 | `src/models/qwen3_5/integration.py` |
| Prompt 构造 | `src/models/heatmap/input_constructor.py` |
| Hook 特征提取 | `src/models/heatmap/feature_extractor.py` |
| ViT / LLM DPT 融合 | `src/models/heatmap/dpt_lite_fusion.py` |
| Trajectory attention coarse | `src/models/heatmap/trajectory_attention.py` |
| Fine localization | `src/models/heatmap/fine_localization.py` |
| Heatmap 总装 | `src/models/heatmap/heatmap_vln.py` |
| 数据集 / 相对位姿 | `src/data/vln_sliding_window_dataset.py` |
| tokenized collator | `src/data/panoramic_tokenized_collator.py` |

---

## 8. 当前应以哪些内容为准

当图、文档、注释与代码冲突时，建议优先级如下：

1. `configs/train_config_internnav.yaml`
2. `src/models/pipeline.py`
3. `src/models/heatmap/*.py`
4. `assets/achitecture.svg`
5. 本文档

如果你后续继续调整架构，建议同时更新：

- `assets/achitecture.svg`
- 本文档
- `src/models/heatmap/heatmap_vln.py` 顶部 docstring
- `src/models/heatmap/feature_extractor.py` 顶部 docstring

