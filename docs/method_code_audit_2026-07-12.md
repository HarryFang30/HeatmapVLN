# HeatmapVLN Method 恢复、代码证据与论文主张审计

> 审计日期：2026-07-12
> 审计口径：以当前仓库的正式全景 Stage 1-S2、teacher-sidecar h1024 Stage 2、adapter-only Stage 3 和 Habitat 闭环评估路径为准。仓库中的 `stage2_bridge_only` / `stage2_wider_bridge` 属于另一套遗留 Stage 2，不纳入本文主方法。
> 证据边界：完成静态调用链、配置和本地 checkpoint 交叉核对；`pytest -q` 为 **120 passed, 1 skipped**。工作区不含真实训练数据、完整外部 InternNav 权重和正式 Stage 3 checkpoint，因此未做端到端 GPU replay，也不把单元测试通过等同于论文主张成立。
>
> **修复更新（同日）：** 本报告最初识别的 Stage 1 image occurrence/anchor 顺序、真实历史 mask、lazy heatmap checkpoint 加载及评估 relative-pose 漏传问题，已在随后代码修复中解决，并新增回归测试；修复后全量结果为 **138 passed, 1 skipped**。报告中对这些问题的机制分析仍用于解释旧实现与旧 checkpoint 风险；“当前仍存在”的表述应理解为修复前审计结论。由于 `stage1_latest.pth` 很可能来自旧 prompt 合同，修复代码不能使旧权重自动获得正确语义，Stage 1 仍需按新布局重新训练后才能支撑论文主张。

## A. 核心结论

### A.1 一句话方法定义

当前代码实现的是：**先用仿真器几何生成的逐历史访问点、逐当前视角投影图监督 Qwen LoRA，再将这些 LoRA 参数用于四视角未来 waypoint 的自回归 SFT，最后训练一个残差 latent adapter——Stage 2 以离线 InternNav 表征与专家轨迹联合约束，Stage 3 仅以专家连续轨迹进一步校正——把冻结的全景 VLM 查询状态接入冻结的 NextDiT 策略。**

这句话刻意没有写“预测热图被转换为全景坐标并注入策略”，因为当前代码中不存在这条 tensor 边。

### A.2 最准确的核心 novelty

最可辩护的 novelty 是一种 **geometry-supervised-to-policy progressive adaptation curriculum**：用历史位置投影辅助任务塑造 VLM LoRA 初始化，随后切换到结构化全景未来 waypoint 生成，再用轻量 latent translator 完成 VLM 表示到既有连续轨迹策略条件空间的迁移。

它的创新中心应写成“分阶段表征迁移与策略接口”，而不是“显式热图在最终策略中持续流动”。后者与正式 Stage 1-S2、Stage 2、Stage 3 的 `heatmap.enable: false` 冲突。

### A.3 最稳妥的论文主张

> Geometry-derived per-visit projection supervision is used to initialize a panoramic waypoint model, while a residual latent adapter aligns waypoint-conditioned VLM query states with a frozen trajectory policy through offline teacher-representation targets and expert trajectory flow supervision.

中文等价表述：几何投影热图是 **LoRA 预训练监督**；全景坐标是 **未来 waypoint 文本目标**；adapter 接收的是 **waypoint-conditioned VLM latent**，而不是热图或显式坐标 tensor。

### A.4 最容易被过度表述的内容

最危险的表述是：

> “最终策略显式读取历史访问热图，并把该热图转换成全景坐标后用于动作预测。”

当前代码直接否定这句话：Stage 1-S2 没有读取热图，坐标标签来自未来真值位姿；Stage 2/3 的 adapter 输入是四个 3584 维 TRAJ query hidden states；最终闭环推理不实例化热图头。

### A.5 会阻断当前论文叙事的实现事实

| 优先级 | 事实 | 对 Method/实验的影响 |
|---|---|---|
| P0 | **不存在 heatmap → panoramic coordinate → adapter 的 tensor 流。** Stage 1 到 Stage 1-S2 只迁移兼容的 LoRA 参数。 | 不能把四阶段写成一个显式空间记忆架构；只能写成训练课程。 |
| P0 | **Stage 1 prompt 与 feature extractor 图像顺序错位。** prompt 先放历史图、后放当前图，但 extractor 固定把前四图当当前图；历史 anchor 又位于对应图片之前。 | 当前 checkpoint 下不能安全声称 decoder 的 spatial canvas 是当前观测，也不能声称 anchor query 看到了自己的历史图。 |
| P1 | **Stage 2/3 用 GT future waypoint teacher-force latent，推理用模型生成 waypoint。** | 存在明确的 gold-coordinate → predicted-coordinate 分布偏移。 |
| P1 | **Stage 1 同时接收由标签同一组 simulator poses 计算的相对位姿。** | 这是强 pose-conditioned shortcut；不应写成纯视觉 correspondence 学习。部署若无可靠 odometry，则该监督含训练时特权信息。 |
| P1 | **闭环 evaluator 默认用真值 `distance_to_goal ≤ 3m` 自动 STOP。** | 默认可称 teacher-free，但不能称完全 oracle-free/privilege-free；正式结果必须报告并建议关闭此项。 |
| P1 | **NextDiT 训练 3 维 `[dx,dy,delta_yaw]`，最终离散解码完全忽略 `delta_yaw`。** | 不能声称最终动作显式使用预测航向；动作是由 XY path tangent 启发式离散化。 |
| P1 | **Stage 1 专项评估/可视化在 lazy heatmap head 构造前加载 checkpoint，且漏传相对位姿。** | 现有专项脚本很可能评估随机 head；不能据此支撑热图质量主张。 |
| P1 | **Stage 1 padding mask 与真实历史长度不一致。** `load_history_frames=false` 产生长度 1 占位，loss 丢弃错误 mask，padding 可能被当负样本。 | 会污染可见性/负样本监督；需修复后再报告热图指标。 |
| P1 | **配置/checkpoint 存在漂移。** Stage 1-S2 同时有 all-layer 与 layers 12–27 两种 checkpoint；当前 Stage 2/3 要求 all-layer exact match；本地无正式 Stage 3 checkpoint。 | 论文必须固定 authoritative config、artifact、commit 和完整启动参数。 |

## B. 术语建议

| 候选名称 | 适用范围与准确性 | 是否暗示时间顺序 | 是否暗示全局地图 | 过度宣传风险 |
|---|---|---|---|---|
| **Pose-Conditioned View-Aligned Visitation Projection (PC-VAVP)** | 最准确描述 Stage 1：相对位姿条件下，对每个历史相机中心生成四视角投影图。 | 低；`per-visit` 只表示有序样本，不暗示时序建模。 | 否。 | 低；明确暴露 pose 条件。 |
| **Per-Visit Panoramic Projection Maps (PVPPM)** | 极保守；准确反映 `[K,4,H,W]`，但“panoramic”实际是四张透视图。 | 否。 | 否。 | 很低。 |
| **View-Aligned Historical-Location Projection (VAHLP)** | 语义直观，突出历史位置到当前 view 的投影目标。 | 否。 | 否。 | 中；当前 prompt/extractor bug 修复前，“aligned”对真实 feature canvas 仍需谨慎。 |
| **Heatmap-Supervised Panoramic Policy Adaptation (HSPA)** | 最适合整个方法：热图只限定为监督/初始化，后续是全景 waypoint 与策略适配。 | 否。 | 否。 | 低到中；必须始终写 `heatmap-supervised`，不能偷换为 `heatmap-conditioned`。 |
| **Progressive Panoramic Waypoint-to-Policy Alignment (PPWPA)** | 最准确描述 Stage 1-S2→Stage 3，但弱化了 Stage 1。 | 否。 | 否。 | 低。 |

**全文统一推荐：Heatmap-Supervised Panoramic Policy Adaptation (HSPA)。**

Stage 1 的表示在内部统一称为 **pose-conditioned per-visit projection maps**。不建议用 `trajectory heatmap`、`global spatial memory` 或 `panoramic visitation coordinate`：主全景热图没有跨历史聚合、时间衰减或跨历史 attention；全景坐标又表示未来 waypoint，不表示访问位置。

## C. 代码证据表

| 论文概念 | 代码中的真实实现 | 文件与函数 | 输入/输出 | 已确认事实 | 仍不确定内容 |
|---|---|---|---|---|---|
| 历史采样 | 从 `[0,t)` 全部历史中用 `linspace` 均匀取最多 K=8，保持升序，不含当前和未来。 | `src/data/sliding_window_dataset.py:825-848,1174-1188` | indices → `history_panoramas [K,4,3,256,256]` | 不是最近 K 步；是对完整已执行历史的稀疏采样。 | 真实数据是否每个方向均有合法图像/pose。 |
| 相对历史位姿 | 当前 robot frame 下 `[dx,dy,cos(yaw),sin(yaw)]`。 | `src/data/trajectory_utils.py:43-116` | poses → `[K,4]` | Stage 1 模型显式消费该 tensor；最终 Stage 3 配置不保留它。 | 部署时是否假设 odometry/pose 可用。 |
| Stage 1 几何标签 | 将每个历史相机中心变换到每个当前相机，pinhole 投影、深度遮挡过滤、Gaussian 渲染。 | `src/data/heatmap_geometry.py:13-42,101-216`; `src/data/sliding_window_dataset.py:965-1010` | poses/depth/K → `[K,4,64,64]`, visibility `[K,4]` | GT 为 `[0,1]` intensity，不是先验概率；主路径每图最多一个峰。 | 数据的深度单位启发式与四向 pose 完整性。 |
| 热图集合语义 | 全景主路径逐历史位置保存，不做 K 维合并。 | `src/data/sliding_window_dataset.py:989-1010` | K 个 visit × 4 views | K 轴顺序保留；无 visit count、时间衰减、跨 visit interaction。 | 无。 |
| Stage 1 predictor | Qwen hooked features + history anchor query + relative pose → coarse `[K,4,8,8]`/visibility → fine `[K,4,64,64]`。 | `src/models/heatmap/heatmap_vln.py:260-387`; `trajectory_attention.py:184-279`; `fine_localization.py:38-137` | RGB/pose/text → heatmaps/visibility | 每个历史点独立经过 258-token Transformer；不是轨迹级 attention。 | 完整 Qwen 权重下的端到端数值尚未 replay。 |
| Stage 1 loss | visibility BCE + visible-view spatial soft-target CE + soft-argmax coordinate distance + invisible-view suppression。 | `src/models/heatmap/heatmap_vln_loss.py:118-183,185-291` | pred/GT maps → scalar | `lambda_kl` 未使用；正样本 Gaussian 峰值经 L1 归一化后幅度衰减被抵消。 | 需修复 batch mask 后才能信任定量指标。 |
| Stage 1 图像错位 | prompt 为 history-first/current-last；extractor 把 image 0..3 当 current，anchor 在其图片之前。 | `src/models/heatmap/input_constructor.py:306-341`; `feature_extractor.py:130-146,185-193,248-299` | image token occurrence order | 静态调用链高置信；git 显示 2026-05-19 commit `b75395d` 将 prompt 改为 history-first，extractor 未同步。 | checkpoint 是否由完全相同 commit 训练，缺少 git hash；文件 mtime 只能提供旁证。 |
| Stage 1 评估协议 | lazy head 构造前加载 state，forward 漏传 `history_rel_poses`。 | `scripts/evaluation/heatmap.py:394-399,272-291`; `scripts/visualization/heatmap.py:186-198,249-283`; `src/models/pipeline.py:688-690` | checkpoint → standalone metric/plot | 与训练输入不一致，head 权重可能 0 匹配。 | 未在完整模型/数据环境执行重现。 |
| Stage 1-S2 输入 | 最多 8×4 历史透视图 + 当前 4 透视图 + 指令，共最多 36 张独立图。 | `src/data/trajectory_dataset.py:799-808,1054-1057`; `panoramic_tokenized_collator.py:468-518` | images/text → Qwen tokens | 不是 equirectangular 或横向拼接 panorama；无 heatmap tensor。 | 训练数据真实 HFOV；eval 固定 HFOV=79°。 |
| Stage 1-S2 标签 | 从 episode 末尾反向搜索最远可投影的**未来** pose，输出 canonical view + local integer `[u,v]`。 | `src/data/pano_view_pixel_goal.py:145-260,269-321`; `trajectory_dataset.py:296-336` | future GT poses → `view`, `[u,v]` | front 优先；side 无 depth occlusion 且限 6m；单一 future waypoint。 | side-view 假可见比例；训练/eval FOV 是否一致。 |
| Stage 1-S2 objective | assistant 文本 `view: d\npixel: u v` 的 causal LM token CE。 | `src/models/heatmap/input_constructor.py:62-99`; `panoramic_tokenized_collator.py:532-605`; `configs/train_system2_panoramic_sft_8gpu.yaml:94-136` | token labels `[B,L]` → LM loss | 无坐标回归头、argmax/soft-argmax、circular loss；只更新 LoRA。 | authoritative LoRA layer range是 0–27 还是 12–27。 |
| Stage 1→Stage 1-S2 | Stage 1 checkpoint 初始化 Stage 1-S2；由于 heatmap disabled，只匹配 LoRA 等现存参数，热图 decoder 不进入模型。 | `scripts/run_stage1_s2_8gpu.sh:31-39,73-74`; `scripts/train.py:781-826` | checkpoint → compatible state | 是 parameter transfer，不是 representation tensor transfer。 | 外部初始 Stage 1 `best.pth` 的来源。 |
| 正式 adapter | 逐 token 两层 residual MLP，`3584→1024→3584`，末层零初始化；之后走 frozen cond projector `3584→768→768`。 | `src/models/adapters/pano_latent_adapter.py:45-92`; `nextdit_action_head.py:83-89` | `[B,4,3584]→[B,4,3584]→[B,4,768]` | 无 geometry/view/pixel 显式输入、无 attention/gate/norm；geometry-aware 类未被正式脚本实例化。 | 无。 |
| Teacher sidecar | 完整 InternNav 教师在 front history/current + front_down + dataset GT coordinate 条件下离线保存 raw/cond latents，另可保存 actions。 | `collect_internnav_teacher_sidecar.py:369-410,684-790,929-1087`; launcher `:440-463` | GT-conditioned context → `[1,4,3584]`, `[1,4,768]`, optional `[32,32,3]` | 正式 collector 不让 teacher 预测 coordinate；`dp_actions` 不被 Stage 2 loss 读取。 | teacher/student resolver 是否选到同一个 future frame，sidecar 数据不在工作区。 |
| Stage 2 student latent | 将 GT structured pano answer 写进上下文、追加 4 个 TRAJ token，冻结 Qwen 提取末尾 hidden states并 detach。 | `train_pano_latent_adapter.py:785-846`; `panoramic_tokenized_collator.py:562-579` | gold-answer context → `[B,4,3584]` | student 也使用 future GT waypoint；不是只有 teacher 拥有 privileged coordinate。 | 无。 |
| Stage 2 objective | raw cosine+norm、cond SmoothL1+cosine、GT trajectory flow MSE；权重 0.1/1.0/0.2。 | `train_pano_latent_adapter.py:1321-1467`; `adapter_pano_stage2_h1024.yaml:32-38` | adapter output/teacher targets/GT trajectory → scalar | 只更新 adapter；冻结模块仍把梯度传回 adapter。不是 KL、logit 或 teacher-action imitation。 | 需要消融验证三项 loss 各自价值。 |
| Stage 3 objective | teacher-force GT structured waypoint，取 frozen Qwen latent；Stage 2 adapter 初始化后只用 GT trajectory flow loss继续训练。 | `configs/train_stage3_pano_system1_h1024_8gpu.yaml:137-225`; `scripts/train.py:602-655`; `train_loop.py:397-452` | `[B,4,3584]`, `[B,N,32,3]` → flow MSE | 唯一更新模块是 adapter；无 teacher sidecar、LM loss、heatmap loss；L2-SP 在 adapter-only 下 inactive。 | 本地无正式 Stage 3 checkpoint，无法确认实际完成训练。 |
| Policy condition | frozen cond projector 4 tokens + front_down visual memory 32 tokens拼成 36×768，再条件化 NextDiT。 | `nextdit_action_head.py:263-365` | `[BN,4,768] + [BN,32,768] → [BN,36,768]` | train sequence N=12 时同一 VLM condition 在不同 visual current frame上重复。 | 无。 |
| 动作目标与解码 | 未来 pose 重采样为 32×`[dx,dy,delta_yaw]`，flow matching；推理只累加 XY，并按 path tangent 转 15° turn/0.25m forward。 | `trajectory_dataset.py:699-771,991-1050`; `trajectory_utils.py:210-265`; `r2r_val_unseen.py:1775-1934` | continuous trajectory → discrete Habitat actions | 第三维 yaw 在最终 decoder 中未使用；局部队列最多执行 4 步。 | 需实验量化 yaw 丢弃影响。 |
| 最终闭环 | 历史/当前四视图 → greedy structured waypoint → latent queries → adapter → frozen System1 → trajectory → discrete actions。 | `r2r_val_unseen.py:2439-2554,2773-2869` | observations/history → action queue | 默认无 teacher、无 heatmap、无真值 pose输入模型；但默认 auto-stop 使用真值 goal distance，首个 local STOP 会强制 LEFT。 | 正式报告实际命令行设置。 |

### Checkpoint 旁证

- `stage1_latest.pth`：`heatmap_lora_64`，epoch 8，289 个 trainable tensors，其中 224 个 LoRA；loss 只有 heatmap 项；checkpoint 配置为 all-layer LoRA。
- `stage1-s2_latest.pth`：all-layer 版本，224 个 LoRA tensors，loss 只有 LM；`checkpoints/stage1-s2_latest.pth`：layers 12–27 版本，128 个 LoRA tensors，loss 同样只有 LM。
- `checkpoints/stage2_adapter_h1024_latest.pth`：`pano_latent_space`，4 个 adapter tensors、7,344,640 参数；保存参数显示实际运行 5 epochs，三项 loss 权重为 0.1/1.0/0.2。
- 工作区未发现与正式 h1024 Stage 3 配置对应的 checkpoint；`stage2_latest.pth` 和 `stage2_wider_latest.pth` 是遗留 bridge 训练，不是 teacher-sidecar adapter。

## D. 四阶段训练表

| 阶段 | 核心目标 | 输入 | 监督信号 | 输出 | 更新参数 | 冻结参数 | 推理时是否存在 |
|---|---|---|---|---|---|---|---|
| Stage 1 | 学习逐历史访问点在四个当前相机中的可见性与像素投影；实际为 pose-conditioned auxiliary task。 | 指令；历史/当前四视图 RGB；`[dx,dy,cos yaw,sin yaw]` | simulator poses、相机内参和当前 depth 几何生成的 `[K,4,64,64]` Gaussian 与 `[K,4]` visibility | per-visit heatmaps/visibility；含热图监督后的 LoRA checkpoint | 正式本地 checkpoint：heatmap decoder + Qwen LoRA | Qwen base；action head | **不在正式最终推理架构中。** 只通过 LoRA 初始化产生间接影响。 |
| Stage 1-S2 | 将共享 LoRA SFT 为结构化全景未来 waypoint 生成器；不是热图到坐标转换。 | 最多 36 张四向透视图 + 指令 | 未来 GT pose 投影得到的 `view + local [u,v]` 文本，及 STOP | panoramic Qwen LoRA；自回归 waypoint text | Qwen LoRA | Qwen base、heatmap、action head、latent queries | **存在。** 最终推理先生成 waypoint，再提取 latent。 |
| Stage 2 | 使 frozen pano-Qwen 的 GT-waypoint-conditioned latent 接近 InternNav native latent/condition，并能条件化冻结策略重建专家轨迹 flow。 | student `[B,4,3584]`；sidecar raw/cond targets；GT `[B,N,32,3]`；front_down images | raw cosine+norm；cond SmoothL1+cosine；GT flow velocity MSE | residual adapter checkpoint | adapter only | student Qwen/LoRA、teacher、cond projector、visual memory、NextDiT | **adapter 存在；teacher sidecar 不存在。** |
| Stage 3 | 用任务级连续轨迹目标进一步校正 Stage 2 adapter。 | GT-waypoint-conditioned frozen student latent；GT continuous trajectories；front_down sequence | trajectory flow-matching MSE only | refined adapter | adapter only | Qwen/LoRA、latent queries、cond projector、visual memory、NextDiT、heatmap | **refined adapter 存在；GT waypoint/trajectory监督不存在。** |

四阶段是 **训练课程**，不是四个并列的最终网络模块。正式最终架构只有：全景 waypoint VLM、latent queries、residual adapter、冻结的 InternNav System1 和闭环动作启发式。

## E. 训练与推理数据流

### E.1 Training Flow

#### Stage 1：几何访问投影监督

1. 对当前时间 `t`，在 `[0,t)` 中均匀采样 `K≤8` 个历史索引；不含当前和未来。
2. 读取：
   - `P_t ∈ R^{B×4×3×256×256}` 当前 `front/right/back/left`；
   - `H_t ∈ R^{B×K×4×3×256×256}` 历史全景；
   - `R_t ∈ R^{B×K×4}`，每项为 `[dx,dy,cosΔψ,sinΔψ]`。
3. 对历史相机中心 `p^w_k` 和每个当前 view 的 C2W `T_{t,v}`：
   - `p^c_{k,v}=T_{t,v}^{-1}[p^w_k;1]`；
   - Habitat camera 为 X right、Y up、−Z forward；
   - pinhole 投影到当前 view，深度 buffer 判断遮挡；视野外、背后、超过 15m 或被遮挡为 invisible；
   - 在 64×64 网格绘制单 Gaussian。
4. GT：`M_t^* ∈ [0,1]^{B×K×4×64×64}`，`s_t^*∈{0,1}^{B×K×4}`。它不是 K 维聚合 map，也不是归一化概率。
5. Qwen 多层 ViT/LLM features 经 DPT；每个历史点的 query、relative-pose token 和 256 个 view-spatial tokens独立进入 coarse Transformer；fine head输出 `M_hat` 和 visibility logits。
6. Loss：`L_vis + L_peak + 0.2L_coord + L_neg`。梯度更新 heatmap decoder，并在实际 LoRA 配置中穿透 frozen base 到 LoRA。
7. **实现警告**：prompt/extractor 顺序使所谓 current spatial features 实际对应第一组历史图；padding mask 也可能把补零历史当负样本。修复并重训前，应把 Stage 1 论文语义写成“监督目标/设计意图”，不能把 current implementation 的成功对齐当事实。

#### Stage 1-S2：未来 waypoint SFT

1. 读取同类四视角历史/当前 RGB 和指令；不读取 `M_hat`、`M^*` 或相对位姿。
2. 从完整未来 GT poses 的末尾向 `t+1` 反向搜索，选最远可投影 future camera center：
   - front 投影有效则优先 front；
   - 否则选 side view 中距图像中心最近者；
   - 输出一个离散 view 和该 256×256 view 内整数 `[u,v]`。
3. target 是 assistant token sequence：`view: d\npixel: u v` 或 `view: stop`。
4. causal LM CE 只监督 assistant tokens；梯度只更新 LoRA。
5. Stage 1 的 heatmap decoder 被丢弃；共享 LoRA 仅作为初始化随后继续被 SFT 修改。

#### Stage 2：离线 teacher-guided latent/condition alignment

1. Student 训练上下文直接写入 GT structured pano waypoint，再追加 4 个 TRAJ queries；frozen Qwen 输出并 detach：
   - `Z_S ∈ R^{B×4×3584}`。
2. Adapter：`Z_A=A_phi(Z_S)∈R^{B×4×3584}`。
3. Frozen cond projector：`C_A=C(Z_A)∈R^{B×4×768}`。
4. 离线 teacher sidecar提供：
   - `Z_T∈R^{B×4×3584}`；
   - `C_T∈R^{B×4×768}`；
   - teacher 也在 dataset GT future coordinate 条件下运行，默认协议为 front history/current + front_down。
5. Dataset 同时给出 `I∈R^{B×N×224×224×3}`、`Y∈R^{B×N×32×3}`、valid mask，`N=12`。
6. Action head将同一 `C_A` 沿 N 重复为 `[BN,4,768]`；每个样本的第一张 front_down 图作为 anchor，与 N 个 current图配对；visual memory输出 `[BN,32,768]`，拼接成 `[BN,36,768]`。
7. `Y_sigma=(1-sigma)Y+sigma epsilon`，target velocity `epsilon-Y`；冻结 NextDiT 预测 velocity。
8. `L_S2=0.1L_raw+1.0L_cond+0.2L_flow`；仅 adapter 参数进入 optimizer。cond projector/NextDiT 虽冻结，autograd仍穿过它们返回 adapter。

#### Stage 3：trajectory-supervised adapter refinement

1. 严格加载 Stage 1-S2 LoRA、InternNav System1 和 Stage 2 h1024 adapter。
2. 与 Stage 2 一样，把 **GT future pano waypoint** 写入 frozen Qwen 上下文，但不构造 LM labels。
3. 获取 `Z_S`，经 trainable adapter、frozen cond projector、visual memory和NextDiT。
4. 只计算 valid-masked flow velocity MSE；heatmap、LM、raw/cond teacher losses为零/不存在。
5. 只有 adapter 更新；配置中的 L2-SP 因没有 trainable System1 参数而返回空 reference，实际不生效。
6. `load_traj_images=true` 的索引主路径只保留 pixel-waypoint states；STOP/turn 不参与该 trajectory adapter 训练。

### E.2 Inference Flow

正式闭环每次高层重规划为：

```text
已执行历史四视图（完整保存，均匀采样至 K） + 当前四视图 + 指令
    → frozen panoramic Qwen greedy generation
    → {STOP | TURN | view + local pixel}
        ├─ STOP/TURN：直接执行 Habitat 离散动作，绕过 adapter/System1
        └─ pixel：将模型生成答案重新条件化进 Qwen + 4 TRAJ queries
                 → Z_hat [1,4,3584]
                 → residual adapter [1,4,3584]
                 → frozen cond projector [1,4,768]
                 + current/anchor front_down visual memory [1,32,768]
                 → frozen NextDiT [num_samples,32,3]
                 → 选择/平均 XY trajectory
                 → path-tangent heuristic
                 → 最多 4 个 Habitat actions
```

- 正常推理不加载 teacher sidecar，也不生成/消费 Stage 1 heatmap。
- 模型路径不接收 GT pose 或 GT future waypoint；历史来自真实执行过的观测，并在每次执行/重规划时追加。
- 但 evaluator 默认读取 simulator `distance_to_goal` 自动 STOP；要做无特权推理必须显式设 `--auto_stop_distance <= 0`，同时确保 `--force_teacher_coord=false`、`--oracle_system2=false`。
- NextDiT 的第三维 `delta_yaw` 不进入最终离散动作；首个 local STOP 被手工改为 LEFT；同一 waypoint latent 可在局部执行中复用，只刷新 front_down visual memory。

## F. 机制级方法概括

### F.1 View-Aligned Visitation Representation

**输入信息。** 历史/当前四视图 RGB、导航指令和每个历史 pose 相对当前 pose 的四维编码。
**信息缺口。** 为“某一历史访问相机中心在当前相机中应出现在哪里”提供明确的几何监督。
**输出。** 有序的 per-visit/per-view heatmaps 与 visibility，而不是单张聚合 visitation map。
**为什么需要。** 它可以作为 VLM LoRA 的空间投影辅助任务，减少完全依赖隐式 latent 自行恢复相对位置的负担。
**如何连接下一部分。** 当前只通过 LoRA checkpoint 初始化连接；没有热图 tensor 连接。
**审计判断。** 标签语义成立；“模型真实对齐到当前图像”因图像顺序 bug 未被当前实现可靠实现。

### F.2 Panoramic Coordinate Adaptation（应改名为 Structured Panoramic Future-Waypoint Prediction）

**输入信息。** 稀疏历史四视图、当前四视图和指令。
**信息缺口。** 给冻结策略上游提供一个显式的未来局部导航意图：哪个 view、该 view 内哪个 pixel。
**输出。** 单个自回归文本 waypoint 或 STOP。
**为什么需要。** 后续 latent queries会在该 waypoint answer 条件下编码计划语境。
**如何连接下一部分。** 生成/GT answer 与四个 TRAJ queries共同产生 adapter 输入。
**审计判断。** 这不是前一空间信号的 panorama conversion；它的目标来自未来轨迹而非历史访问投影。

### F.3 Spatial-to-Policy Adapter（更准确：Waypoint-Conditioned VLM-to-Policy Latent Adapter）

**输入信息。** 四个 3584 维 waypoint-conditioned Qwen latent tokens。
**信息缺口。** Pano-Qwen latent 分布与 InternNav 原生 System1 cond projector/NextDiT 所期望的 latent “方言”不一致。
**输出。** 同 shape 的 residual-translated latent，再由 frozen projector变为四个 768 维 policy condition tokens。
**为什么需要。** 在不改动大模型和既有策略的情况下建立轻量接口。
**如何连接下一部分。** Stage 2 用 teacher raw/cond targets和flow loss训练接口，Stage 3 用任务流损失继续校正。
**审计判断。** adapter 不接收热图、坐标 tensor或显式几何；“spatial-to-policy”只能作为高层抽象，不能作为接口事实。

### F.4 Teacher-Guided Adapter Training

**输入信息。** GT-waypoint-conditioned student latent、离线 InternNav raw/cond latent、专家连续轨迹和 front_down visual memory。
**信息缺口。** 直接让新 latent 接口驱动冻结策略可能存在表示尺度/方向和条件空间错配。
**输出。** 同时接近 teacher latent dialect、teacher projected condition并可支持 GT trajectory flow的 adapter初始化。
**为什么需要。** 提供比单一任务损失更密集的中间约束；但“改善优化”仍需 Stage2-vs-no-Stage2 消融。
**如何连接下一部分。** Stage 2 adapter checkpoint被 Stage 3 strict load。
**审计判断。** 可称 teacher-guided representation alignment；若用 distillation，应限定为 oracle-coordinate-conditioned feature distillation，不是动作模仿或 KL distillation。

### F.5 Action-Supervised Refinement（更准确：Expert-Trajectory Flow Refinement）

**输入信息。** GT-waypoint-conditioned frozen student latent、Stage 2 adapter初始化、专家 pose-derived continuous trajectories。
**信息缺口。** teacher representation similarity不保证 adapter最有利于最终 trajectory generator。
**输出。** 仅由 trajectory flow objective校正后的 adapter。
**为什么需要。** 将优化目标从中间表示切换到冻结策略的任务输出空间。
**如何影响动作。** adapter改变 cond projector输入，从而改变 NextDiT velocity field和采样的 XY trajectory，最终改变离散 action queue。
**审计判断。** 不是离散动作 CE、RL或在线 imitation learning；训练仍是 GT waypoint teacher forcing。

## G. 数学形式化

设当前四视图为 (P_t=\{I_{t,v}\}_{v=1}^4)，历史采样为 (H_t=\{P_{i_k}\}_{k=1}^K)，历史相机中心为 (p^w_{i_k})，当前 view 的 camera-to-world 为 (T_{t,v})。

### G.1 Stage 1 几何目标

对历史点 (k) 和 view (v)：

\[
p^c_{t,k,v}=T_{t,v}^{-1}[p^w_{i_k};1],\quad
u=f_x\frac{x}{-z}+c_x,\quad
v=f_y\frac{-y}{-z}+c_y.
\]

投影在相机前方、图像内、15m 内且不被 depth buffer遮挡时：

\[
M^*_{t,k,v}(q)=\alpha(d_{t,k,v})
\exp\!\left[-\frac{\|q-\mu_{t,k,v}\|_2^2}{2\sigma(z)^2}\right],
\]

否则 (M^*_{t,k,v}=0)，并令 (s^*_{t,k,v}=0)。
**标注：代码直接支持。** 对应 `heatmap_geometry.py:101-216`。这里每个 (k,v) 只有一个峰；不存在 K 维聚合。

模型目标可记为：

\[
(\hat M_t,\hat s_t)=F_\theta(P_t,H_t,R_t,\text{instruction}),
\quad
\hat M_t\in(0,1)^{K\times4\times64\times64}.
\]

**标注：合理抽象。** 模块接口和 shape 有代码支持，但当前 prompt/extractor 错位使实现中的 (P_t) spatial canvas 实际可能是第一组历史图；修复前不能把该式当作已验证行为。

Stage 1 objective：

\[
\mathcal L_{\rm S1}
=\mathcal L_{\rm vis}
+\mathcal L_{\rm peak}
+0.2\mathcal L_{\rm coord}
+\mathcal L_{\rm neg}.
\]

其中 visible view 上：

\[
\mathcal L_{\rm peak}
=-\sum_q \frac{M^*(q)}{\sum_{q'}M^*(q')}
\log \operatorname{softmax}_q(\operatorname{logit}\hat M(q)),
\]

\[
\mathcal L_{\rm coord}
=\left\|\sum_q q\,\operatorname{softmax}_q(\hat M(q)/\tau)
-\arg\max_qM^*(q)\right\|_2.
\]

**标注：代码直接支持。** 当前实现写成 `pred * temperature`；这里 (1/\tau) 只是等价记号，论文应与实际超参数定义保持一致。`lambda_kl` 不进入 loss。

### G.2 Stage 1-S2 全景 future waypoint

令 (Pi_v(p^w_j;T_{t,v})) 表示当前 view (v) 下的可接受投影，代码选择：

\[
j^*=\max\{j>t\mid \exists v,\Pi_v(p^w_j;T_{t,v})\text{ valid}\},
\quad
c_t^*=(v^*,u^*,v^*_{\rm px}).
\]

**标注：代码直接支持。** `max` 是从 episode 末尾反向扫描的时间索引；canonical view 还有 front-first规则。

LoRA SFT 目标为：

\[
\min_{\theta_{\rm LoRA}}
\mathcal L_{\rm S1S2}
=-\sum_{\ell\in\text{assistant}}
\log p_{\theta}(y_\ell^*\mid P_t,H_t,\text{instruction},y_{<\ell}^*).
\]

**标注：代码直接支持。** (y^*) 是 `view + pixel` token序列，不是坐标回归 tensor。

下面这个作者预期等式 **不能** 用于当前论文 Method：

\[
\hat C_t=G_\psi(\hat M_t,P_t).
\]

**标注：需要作者确认且当前代码不支持。** 仓库中没有 (G_\psi) 或 (hat M_t\to C_t) 数据边；若这是论文必要机制，必须先新增实现和实验。

### G.3 Adapter 与 Stage 2

令 (E_\theta) 表示 frozen pano-Qwen 在 GT answer (y_t^*) 后追加四个 query得到的状态：

\[
Z_S=E_\theta(P_t,H_t,y_t^*)\in\mathbb R^{B\times4\times3584}.
\]

Adapter 为：

\[
A_\phi(Z_S)=Z_S+W_2\,\mathrm{Dropout}(\mathrm{GELU}(W_1Z_S+b_1))+b_2.
\]

**标注：代码直接支持。** (W_1:3584\to1024)，(W_2:1024\to3584)，最后一层零初始化。

令 frozen cond projector 为 (C)，teacher raw/cond targets为 (Z_T,C_T)：

\[
\mathcal L_{\rm raw}
=1-\cos(\operatorname{vec}A_\phi(Z_S),\operatorname{vec}Z_T)
+\lambda_n\,\mathbb E_{bq}
\log^2\frac{\|A_\phi(Z_S)_{bq}\|_2}{\|Z_{T,bq}\|_2},
\]

\[
\mathcal L_{\rm cond}
=\operatorname{SmoothL1}(C(A_\phi(Z_S)),C_T)
+\lambda_c[1-\cos(\operatorname{vec}C(A_\phi(Z_S)),\operatorname{vec}C_T)].
\]

**标注：代码直接支持。** MSE 仅记录日志，不进入 `L_cond`。

对 expert trajectory (Y)、噪声 (epsilon) 和 schedule (sigma)：

\[
Y_\sigma=(1-\sigma)Y+\sigma\epsilon,
\qquad v^*=\epsilon-Y,
\]

\[
\mathcal L_{\rm flow}
=\operatorname{MSE}_{\rm valid}
(D_\omega(Y_\sigma,\sigma, C(A_\phi(Z_S)),V),v^*),
\]

其中 (D_\omega) 是 frozen NextDiT，(V) 是 front_down visual memory。
**标注：代码直接支持。**

\[
\min_\phi\mathcal L_{\rm S2}
=0.1\mathcal L_{\rm raw}
+1.0\mathcal L_{\rm cond}
+0.2\mathcal L_{\rm flow}.
\]

**标注：代码直接支持。** 只有 (phi) 更新。

### G.4 Stage 3 与推理

\[
\min_\phi\mathcal L_{\rm S3}=\mathcal L_{\rm flow},
\]

其余权重为零，teacher sidecar不在图中。
**标注：代码直接支持。** 训练时 (Z_S) 仍由 GT waypoint answer条件化。

推理时先生成：

\[
\hat y_t=\operatorname{Generate}_\theta(P_t,H_t,\text{instruction}),
\quad
\hat Z_t=E_\theta(P_t,H_t,\hat y_t),
\]

再采样：

\[
\hat Y_t\sim D_\omega(\,C(A_\phi(\hat Z_t)),V_t\,),
\qquad
\hat a_{t:t+m}=h_{\rm XY\to Habitat}(\hat Y_t[:,:,:2]),\;m\le4.
\]

**标注：代码直接支持。** (h) 忽略第三维 yaw；STOP/turn text 分支会绕过该式。

将整个方法概括为“空间监督逐步迁移到 waypoint 表示和策略 latent”是 **合理抽象**；把它概括为“最终策略显式条件化于预测 visitation map”则是 **当前代码不支持**。

## H. 建议的 Method 章节结构

| 小节 | 核心问题 | 核心公式/模块 | 不应写入的工程细节 | 与下一节的逻辑连接 |
|---|---|---|---|---|
| **3.1 Problem Formulation and Scope** | 隐式历史无法直接指出过去访问位置在当前观测中的像素对应；同时说明最终目标是生成局部连续轨迹。 | (P_t,H_t,M^*_{t,k,v},c_t,Y_t) 的定义；区分访问投影与未来 waypoint。 | 训练脚本名、GPU 数、文件路径。 | 先定义几何辅助监督，再说明它如何作为初始化而非推理输入。 |
| **3.2 Pose-Conditioned Per-Visit Projection Supervision** | 如何用 pose/depth 构造每个历史点在四个当前 views中的目标，并训练 LoRA+decoder。 | 投影公式、Gaussian、visibility与四项 loss。 | lazy module、hook实现、batch padding等工程 bug放在修复/附录，不写成方法。 | 得到 spatially supervised LoRA initialization。 |
| **3.3 Structured Panoramic Future-Waypoint Generation** | 如何让共享 VLM从历史+当前四视图预测策略上游可消费的未来局部意图。 | canonical view + local pixel；LM objective。 | 不写“由热图 argmax得到坐标”；不宣称无缝360°。 | structured answer条件化 TRAJ queries。 |
| **3.4 Residual VLM-to-Policy Latent Interface** | 如何连接 pano-Qwen latent和冻结 InternNav System1 latent space。 | (A_\phi(Z)=Z+\mathrm{MLP}(Z))，cond projector与visual memory。 | 未使用的 geometry-aware adapter类。 | 引出为何需要 teacher表征和轨迹双重监督。 |
| **3.5 Offline Teacher-Guided Representation Alignment** | 如何用 native teacher raw/cond targets缓解 latent mismatch。 | `L_raw`, `L_cond`, Stage 2 combined objective；明确 oracle coordinate条件。 | sidecar JSONL格式、缓存、采集并行。 | teacher目标只作初始化，下一阶段转向最终任务。 |
| **3.6 Expert-Trajectory Adapter Refinement** | 如何只调整 adapter，使其更适合冻结 NextDiT的trajectory flow。 | flow-matching公式、Stage 3 objective、冻结集合。 | optimizer/DDP细节；不要写 action CE/RL。 | 接到真实推理的 generated-coordinate condition。 |
| **3.7 Training–Inference Protocol** | 区分四阶段课程和最终架构；公开 GT waypoint teacher forcing与推理生成的差异。 | 训练参数更新表；最终 inference equation。 | auto-stop属于 Evaluation Protocol，不应隐藏在 Method。 | 为实验消融和公平评测设定边界。 |

如果论文坚持原建议标题 `View-Aligned Visitation Memory / Panoramic Spatial Adaptation / Teacher-Guided Policy Adapter`，至少应在 3.3 开头明确：**panoramic waypoint不是 visitation heatmap的显式变换，而是共享 LoRA 初始化后的独立任务 SFT。**

## I. Method Overview 初稿

现有视觉导航系统通常将过去观测压缩到 recurrent state、history tokens或多模态 hidden states中，因此历史与当前视觉区域之间的空间对应仍需由模型隐式恢复。为提供可监督的对应关系，我们从仿真器位姿构造逐访问点的视角对齐投影目标：对于每个采样的历史相机中心，将其投影到当前时刻的四个方向相机中，并利用当前深度过滤不可见或被遮挡的位置。该目标描述的是“过去访问位置在当前 view 中应出现在哪里”，而不是目标物体、导航目标或全局地图。模型为每个历史位置和每个当前 view预测一个可见性分数及局部空间分布，并以该辅助任务训练热图解码器和 VLM LoRA。

热图监督随后被用作参数初始化，而不是最终策略的显式输入。基于同一组 LoRA 参数，我们对全景 VLM进行结构化 waypoint SFT。模型接收稀疏采样的历史四视图、当前四视图和导航指令，自回归输出一个离散 view标识及该 view内的局部像素坐标。其监督目标由未来专家轨迹中最远的可投影状态生成，因此该输出表示一个未来局部导航意图，而非历史访问坐标。这个阶段把空间监督后的视觉语言表示转向策略上游所需的 view-indexed waypoint表达，但不执行热图到坐标的显式解码。

为复用冻结的 InternNav轨迹策略，我们在全景 VLM的四个 latent query状态与原生 System1条件空间之间加入一个轻量残差映射。该 adapter 保持 token数和隐藏维度不变，并将映射后的状态送入冻结的条件投影器；投影后的四个条件 tokens再与 front-down视觉记忆融合，以条件化冻结的 NextDiT轨迹生成器。Stage 2 使用离线 InternNav sidecar提供的原生 raw latent和投影后 condition作为中间监督，同时通过专家连续轨迹的 flow-matching loss约束映射后的条件是否能驱动既有策略。教师与学生在该阶段都由专家未来 waypoint条件化，因此这一阶段更准确地属于 oracle-conditioned representation alignment，而不是教师动作模仿。

最后，我们移除 teacher-sidecar，并仅利用专家连续轨迹进一步优化 adapter，所有 VLM、视觉记忆和轨迹策略参数均保持冻结。训练时仍将真值 structured waypoint写入 VLM上下文，而推理时先由模型生成 waypoint，再重新提取 latent queries，经 adapter和冻结策略生成局部连续轨迹，并转换为少量离散导航动作。因此，该训练课程从几何访问投影监督过渡到未来 waypoint生成和策略 latent适配；与此同时，预测 waypoint与真值 waypoint之间的条件分布差异构成必须通过消融和闭环实验验证的关键问题。

## J. 两版论文叙事

### J.1 保守版

> HSPA uses geometry-derived per-visit projection supervision to pretrain panoramic VLM LoRA parameters. The same adaptation is then specialized for view-indexed future-waypoint generation. A residual latent translator connects the resulting waypoint-conditioned VLM queries to a frozen trajectory policy, first through offline native-teacher representation targets plus expert trajectory flow supervision and then through trajectory-only refinement. The final policy is teacher-free and heatmap-free at inference; the visitation objective contributes through initialization rather than an explicit runtime memory tensor.

这版完全贴合当前正式数据流。它可以主张“explicit visitation grounding as auxiliary supervision”和“progressive latent-to-policy alignment”，但不能主张最终动作直接读取 projected visitation information。

### J.2 较强版（有条件）

> HSPA establishes an explicit spatial-memory interface that reduces ambiguity between navigation history and current perception, and progressively aligns this interface with future waypoint selection and action generation.

当前代码和实验不足以支撑这版，原因有三：

1. 显式 visitation map 没有进入最终策略；只有 LoRA 初始化可能保留间接效应。
2. Stage 1 当前图像顺序错位，相对 pose又构成强 shortcut。
3. 没有证明性能变化来自 visitation supervision，而非额外训练、LoRA参数、adapter或GT waypoint teacher forcing。

要让较强版成立，最低需要：修复并重训 Stage 1；加入严格无热图预训练的匹配训练基线；验证 Stage 1 投影质量；做 Stage 1→SFT迁移消融；若坚持“interface”一词，则必须真正把预测 map/derived spatial tokens注入 adapter或policy并做 shuffled/noisy/GT-vs-predicted对照。

### J.3 建议实验与代码可行性

| 实验 | 当前可行性 | 实际验证的主张 |
|---|---|---|
| 1. 完整课程 vs 不做热图预训练 | **少量修改/重新训练。** 让 Stage 1-S2 分别从 Stage 1 LoRA和相同 base LoRA初始化；匹配训练步数。 | 热图辅助监督是否给后续 waypoint/policy带来增益。当前“去热图”只能定义为去掉初始化，而不是移除推理输入。 |
| 2. 参数量相同、随机 adapter | **少量配置修改。** 随机初始化同结构 adapter，再做/不做 Stage 3。 | 收益是否只是 7.34M 额外参数。 |
| 3. 真实热图 vs 打乱热图 | **需新增 action 注入管线。** 当前热图不进后续策略。 | 最终动作是否真的依赖空间内容，而非预训练正则化。 |
| 4. 真实热图 vs 模糊/噪声热图 | **需新增 action 注入管线。** 仅做 Stage 1 standalone robustness较容易，但专项评估要先修。 | policy对显式空间质量的敏感性。 |
| 5. GT 热图 vs Stage 1 预测热图 | **需新增 action 注入管线。** | 训练/推理 map gap及上限。 |
| 6. 热图输入 vs 仅全景坐标 | **需新增 action 注入管线。** 当前只有 waypoint-conditioned latent。 | 多峰访问表示是否比单点 future waypoint更有用。 |
| 7. Stage 2+3 vs 仅 Stage 3 | **少量配置修改，强烈建议。** Stage 3从同一 identity/random adapter开始。 | teacher-sidecar是否改善初始化/优化。 |
| 8. Stage 2+3 vs 仅 Stage 2 | **评估器已有 adapter checkpoint加载能力；需要正式 Stage 3 artifact。** | trajectory-only refinement是否提升任务相关性。 |
| 9. 随机 adapter vs teacher-sidecar初始化 | **少量修改。** | Stage 2表示目标是否优于参数量和优化步数本身。 |
| 10. 全景适配前 vs 后 | **需定义匹配基线。** 可用原生/前视 System2、pano SFT但无Stage1初始化、完整pano SFT三组。 | 四视角 structured waypoint是否优于原协议。 |
| 11. 不同历史长度 K | **现有 config/CLI支持。** | 稀疏历史范围与性能/代价关系。需区分均匀全历史和最近K。 |
| 12. 时间衰减 vs 无衰减 | **需修改标签/模型。** 当前 per-visit maps没有聚合时间权重。 | temporal recency是否增加价值。 |
| 13. 全部历史 vs 最近历史 | **少量修改采样器。** 当前默认均匀覆盖完整历史。 | 长程记忆还是近期上下文主导。 |
| 14. 热图定位误差与动作错误相关性 | **当前不可直接做，需建立同一步 map→action配对或注入管线。** | projected visitation quality是否与导航决策相关。 |
| 15. panorama边界附近坐标评估 | **少量新增离线分析脚本。** | view hard boundary、79° gap和token CE是否造成不连续错误。 |
| 16. revisit/loop/backtracking分析 | **少量新增分析。** evaluator已有step recorder和位姿日志。 | 历史监督是否减少重复探索。 |
| 17. 热图质量下降时动作鲁棒性 | **需新增注入管线。** | 显式空间接口的鲁棒性曲线。 |
| 18. GT/oracle waypoint vs predicted waypoint | **现有 evaluator含 `oracle_system2`/`force_teacher_coord`诊断路径。** 正式主结果必须关闭oracle。 | 量化 Stage 2/3 teacher-forcing分布差异。 |
| 19. auto-stop off/on、yaw解码 on/off | **auto-stop已有开关；yaw需少量实现。** | 区分 learned policy、oracle停止和动作后处理的贡献。 |

最优先的四项是：`无 Stage 1 初始化`、`仅 Stage 3`、`Stage 2-only`、`GT vs predicted waypoint`。它们在不新增热图注入架构的前提下，最直接检验当前代码真实可以提出的论文主张。

## K. Claim–Evidence 审计

| 候选论文主张 | 代码是否支持 | 还需什么实验 | 是否存在过度表述风险 |
|---|---|---|---|
| Stage 1 的监督表示历史访问位置在当前 views中的投影 | **标签生成支持；当前模型实现仅部分支持。** 几何目标正确，但 prompt/extractor错位。 | 修复、重训；held-out visibility/peak error；pose-only、RGB-only、shuffle-history消融。 | 高：不能把标签语义直接等同于模型已经学会该语义。 |
| 热图提升空间定位 | **只支持机制可能性，不支持效果。** | 对比无Stage1初始化；可靠Stage1评估；下游 waypoint pixel误差。 | 高。 |
| 热图提升历史感知 | **间接可能，代码无因果证据。** | history shuffle/drop、不同K、no-heatmap pretraining；匹配训练步数。 | 高；pose shortcut可能解释全部收益。 |
| 热图减少重复探索 | **代码不支持效果主张。** 最终推理无热图tensor。 | revisit/loop/backtracking闭环指标；无Stage1初始化对照。 | 极高。 |
| Stage 1-S2 将历史热图适配为全景坐标 | **否。** 标签是未来 waypoint，输入无热图。 | 若必须提出，需新增真实 map-to-pano模块和对应实验。 | 极高，属于事实错误。 |
| 四视角 waypoint提升空间一致性 | **只支持结构，不支持提升。** 四离散views避免连续0/360数值回归，但有hard seam、可能有11° gap且无几何损失。 | front-only对照、seam bucket accuracy、view confusion和像素误差。 | 中高。 |
| Teacher-sidecar改善 adapter初始化 | **代码支持该训练机制，不支持改善结果。** | Stage2+3 vs Stage3-only；random/identity adapter；分解raw/cond/flow loss。 | 中。可称“provides initialization”，不能称“improves”而无消融。 |
| Stage 2 是 privileged-information distillation | **有条件支持。** teacher/student都消费future GT coordinate；teacher还有不同front_down协议。 | 预测坐标训练、same-future-frame对齐检查、去teacher target消融。 | 中高；不应暗示只有teacher有特权或属于动作蒸馏。 |
| Stage 3 提升动作相关性 | **机制支持，效果未证。** objective切换为trajectory flow。 | Stage2-only vs Stage2+3；open-loop ADE/FDE与closed-loop SR/SPL。 | 中。 |
| 整体方法提升动作准确率/导航性能 | **代码路径允许，但无结果证据。** | 完整基线与标准SR/SPL/nDTW等；关闭auto-stop oracle；固定同一评测协议。 | 高。 |
| 最终策略显式条件化于访问热图 | **否。** | 新增推理期热图/spatial token注入。 | 极高。 |
| 最终推理不需要 teacher | **正常路径支持。** | 报告flags并审计日志。 | 低；但不能进一步说默认完全无oracle。 |
| 最终推理完全无 privileged information | **默认不支持。** `auto_stop_distance=3.0`读取真值goal distance。 | 设≤0重跑；关闭teacher/oracle flags；报告anti-deadlock。 | 极高。 |
| 策略学习并执行 yaw-aware 3D action | **训练target有yaw，最终执行不使用。** | 实现yaw-aware decoder并消融。 | 高。 |

## L. 待作者确认的问题

以下问题无法仅靠继续阅读当前代码解决，并会改变 Method 或实验合法性：

1. **论文的 authoritative Stage 1-S2 artifact是哪一个？** 工作区同时有 all-layer 0–27 和 layers 12–27 两种 checkpoint；当前 Stage 1-S2 YAML是 12–27，而正式 Stage 2/3配置要求并 exact-check all-layer。请给出论文实验对应的 config、checkpoint路径和启动命令。
2. **`stage1_latest.pth` 使用的确切 git commit是什么？** checkpoint 文件时间晚于 2026-05-19 的 history-first prompt改动，但 checkpoint不含 git hash。若它确实在当前错位代码上训练，Stage 1需修复和重训后才能作为核心机制证据。
3. **外部 Stage 1 base checkpoint `/home/intern/zhr/fjl/model/best.pth` 的来源是什么？** 本地 Stage 1 checkpoint是从该文件续训，当前仓库无法恢复第一段heatmap-only训练、数据版本和指标。
4. **正式 h1024 Stage 3 是否完成训练，论文结果对应哪个 checkpoint？** 当前工作区没有正式 Stage 3 artifact，无法验证实际 epoch、loss、可训练参数或评估命令。
5. **正式结果的 Habitat 评估参数是什么？** 尤其是 `auto_stop_distance`、`force_teacher_coord`、`oracle_system2`、trajectory selection和是否使用默认首个STOP→LEFT启发式。若 `auto_stop_distance=3.0`，论文必须披露其为oracle停止规则。
6. **真实训练数据的四向 camera intrinsics/FOV、directional pose与depth单位是什么？** 当前数据不在工作区；eval固定79°，而标签fallback为90°，这会决定“360°覆盖”和seam gap表述是否成立。
7. **native teacher sidecar与student pano label是否验证过同一 future frame？** 两边按同一 `(clip,t)` 对齐，却在front_down与四水平views中分别搜索可见future pose；若未验证，论文应称跨协议分布对齐，而非同目标feature matching。
8. **论文要描述当前实现，还是作者原本意图的“显式热图→全景坐标→策略”架构？** 如果后者是不可放弃的核心贡献，当前仓库缺少关键模块和实验，不能通过文字改写补齐；需要先决定是收缩论文主张，还是补实现、重训和新增消融。
