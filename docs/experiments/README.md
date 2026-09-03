# 实验台账

这个仓库后面要跑大量实验。台账的作用不是记结果——结果在各自的报告里——而是记
**为什么跑、判据是什么、结论能支撑到哪一步**。三个月后回头看，最贵的不是重跑，
是想不起来当初的判据，于是把一个模棱两可的结果读成自己想要的样子。

## 0. 使用约定

1. **一个实验一个条目**，编号 `EXP-NN`，编号只增不改，作废的条目标 `✗ 作废` 但保留。
2. **开跑前先写"问题 / 假设 / 判据 / 设置"四段并 commit**。
   判据必须**在看到任何结果之前**写死：什么数值算支持假设、什么数值算否定、
   什么数值算"没测出来"。看到结果再补判据，等于没有判据。
3. 跑完补 **"结果 / 结论 / 边界"**，状态改成 ✅ / ❌ / ⚠️。
   "边界"一段是强制的：这个结论**不能**推广到哪里。
4. 详细证据链另开 `expNN-<名字>-report.md`，台账只留一句话结论 + 链接。
   复现命令另开 `expNN-<名字>-runbook.md`（或网站提交物 `-submission.md`），
   两者都放在本目录下，文件名前缀与编号一致，这样按名字排序就是按编号排序。
   未编号的历史材料放 `legacy/`。
5. **每个数字都要带产物路径**。台账里出现的任何指标，都要能在 §4 的路径下查到原始文件。
6. 集群任务一律走网站提交（形态见 `CLAUDE.md` §1.1），提交物本身进仓库，不要只留在聊天里。

状态图例：✅ 已定论 ｜ ⚠️ 有结论但边界重要 ｜ ❌ 假设被否定 ｜ 🔬 进行中 ｜ ⏳ 待跑

## 1. 总览

| 编号 | 问题 | 状态 | 一句话结论 |
|---|---|---|---|
| [EXP-01](#exp-01-把闭环-sr-拉回-native-水平) | 桥接上线后闭环 SR 从 62.5% 掉到 18.1%，能不能在不改架构、不做部署端衰减的前提下拉回来 | ✅ | 能。SR 62.81% vs native 62.48%，桥生效率 99.6% |
| [EXP-02](#exp-02-历史认知头是不是在用视觉定位历史) | 历史认知头到底是"从视觉定位历史"，还是"把位姿投影出来" | ❌ | 是投影。定位完全由位姿决定，外观只贡献可见性（AUPRC +4~6 点） |
| [EXP-03](#exp-03-部署头本身是否也只看位姿) | 部署权重（而非探针）是否也只看位姿 | ⏳ | — |
| [EXP-04](#exp-04-位姿有噪声时外观是否变重要) | 位姿换成 AMB3R VO（有噪声）后，外观是否变重要 | ⏳ | — |
| [EXP-05](#exp-05-信赖域重训到底值多少-sr) | 信赖域重训本身值多少 SR（把 v1 桥放到修复后的评测栈上） | 🔬 | 2026-09-03 提交，全量 1839 集 |
| [EXP-06](#exp-06-零桥在修复栈上是否等于-native) | 修复后的评测栈跑零桥，是否精确等于 native | ⏳ | 2026-09-03 消融规划暂缓（见条目） |
| [EXP-07](#exp-07-主表第二个种子parity--ne-是否可复现长路径增益是否存在) | 主表 parity + NE 改善在第二个种子上是否复现；长路径（≥10 m）增益是否存在 | 🔬 | 2026-09-03 提交桥臂（种子 1337），native 臂待提 |

---

## 2. 已完成

### EXP-01 把闭环 SR 拉回 native 水平

**问题.** 零初始化交叉注意力桥把历史记忆注入 4 个 plan token 之后，R2R val-unseen
闭环 SR 从原生 System1 的 62.5% 掉到 18.1%。要求：不改架构、不在部署端对 Δ 做衰减。

**假设（当时）.** 无约束的 action loss 把 plan_z 推离冻结 DiT 的条件流形（"桥漂移"）。

**结果.** 假设**只对了一小半**。逐 episode 分解发现 18.1% 是两个病灶叠加，且大头被误诊：

- 病灶 1（小头）：桥确实漂移——Δ 每元素 RMS 漂到 ~0.7，换来的 teacher-forced 速度
  MSE 改善 ≤4%（噪声内），而 best checkpoint 恰好按 teacher-forced MSE 选中漂得最远的。
- 病灶 2（大头）：**v3 评测栈给冻结 System2 喂了错误输入**。41% 的 episode（758 集）
  从未产生一次 pixel goal，中位 13 步内死亡，两轮评测的死亡集合逐集 100% 重合；
  同一批 episode 在 native 复刻臂上 SR 68.3%。相对认证复刻栈有六处逐字节差异
  （lookdown 分辨率、history 图间分隔符、instruction 句号、turn 执行粒度、
  `↑`/混合箭头解析、二轮箭头处理）。

**修复.** ① v2 重训：硬信赖域（逐 token ‖Δ‖ ≤ 5%·‖plan_z0‖，训练/部署同款）+ 相对
delta 惩罚 + advantage 加权 action loss + **采样 rollout 终点误差**作为选点指标；
② 评测栈逐字移植认证复刻栈的 System2 前端，token parity 逐位验证。

**结论.** SR **62.81%**（1155/1839）vs native 62.48%，SPL 55.04% vs 55.23%，
桥生效率 99.6%。死亡集恢复到 67.68%（native 68.34%），存活集 59.39% 反超 native 1.0pt。

**边界.** 两个变量是**同时**改的（桥的训练方式 × 评测栈保真度），所以这条结果
**不能**单独证明信赖域值多少分——那要 EXP-05 补。

**教训.** stage0 的 A/B 只审计了"v3 栈内部零桥前向等价"，从未审计"v3 栈 ≟ 认证复刻栈"。
**冻结模块的输入保真度必须逐字节对齐，不能靠栈内自洽。**

📄 [exp01-sr-recovery-report.md](exp01-sr-recovery-report.md) ｜
🔁 [exp01-sr-recovery-submission.md](exp01-sr-recovery-submission.md)

---

### EXP-02 历史认知头是不是在用视觉定位历史

**问题.** 历史认知头给每个历史观测预测一张四方向热力图。它拿到三样东西：当前帧图像、
每个历史一张图像、每个历史的相对位姿。**位姿几乎已经几何地决定了答案**，所以分数高
本身不能证明它读了图像。论文 §3.3 能不能把"从视觉定位历史"写成主张？

**假设.** 若该头确实在用视觉，则 (a) 去掉位姿后仍应显著优于地板值；
(b) 打乱历史图像应当破坏对应槽位的预测。

**判据.**（本次是结果出来后补写的——见 §5 教训 3）

**设置.** 部署所用的 `internnav_single_view` 头（8.75M 参数），骨干是未改动的 released
InternNav ViT、全程冻结无 LoRA。四种输入配置共享同一种子、同一份逐字节相同的全新头部
初始化、同一批样本、12000 步预算；两个种子（42/1337）；`heatmap_randomwalk_train_v1`，
按场景 MD5 划分 54 训练 : 7 验证场景零重叠；K=8，验证 400 样本（3168 个历史槽位）。
十项匹配性检查由 summarizer 在出表前 fail-closed 校验。

**结果（两种子均值）.**

| 配置 | joint_pck8 | 中位像素误差 | visibility AUPRC |
|---|---|---|---|
| full（真图 + 真位姿） | 0.882 | 2.12 px | 0.893 |
| pose-only（全黑图 + 真位姿） | 0.848 | 2.24 px | 0.845 |
| vision-only（真图 + 常量位姿） | 0.438 | 6.04 px | 0.416 |
| no-input（全黑图 + 常量位姿） | 0.438 | 6.20 px | 0.353 |

对已训练 full 探针的干预：历史图像整体倒序 **0.882→0.882（零影响）**；
位姿错位一格 0.882→0.432；**位姿与标签同步错位一格 →0.879（完全恢复）**。

**结论.** ❌ 假设被否定。**定位完全由相对位姿决定**：某槽位的输出只取决于送进去的位姿，
与该槽位放哪张图无关。没有位姿时，图像不足以定位（vision-only 与 no-input 地板值持平，
且都退化成对全部槽位预测"后"视角的常量预测器）。外观唯一跨种子稳定的贡献是
**可见性/遮挡**（AUPRC +4~6 点）。这与标签构造自洽：标签是位姿与内参的确定性投影，
唯一需要看场景的成分是 0.5 m 深度容差的遮挡判定。

**对论文的影响.** §3.3 **不能**写"从视觉定位历史"；现稿的保守口径（遮挡/可见性需外观
+ 记忆向量 m 注入决策）是正确的且现在有证据。"显式时空认知"的价值落点应放在
**可监督、可注入**，而不是"视觉定位能力"。

**边界.** ① 位姿用的是 **Habitat GT**，捷径的最强形态，部署时是 AMB3R VO（有噪声）
→ EXP-04；② 这是**从零初始化的 head-only 探针**，不是走完 Stage1/2 的部署头
→ EXP-03；③ 数据是 random-walk，不是部署评测的 R2R。四条训练曲线在后 8000 步全部走平，
所以不是训练不足造成的。

📄 [exp02-shortcut-probe-report.md](exp02-shortcut-probe-report.md) ｜
🔁 [exp02-shortcut-probe-runbook.md](exp02-shortcut-probe-runbook.md) ｜
`scripts/run_heatmap_shortcut_diagnostic_8gpu_mxc500.sh`

---

## 3. 待跑队列

条目按"性价比 = 结论价值 / 卡时"排序。**开跑前把判据再读一遍，跑完只填结果，不改判据。**

### EXP-03 部署头本身是否也只看位姿

**问题.** EXP-02 的结论是对**探针**成立的。走完 Stage1/Stage2 的**部署权重**是否同样
只是位姿的函数？这是"归因"问题，EXP-02 回答的是"可识别性"问题，两者不能互相替代。

**假设.** 部署头与探针行为一致——长训练不会凭空长出视觉定位能力，因为标签本身不要求。

**判据（预注册）.**
- **支持假设**：部署头在 `pose-conflict-shifted-target` 下恢复到基准的 ≥95%，
  且 `history-shuffle` 相对基准变化在 ±2pt 以内。→ §3.3 保守口径确立，可以写进论文。
- **否定假设**：上述两项中任一明显退化（shifted-target 恢复 <85%，或 history-shuffle
  掉 >5pt）。→ 说明长训练确实用上了外观，措辞可以放宽，但要给出具体是哪一项在用。
- **测不出来**：两个指标落在中间区间 → 说明干预强度不合适，换更强的干预（单槽位替换）再来。

**设置.** 纯评测，无训练：`diagnose_heatmap_shortcuts.py --architecture internnav_single_view
--mode full --head-checkpoint <部署头> --standard-only 关闭`，跑全部六项干预。
需要先把部署 checkpoint 里的历史头张量抽成 head-only 格式（`head_state_dict` +
`initial_head_hash`）。**代价：1 卡、约 1 小时**（400 样本 × 7 条件）。

**注意.** 部署头是在 R2R + AMB3R 位姿上训的，若在 random-walk + GT 位姿上评测，
分布偏移会混进来。要么换 R2R 验证集评测，要么明确写清这是跨分布干预。

---

### EXP-04 位姿有噪声时外观是否变重要

**问题.** EXP-02 用的是 GT 位姿——捷径的最强形态。部署时位姿来自 AMB3R VO，有噪声。
噪声位姿下，外观是否终于变得重要？

**假设.** 是。位姿越不可靠，头越需要用外观纠正/补偿。

**判据（预注册）.**
- **支持**：VO 位姿下 `full − pose-only` 的 joint_pck8 差距在**两个种子上都 > 3pt**
  （GT 位姿下是 +6.8 / +0.1，不稳定）。→ 可以写"外观在位姿不可靠时承担纠错"。
- **否定**：差距仍不稳定或 < 3pt。→ EXP-02 的结论原样成立，不必加限定语。

**设置.** 同 EXP-02，但数据换成带 AMB3R 缓存的配置
（`amb3r_pose_cache_root` + `require_amb3r_pose_cache: true`），
数据根用 `r2r_panoramic_data_v2/train` + `data/amb3r_endpoint_v3_full_r2r`。
**代价：8 卡 × 约 5 小时**（4 模式 × 2 种子）。
注意 AMB3R 缓存模式下 `single_view_rgb_input` 必须为 true，且有效帧由缓存决定。

---

### EXP-05 信赖域重训到底值多少 SR

**问题.** EXP-01 同时改了两个变量。"没有信赖域 → 18.1%"是**错误**的读法，因为 18.1%
主要是评测栈缺陷造成的。信赖域本身值多少分，需要单独测。

**思路：把它当成一个 2×2 矩阵，现在已经有三格。**

| | 缺陷 v3 栈 | 修复栈（逐字节对齐） |
|---|---|---|
| v1 无约束桥 | **18.1%**（全量，已有） | ❓ 唯一缺格 = 本实验 |
| v2 信赖域桥 | **13.6%**（940 集部分数据，已有） | **62.8%**（全量，已有） |

矩阵能分解出两条独立的消融轴：
- **评测协议保真度**（固定 v2 桥）：13.6% → 62.8%，−49pt 全部来自六处输入差异。
  这是全文最有普适价值的结论。缺陷栈下 v1 桥 18.1% ≈ v2 桥 13.6%，说明**桥怎么训在
  死锁面前几乎无差别**，栈效应主导。
- **桥的训练方式**（固定修复栈）：62.8% vs ❓ ← 本实验补的就是这一格。

**假设.** v1 的漂移桥（Δ RMS ~0.7）在干净评测栈上仍会明显伤 SR。

**判据（预注册）.**
- **支持**：v1 桥在修复栈上 SR 比 62.8% 低 **> 3pt**。→ 可以写"约束集是必要的，
  信赖域挽回了 X 个点"。
- **否定**：SR 落在 62.8% ± 1.5pt。→ **信赖域的卖点必须改写**成"可控性与防御性设计"，
  不能声称它挽救了 SR。这个结果同样要如实写。
- **没测出来**（2026-09-03 开跑前补的第三档）：SR 比 62.8% 低 1.5–3pt。→ 方向支持但幅度
  在单种子噪声边缘，只能写"无约束注入有害但幅度有限"，不得写"挽回了 X 个点"；
  要写具体分数必须再补一个种子。
- 判读一律用同 episode 配对 bootstrap（`scripts/tools/paired_closed_loop_bootstrap.py`，
  分别对 native 与 v2 终局配对），不用裸均值差。

**设置.** 一次网站提交，复用 `scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh`
（修复栈，与 EXP-01 终局评测同一套；协议种子 42），只换四个变量：
`PPA_EVAL_CHECKPOINT` → `model/output_past_plan_action_action_refine_v1_8gpu/run_20260818_225001/checkpoints/best_deployment_full.pth`
（自足部署文件：79+11+10 张量，v1 评测当年加载的就是它；`best.pth` 缺 11 个冻结的未来头张量，评测端预检会拒绝。
桥 out_proj 权重 RMS 0.0032，是 v2 的 5 倍）；`PPA_EVAL_CONFIG` →
`configs/ppa_action_refine_8gpu.yaml`（与该 run 的 manifest 在模型字段上逐项相同：
无 `max_delta_ratio` ⇒ 部署端不截断，preserve 0.5 / delta 0.01 绝对，teacher-forced 选点）；
`PPA_EVAL_OUTPUT_ROOT` → `model/eval_ppa_refine_v1_unconstrained_nativefix_r2r_val_unseen_8gpu`；
`PPA_EVAL_ARM` → `ppa_refine_v1_unconstrained_online_amb3r`。全量 1839 集，不用子集。
**代价：8 卡 × 约 18 小时**（EXP-01 终局实测 07:54 → 次日 01:55）。
提交物：[exp05-v1-bridge-fixed-stack-submission.md](exp05-v1-bridge-fixed-stack-submission.md)。

---

### EXP-06 零桥在修复栈上是否等于 native

**问题.** 修复后的评测栈跑**零桥**（桥不生效），是否精确回到 native 的 62.5%？

**为什么值得跑.** 两个用途：① 给 EXP-05 的桥消融轴补上"零点"，构成
零桥 / 漂移桥 / 信赖域桥三点对照；② 这是"修复栈 ≡ 认证复刻栈"这一命题的**闭环终极证明**。
EXP-01 的栈修复只验证到 token parity（前向逐位相等），没有验证到闭环 SR 相等。

**判据（预注册）.**
- **支持**：SR 落在 62.48% ± 1.5pt。→ 修复栈与认证复刻栈闭环等价，
  所有基于该栈的结论成立。
- **否定**：显著偏离。→ **修复栈仍与认证复刻栈不等价，EXP-01 之后所有基于该栈的
  结论都要重新审视**，优先级立刻升到最高。

**设置.** 同 EXP-05 的评测提交，用 stage0 零桥 arm 开关（`--ppa_stage0_action_arm baseline`，
启动脚本尚未透传，且脚本末尾的"PPA 至少生效一次"检查要绕过）。**代价：8 卡 × 约 18 小时。**

**状态（2026-09-03）.** 消融规划里决定**暂缓**：stage0 treatment 臂已在代码里逐张量验证
`plan_z == plan_z0`，EXP-01 的死亡集也已恢复到 native 水位（67.68% vs 68.34%），
这 18 小时目前买不到论文里的新主张。全文对照只用一个定义（认证复刻栈的 native），
EXP-07 的第二个种子也按此跑。若 EXP-05 或 EXP-07 出现无法用桥解释的偏差，本条立刻升为最高优先级。

---

### EXP-07 主表第二个种子：parity + NE 是否可复现，长路径增益是否存在

**问题.** 主表（EXP-01 终局 vs native）只有协议种子 42 这一份配对样本。同 1839 集、同种子的
配对 bootstrap（2000 次重采样）给出：SR +0.33 [−1.25, +2.01]、SPL −0.19 [−1.50, +1.17]、
OS +1.14 [−0.16, +2.34]、**NE −0.16 m [−0.30, −0.02]**；预先取整的长路径分层
（测地距离 ≥ 10 m，546 集）：SR +1.65 [−1.10, +4.58]、SPL +0.89、OS +1.28、
**NE −0.33 m [−0.69, −0.01]**；其余分层 ≈0（8.4–10.4 m 一段甚至 SR −1.96）。
这些是单种子、其中长路径是事后看到的分层——都还不能写成主张。
（产物：`model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/analysis/paired_vs_native_seed42.json`，
脚本 `scripts/tools/paired_closed_loop_bootstrap.py`。）

**假设.** H1（主表）：SR/SPL 与 native 持平、NE 改善在第二个种子上复现。
H2（长路径）：桥的增益集中在测地距离 ≥ 10 m 的 episode。

**判据（预注册，2026-09-03，在种子 1337 出任何结果之前写死；阈值固定 10 m，不再换）.**
- **H1 支持**：两种子合并（2×1839 配对）SR 与 SPL 的 95% CI 都包含 0 且下界 > −1.5pt，
  且 NE 的 95% CI 上界 < 0。→ 主表写"SR/SPL 持平、NE 显著改善"。
- **H1 否定**：SR 或 SPL 的合并 CI 上界 < −1.5pt（桥有害）→ "不劣于 native"不成立，如实写；
  或 NE 合并 CI 包含 0 且点估计 > −0.05 m → 删除 NE 主张。
- **H1 没测出来**：NE 合并 CI 包含 0 但点估计 ≤ −0.10 m，或两种子 NE 方向相反
  → 只写趋势不写主张，补第三个种子再定。
- **H2 支持**：种子 1337 的 ≥10 m 分层 SR 差 > 0 且 NE 差 < 0（与种子 42 同向），
  且合并配对 CI 在 SR 或 NE 至少一项上不含 0 → 写"长路径上提升"，intro 改为"长路径上提升、全量不劣"。
- **H2 否定**：种子 1337 分层 SR 差 ≤ 0 或 NE 差 ≥ 0（方向翻转）→ 长路径主张整体删除，
  不换阈值重试。
- **H2 没测出来**：同向但合并 CI 在 SR 和 NE 上都含 0 → 作为描述性观察放在分析段，不进主张。

**设置.** 两臂全量 1839 集、协议种子 1337、其余与种子 42 完全一致。桥臂：修复栈
`scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh`（`PPA_EVAL_PROTOCOL_SEED=1337`，
checkpoint/config 与 EXP-01 终局相同，输出根
`model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu_seed1337`）。native 臂：认证复刻栈
（把 `evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802` 复制为新 plan 目录、
只改 `PROTOCOL_SEED`，原件不动；提交物待补）。分析：`paired_closed_loop_bootstrap.py`
逐种子 + 合并。**代价：8 卡 × 约 18 小时 × 2 臂。**
提交物：[exp07-seed1337-submission.md](exp07-seed1337-submission.md)。

---

## 4. 公共资源

所有路径都在 `/mnt/afs/liwenhao/agent/370910109/` 下（旧路径 `/mnt/afs/lixiaoou/intern/fjl`
一律是迁移遗留的过期值，见 `CLAUDE.md` §6）。

### 基线数字（引用时以这里为准）

| 量 | 值 | 出处 |
|---|---|---|
| native System1 闭环 SR（R2R val-unseen 全量 1839） | 62.48% / SPL 55.23% / OS 70.58% / NE 4.15 m | `model/eval_internnav_native_r2r_val_unseen_4gpu_rpcv2_x11bundle_v4/merged/` |
| 当前方法闭环 SR | 62.81% / SPL 55.04% / OS 71.72% / NE 3.98 m | `model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/merged/result.json` |
| v1（病灶对照） | 18.11% / SPL 15.62% | `model/eval_ppa_action_refine_online_amb3r_r2r_val_unseen_8gpu/merged/` |

### 金参照与产物

| 物 | 路径 |
|---|---|
| 认证复刻栈（62.5% 的 golden 参照，**不在 git 仓库里**） | `evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/tools/` |
| 部署 checkpoint（v2 信赖域桥） | `model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth` |
| v1 漂移桥 checkpoint（EXP-05 用；部署必须用 `best_deployment_full.pth`，`best.pth` 缺未来头张量） | `model/output_past_plan_action_action_refine_v1_8gpu/run_20260818_225001/checkpoints/best_deployment_full.pth` |
| Stage2 联合 best（79 Heatmap + 11 Future 张量来源） | `model/output_past_plan_action_v1_8gpu_stage2_retry1/stage2_joint/run_20260818_104438/checkpoints/best.pth` |
| 捷径诊断结果 | `model/heatmap_shortcut_probe_v1/seed_{42,1337}/` |
| 种子 42 配对 bootstrap（终局 vs native，含 ≥10 m 分层） | `model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/analysis/paired_vs_native_seed42.json`（`scripts/tools/paired_closed_loop_bootstrap.py`） |

### 数据集

| 数据 | 路径 | 位姿来源 | 划分 |
|---|---|---|---|
| random-walk 全景（历史头预训练/诊断） | `data/heatmap_randomwalk_train_v1` | Habitat GT | 场景 MD5 自动划分，54 : 7 |
| R2R 全景 v2（Stage1/2 训练） | `r2r_panoramic_data_v2/train` | AMB3R VO 缓存 | 场景 MD5 自动划分，26 场景 / 5000 clips |
| AMB3R endpoint 缓存（配 R2R v2） | `data/amb3r_endpoint_v3_full_r2r` | — | train,val |

## 5. 反复踩过的坑

1. **冻结模块的输入保真度必须逐字节验证。** EXP-01 里 49 个点的 SR 差距，全部来自
   六处 prompt/图像级差异。栈内自洽（零桥前向等价）证明不了跨栈等价。
2. **一次只改一个变量，否则消融就废了。** EXP-01 同时改了桥和评测栈，代价是
   还得补 EXP-05 和 EXP-06 才能拆开。
3. **判据必须在看到结果之前写死。** EXP-02 的判据是结果出来后补的。这次结论没有歧义
   （所有指标同向、两种子一致），但下次不一定，而"事后判据"没有任何约束力。
4. **探针的训练预算会左右结论方向。** 短预算天然偏向易学的捷径（位姿→像素是低维光滑映射，
   视觉定位不是）。EXP-02 用 12000 步并确认四条曲线全部走平，才排除了"训练不足"这个解释。
5. **诊断脚本可能是给旧架构写的。** `diagnose_heatmap_shortcuts.py` 原本只支持
   legacy 全景 + LoRA 栈；论文写的是 `internnav_single_view`。跑之前先确认脚本跑的
   是不是论文里那个架构，否则结论无效。
