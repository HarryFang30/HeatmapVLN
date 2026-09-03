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
| [EXP-08](#exp-08-阶段二联合训练买到了什么) | 阶段二（认知头与动作联合训练）对最终桥的行为有没有可测影响 | 🔬 | 2026-09-03 提交（链式任务第 4 臂） |
| [EXP-09](#exp-09-阶段三配方里哪一项真的在起作用) | 阶段三五处改动里，信赖域 / advantage / 惩罚重校准 / rollout 选点各自有没有可测效应 | 🔬 | 2026-09-03 提交 A/B/C（链式任务前 3 臂）；D 免费；参照重验待跑 |
| [EXP-10](#exp-10-未来认知头与桥的关联是否可观测) | 注入的历史记忆是否改变未来预测（关联的可观测性）；Z 的第 i 个向量是否真的绑定第 i 段 | 🔬 | 2026-09-04 桥开臂随 EXP-09-R 跑；桥关臂待跑；token 绑定探针待写 |
| [EXP-11](#exp-11-四方向表征的标签覆盖率) | 四方向表征在标签层面覆盖了什么——替代"只预测更少方向"的重训消融 | 🔬 | 2026-09-04 开发机全量 val 在跑（40 集预览已看到，见条目） |

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

**设置（2026-09-04 确定）.** 同 EXP-02，唯一变量是位姿来源：探针新增
`--amb3r-pose-cache-root`（启动脚本 `SHORTCUT_AMB3R_POSE_CACHE_ROOT`），数据仍用
`data/heatmap_randomwalk_train_v1`，位姿来自 `data/heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu`。
机制已在开发机 CPU 冒烟验证（provider=`amb3r_vo_cache`、位姿有限且异于真值、非单视角架构 fail-closed）；
summarizer 新增两项检查，拒绝把真值域与 VO 域混进同一张表。**代价：8 卡 × 约 6 小时**（4 模式 × 2 种子）。

**边界（预先声明）**：缓存只覆盖 62/78 个场景并限制可用帧，同一 val 划分下带缓存 6578 样本、
真值 6584 样本 —— 本实验**不与 EXP-02 样本匹配**，只能做 VO 域内部比较，绝对数字不可与 EXP-02 对齐着读。
"真值训练 / VO 评测"这一臂**不另跑**，引用既有产物（0.9079 → 0.4984，适配后 0.5935），
边界同样写在提交物里。
提交物：[exp04-vo-pose-probe-submission.md](exp04-vo-pose-probe-submission.md)。

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
`model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu_seed1337`）。native 臂：认证复刻栈的 plan 副本
`evaluation_plans/internnav_native_r2r_val_unseen_8gpu_seed1337`，跑在锁定代码
`HeatmapVLN_native_lock_bd5ead1`（`git archive bd5ead1`）上，原件不动；启动脚本相对金参照只差
5 行（根路径、锁定 repo、plan 路径、输出根、种子），细节与导入检查见提交物。
**边界（预先声明）**：锁定 manifest 里 `scripts/training/utils.py` 的原版本是 2026-08-02 服务器工作区
未提交的状态，git 中不存在；副本用 bd5ead1 版本（差异只在 LoRA 计数与训练损失构造器，
不在 native 评测路径上）。其余 3 个 HeatmapVLN 运行时文件与锁定 hash 逐字节相同。分析：`paired_closed_loop_bootstrap.py`
逐种子 + 合并。**代价：8 卡 × 约 18 小时 × 2 臂。**
提交物：[exp07-seed1337-submission.md](exp07-seed1337-submission.md)。

---

### EXP-08 阶段二（联合训练）买到了什么

**问题.** 方法节给阶段二的理由是"动作监督的梯度必须流入认知分支，认知才能学到对决策有用的内容"。
这句话没有数据支撑。最终部署的桥是阶段三**从零**重训的，认知头在阶段三全程冻结——所以阶段二
留给最终模型的唯一东西就是"被动作损失塑形过的认知头权重"。它有没有可测影响？

**假设.** 有：接阶段二权重的桥（现行 v2）应比接阶段一权重（认知头从未见过动作损失）的桥
在 rollout 上更好。

**判据（预注册，2026-09-03，开跑前写死；全部在 v2 数据 val 划分、512 对共享噪声 rollout 上读）.**
- **支持**：接阶段一权重的臂，其 best checkpoint 的 bridged rollout 终点误差比 v2 参照
  （同样 512 对重验）差 > 0.03 m 且配对 bootstrap 95% CI 不含 0；或动作一致率低 > 5pt。
  → 阶段二的理由句成立，可写。
- **否定**：|终点误差差| ≤ 0.01 m 且一致率变化在 ±2pt 内，且认知指标（PCK@8、top-k 召回）
  两臂差 ≤ 1pt。→ 阶段二对最终桥**无可测影响**，方法节删掉理由句，训练故事简化为
  "阶段一 → 阶段三"（阶段二降为可选）。
- **没测出来**：落在两者之间。→ 补第二个训练种子再定。
- 认知指标附带判读：两臂的头都冻结、各自等于父 checkpoint 的头，因此两臂 PCK@8 / F1 /
  top-k 召回的差就是"阶段二对认知的影响"；差 > 2pt 记为阶段二改变了认知（方向另记）。

**设置.** 配方与 v2 完全相同（`configs/ablation/exp08_stage3_from_stage1_heads_8gpu.yaml`
与 v2 config 只差 `val_rollout_batches: 8 → 64`），**唯一变量是 `--load-weights` 的父 checkpoint**：
阶段一 `model/output_past_plan_action_v1_8gpu/stage1_map_pretrain/run_20260817_205027/checkpoints/best.pth`
（79 热力图 + 11 未来头，桥 0 张量；loader 只取头，桥从零起）。v2 数据、3 epoch、
按 512 对 rollout 终点误差选点。**代价：8 卡 × 约 5–6 h**（链式任务第 4 臂）。
提交物：[exp08-exp09-stage3-ablation-submission.md](exp08-exp09-stage3-ablation-submission.md)。

---

### EXP-09 阶段三配方里哪一项真的在起作用

**问题.** 阶段三相对 v1 改了五处：① 硬信赖域 ρ=0.05；② 保持 2.0 + 相对 Δ 惩罚 ×10（v1 为
0.5 + 0.01 绝对）；③ advantage 加权；④ rollout 终点误差选点（v1 用 teacher-forced MSE）；
⑤ 桥从零重置。EXP-05 测的是"全关"的闭环代价；本实验在开环上逐项拆：哪些是必要的，
哪些只是防御性设计。⑤ 不做（热启动阶段二桥就是 v1，被 EXP-05 覆盖）。

**参照数字（v2 run，64 对 rollout，`run_20260829_115642`）.** 训练侧 Δ 相对幅度均值
0.017–0.020，贴边比例 0.05%–0.5%；bridged 终点误差 1.269–1.275 m vs native 1.281–1.285 m；
动作一致率 0.70–0.81；preserve 损失 1e-4。64 对下 bridged−native 的差本身在噪声内，
所以所有臂（含参照）统一用 **512 对**（`val_rollout_batches: 64`）重读。

**假设.** ①②③④ 各自都有可测效应（这是方法节的隐含主张）。

**判据（预注册，2026-09-03，开跑前写死；"v2@512"指参照重验后的数字，见设置）.**
- **臂 A（去 ρ）**：支持 ρ 必要 —— best checkpoint 的训练侧 `delta_token_ratio_mean` > 0.10，
  或 bridged−native 终点误差差 > +0.03 m 且 512 对配对 CI 不含 0。否定（ρ 冗余）——
  Δ 均值 ≤ 0.05 且 bridged−native 差的 CI 含 0 → 方法节把 ρ 降为"防御性截断，
  v2 配方下训练中未触发"。没测出来 —— 0.05 < Δ ≤ 0.10 且 CI 含 0。
- **臂 B（去 advantage）**：支持 —— B 的 bridged 终点误差比 v2@512 差 > 0.03 m 且 CI 不含 0，
  或一致率比 v2@512 低 > 5pt。否定 —— |差| ≤ 0.01 m 且一致率在 ±2pt 内 → advantage
  从方法节删除或降为实现细节。没测出来 —— 其余。
- **臂 C（惩罚退回 v1 值、保留 ρ）**：支持重校准必要 —— best checkpoint 的
  `delta_at_boundary_frac` > 0.20（Δ 顶到上限）或 preserve 损失 > 10× v2@512。否定 ——
  贴边 < 0.05 且 bridged−native 差的 CI 含 0 → 有 ρ 兜底时惩罚校准冗余。没测出来 —— 其余。
- **臂 D（选点指标，免费）**：v2 run 按 `val_trajectory_loss` 最小选的是 epoch 4（0.3432），
  按 rollout 选的是 epoch 3（1.2688 vs epoch 4 的 1.275，64 对下不可分）。支持 ——
  两个 checkpoint 在 512 对重验下 rollout 终点误差差 > 0.02 m 且 CI 不含 0。否定 ——
  差 ≤ 0.005 m，或 CI 含 0 且差 ≤ 0.01 m → 选点指标无可测影响，方法节把"模型选择看真实
  生成效果"降为流程说明。没测出来 —— 其余。
- 判据双向都接受：四项里任何一项被否定，方法节对应的句子就删或降级，不改判据。

**设置.** 三个重训臂各自从 v2 config 派生、**只改一处**（`configs/ablation/exp09{a,b,c}_*.yaml`，
派生 diff 已在 commit 里核对）：A `max_delta_ratio: null`；B `action_advantage_enabled: false`；
C `preserve_weight 0.5 / delta_z_weight 0.01 / delta_z_relative false`。父 checkpoint、数据、
epoch、选点与 v2 相同；`val_rollout_batches: 64`（512 对）。链式启动
`scripts/run_stage3_ablation_chain_8gpu_mxc500.sh`，顺序 A → B → C → EXP-08，单臂失败不阻塞后续。
**参照重验（EXP-09-R）**：v2 的 `best.pth`（epoch 3）与 `epoch_004.pth` 各用 512 对重验一遍：
`scripts/train.py --validate-only --load-weights <ckpt> --config configs/ablation/exp09r_stage3_v2_revalidate_512_8gpu.yaml`
（该 config 与 v2 只差：加载已训练的桥 `past_plan_action_reset_bridge: false`、
`evaluate_before_training: true`、`val_rollout_batches: 64`；结果在 run 目录
`manifest/pre_training_validation.json`）。**代价：8 卡 × 约 5–6 h × 3 臂 +
重验约 1 h。** 单训练种子起步，边缘结论再补种子（2026-09-03 规划决定）。
提交物：[exp08-exp09-stage3-ablation-submission.md](exp08-exp09-stage3-ablation-submission.md)。

---

### EXP-10 未来认知头与桥的关联是否可观测

**问题.** 方法节说三类输出经同一个 Z_t 关联、信息沿"历史认知 → Z_t → {未来热力图, 动作}"单向流动。
桥的 Δ 只有 ~2%，那么注入到底有没有**改变未来预测**？如果未来指标对桥开/关完全不敏感，
"关联"就只是结构上的说法，没有可观测证据。第二个问题是结构性的：未来头把 Z_t 的第 i 个
向量解码成第 i 段路点，这个绑定是真的还是装饰？

**假设.** H1：桥开时未来指标与桥关（Δ=0）有可测差异。
H2：打乱 Z_t 的 4 个向量顺序会明显破坏未来预测（说明 token↔时段绑定是真实结构）。

**判据（预注册，2026-09-04）.** 已知信息披露：v2 checkpoint 在 **64 对**规模下的未来指标已见过
（Soft-IoU 0.2397 / top-k 召回 0.7717 / 可见性 F1 0.9148，`run_20260829_115642` epoch 4）；
**桥关臂的数字尚未产生**，H1 的判据据此在看到它之前写死。
- **H1 支持**：桥开 vs 桥关的 top-k 支持召回差 > 1pt，或 Soft-IoU 差 > 0.01（同 checkpoint、
  同 val 集、同一次数据顺序）。→ 可写"历史注入改变了未来预测"。
- **H1 否定**：两者三项指标差都 < 0.2pt / 0.002。→ 未来头对注入不敏感，方法节的"关联"
  只保留结构描述（同一 Z_t 解码），不声称可观测效应。
- **H1 没测出来**：落在中间。→ 记为趋势，不进主张。
- **H2 支持**：打乱后 Soft-IoU 或 top-k 召回相对不打乱下降 > 20%（相对值）。→ 可写
  "第 i 个向量负责第 i 段"。**否定**：下降 < 5%。→ 删掉逐段对应的说法，改写成
  "N_z 个向量共同解码未来区域"。**没测出来**：5%–20%。
- 附带（无判据、只报告）：未来侧/后视角 Soft-IoU ≈ 0 与 Soft-IoU 在训练中随 loss 下降而下降
  的现象，如实写进实验，并以 top-k 召回作为主指标（EXP-11 给出标签层面的原因）。

**设置.** H1 = 两次 `--validate-only`：桥开用
`configs/ablation/exp09r_stage3_v2_revalidate_512_8gpu.yaml`（与 EXP-09-R 同一次运行，未来指标顺带产出）；
桥关同 config 但 `past_plan_action_reset_bridge: true`（桥归零 ⇒ Δ=0，前向与 native 逐位一致）。
**代价：开发机 8 卡 × 约 0.5 h**（桥开臂已随 EXP-09-R 在跑）。
H2 需要一个探针工具（`chain.decode_future` 外包一层，捕获 `plan_z / past_output / past_head` 后
用打乱的 plan_z 再解一次，同一前向、同一 batch），**尚未实现**，优先级低于 EXP-04。

---

### EXP-11 四方向表征的标签覆盖率

**问题.** "只预测前视 / 更少方向"的重训消融要改模型和标签、重训阶段一（7.5 h），而结论可以从
标签本身读出来：历史路点根本不在前视里。所以用**描述统计**替代那次重训，直接回答"为什么要四方向"。

**这不是假设检验**，没有支持/否定判据；预先声明的是**它能许可什么措辞**：
- 若"任一视角都不可见"的比例 < 10% 且非前视占绝大多数 → 可写"历史路点绝大多数不在前视视野内，
  只前视的表征无法表达它们"，并据此说明四方向的必要性；
- 若未来标签的侧/后视支持比例 < 5% → 必须如实写"四方向在未来头上主要起统一坐标系的作用"，
  不得把四方向说成对未来预测的增益。

**已看到的部分结果（诚实披露）.** 40 个 val 样本（320 个历史槽位）的预览：历史可见比例
前 0.0% / 右 9.4% / 后 83.4% / 左 2.2%，任一视角都不可见 5.0%，可见但从不在前视 95.0%；
未来时段×视角 bin 前 91.9% / 右 8.1% / 后 0.0% / 左 2.5%，仅前视 89.4%。上面的措辞规则是在
看到这份预览**之后**写的——因此本条目按"描述统计"报告，不作为检验任何假设的证据。

**设置.** `scripts/tools/summarize_direction_coverage.py --config configs/ppa_action_refine_v2_8gpu.yaml
--split val --max-samples 0`，走生产数据集（v2 数据 + AMB3R 缓存 val 划分，4734 个样本），
标签由 `_compute_per_history_multiview_heatmaps` / 未来标签渲染器生成，与训练看到的完全一致。
纯 CPU，**代价约 1 h**，产物 `model/exp11_direction_coverage/coverage_val_full.json`。

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
| Stage1 对齐 best（EXP-08 的父 checkpoint：79 Heatmap + 11 Future，桥 0 张量） | `model/output_past_plan_action_v1_8gpu/stage1_map_pretrain/run_20260817_205027/checkpoints/best.pth` |
| 阶段三消融臂输出（EXP-08/09） | `model/ablation_stage3/<arm>/run_*/` |
| EXP-09-R / EXP-10 桥开重验（512 对） | `model/exp09r_revalidate_512/{best,epoch_004}/run_*/manifest/pre_training_validation.json` |
| EXP-11 方向覆盖率 | `model/exp11_direction_coverage/coverage_val_full.json` |
| 捷径诊断结果 | `model/heatmap_shortcut_probe_v1/seed_{42,1337}/` |
| 捷径诊断（VO 位姿域，EXP-04） | `model/heatmap_shortcut_probe_vo_v1/seed_{42,1337}/` |
| 位姿域偏移代价（真值训练→VO 评测） | `model/output_heatmap_amb3r_pose_adapt_endpoint_v2_4gpu/runs/run_20260814_234429/logs/metrics.jsonl` 的 `pre_training_validation` |
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
5. **认证复刻栈跑不了现行 HEAD。** 它的 `manifests/runtime_code.sha256` 把 4 个 HeatmapVLN
   运行时文件锁在 2026-08-02 的版本上，启动脚本还断言 RPC 协议号是 `heatmapvln-r2r-json-v2`；
   现在 HEAD 上协议号已是 v3，3 个文件改过。要再跑 native 臂，必须在锁定 checkout
   （`HeatmapVLN_native_lock_bd5ead1`）上跑，见 EXP-07 提交物。**永远不要为了让金参照跑起来
   去改它的 manifest 或启动脚本**——那等于把基线改掉。

6. **本地跑不了配置校验。** 本机 python 没有 `yaml`/`pydantic`，`load_and_validate_config`
   只能在开发机上跑（`envs/qwen25/bin/python`）。新配置一律 scp 到开发机验一遍 schema 再提交。

7. **`tests/test_config.py::test_paths_merge_overrides_data_and_log` 有导入顺序依赖。**
   单独跑通过；在某些前置 import（例如先 import `scripts.train`）之后跑会失败。已确认在
   **干净 HEAD** 上也如此，与本轮改动无关，也不在 §4 的基线计数里 —— 但看到它失败先换个
   顺序单独复现，别急着归因给自己的改动。

8. **诊断脚本可能是给旧架构写的。** `diagnose_heatmap_shortcuts.py` 原本只支持
   legacy 全景 + LoRA 栈；论文写的是 `internnav_single_view`。跑之前先确认脚本跑的
   是不是论文里那个架构，否则结论无效。
