# PPA v2 SR 恢复报告：18.1% → 62.8%，与 native 持平

**结论（2026-09-01 定稿）**：在保持 Past→Plan→Action 架构完整在线（桥生效率 99.6%）
的前提下，R2R val-unseen 全量 1839 集闭环 SR 达到 **62.81%**，与 native 原生
System1 基线 **62.48%** 统计等价（+0.33pt，二项噪声 ±1.1pt 以内），SPL 持平
（55.04% vs 55.23%）。最初的 v1 评测为 18.11%。

| 指标 | v1（起点） | **v2 + 栈修复（终局）** | native 基线 |
|---|---|---|---|
| SR | 18.11% | **62.81%** (1155/1839) | 62.48% |
| SPL | 15.62% | **55.04%** | 55.23% |
| OS | 20.4% | **71.72%** | 70.58% |
| NE | — | **3.98 m** | 4.15 m |
| PPA 生效率 | 51% | **99.6%** (1831/1839) | — |

- 终局评测：`model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/merged/result.json`
  （arm=`ppa_stage2_online_amb3r`，协议 v3，确定性种子 42）。
- checkpoint：`model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth`
  + `configs/ppa_action_refine_v2_8gpu.yaml`（信赖域必须由该 config 在推理端启用）。

---

## 一、v1 的 18.1% 是两个叠加病灶，且大头曾被误诊

### 病灶 1（小头）：桥漂移 —— 无约束 action loss 把 plan_z 推离冻结 DiT 的条件流形

v1 action-refine 训练（run_20260819_161917）实测：Δ 每元素 RMS 漂到 ~0.7、
6 个 epoch 单调增长不收敛，换来的 teacher-forced 速度 MSE 改善 ≤4%（在 epoch
波动噪声内）；0.5×preserve + 0.01×delta 的软惩罚在与 action 项的博弈中必输。
best checkpoint 还按 teacher-forced MSE 选择——恰好选中漂得最远的。

### 病灶 2（大头）：v3 评测协议栈给冻结 System2 喂了错误输入

分解两轮 PPA 评测的 per-episode 数据发现：**41% 的 episode（758 集）从未产生
一次 pixel goal**，中位 13 步内死亡，v1/v2 两轮的死亡集合逐集 100% 重合；同一批
episode 在 native 复刻臂上 SR 为 68.3%（比平均还容易）。同一 episode、同一份冻结
VLM、贪心解码，两臂第 0 步输出即不同（`←←←←` vs `→→→→`）⇒ System2 输入不同。

v3 栈相对 62.5% 认证复刻栈的六处实测差异：

| # | 项 | 认证复刻栈 | v3 栈（修复前） |
|---|---|---|---|
| 1 | 会话 lookdown | 640×480 强制 | resize 成 384×384 |
| 2 | history 图间分隔 | 无 `\n`（clean 掉） | 每图后 `"\n"` |
| 3 | instruction 句号 | `replace("<instruction>.")` 不补 | format 后强补 → 双句号 |
| 4 | turn 执行 | 整串 ≤4 步 | 单步执行 |
| 5 | `↑`/混合箭头输出 | ↑=FORWARD 整串执行 | 不识别 → fallback_stop |
| 6 | 二轮后仍是箭头 | 继续 native_actions | fallback_stop 杀 episode |

死锁机制：System2 在错误输入分布下只输出 turn/stop → agent 无平移 → AMB3R VO
无视差无法初始化 → pose 永不 ready → PPA 永不生效 → episode 十几步内终结。
v1 的 18.1% = 病灶 2 杀死 41% 样本 + 病灶 1 拖累存活样本，当时全部记在"桥漂移"
头上是误诊；stage0 A/B 只审计了 v3 栈内部零桥前向等价，从未审计 v3 栈 ≟ rpcv2 栈。

## 二、修复

### v2 重训（commit e73b774，run_20260829_115642）

- 硬信赖域：逐 token ‖Δ‖ ≤ 5%·‖plan_z0‖，训练/部署同款（`max_delta_ratio: 0.05`）；
- 相对 delta 惩罚（×10）+ preserve×2.0：平坦方向衰减归零，"沉默"成为默认最优；
- advantage 加权 action loss（native 错得多才有权重，上限 4×）；
- 采样 rollout 验证（共享噪声 bridged vs native、部署级后处理）作为选点指标。

训练侧结果：Δ 稳定在 ~2%（远未顶到 5% 上限，贴边 token <0.5%）、preserve
~0.0001、**四个 epoch 的采样终点误差全部略优于 native**、动作一致率 70-81%。

### 栈修复（commit ed46c76）

`src/models/heatmap/native_internnav_exact.py` 逐字移植认证复刻栈的 System2 前端
（prompt 构造 + 动作解析），**token parity 逐位验证**（真 processor 下 6 种组合的
input_ids / pixel_values / image_grid_thw 与复刻实现全部相等）；
`rpc_model_server.py` 的 two-turn 分支切换到该构造、强制 640×480 会话 lookdown、
返回认证语义的动作串；client 以 640×480 采集 two-turn lookdown。
训练侧 prompt 契约（`construct_input_stage2`）未动。

## 三、验证链

1. **20 集死亡集探针**（开发机 1 GPU）：SR 0% → **75%**，PPA 生效 20/20，
   行为形态与 native 重合（56.5 步/11 次轨迹调用 vs 62/12）。
2. **全量终局双 cohort 对照**（同一批 episode，三臂）：

| Cohort | v1 | **终局** | native |
|---|---|---|---|
| 死亡集（758） | 1.5% | **67.68%** | 68.34% |
| 存活集（1081） | 29.8% | **59.39%** | 58.37% |

死亡集恢复到 native 水位，存活集反超 native 1.0pt——两个病灶分别痊愈，
且桥在线时存活集不劣于 native（与训练侧开环测量一致）。

## 四、产物索引

| 物 | 路径（`/mnt/afs/liwenhao/agent/370910109/` 下） |
|---|---|
| 终局评测 | `model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu/merged/` |
| v1 评测（病灶对照数据） | `model/eval_ppa_action_refine_online_amb3r_r2r_val_unseen_8gpu/merged/` |
| native 基线评测 | `model/eval_internnav_native_r2r_val_unseen_4gpu_rpcv2_x11bundle_v4/merged/` |
| v2 训练 run | `model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/` |
| 认证复刻栈（golden 参照） | `evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/tools/` |
| 20 集探针 | `tmp/native_fix_dead20_out/` + `tmp/native_fix_dead20_cohort.json` |
| 关键 commit | 重训 `e73b774`；采集器 `77e374a`/`9a9ca49`；栈修复 `ed46c76` |
| 复现命令 | `docs/experiments/exp01-sr-recovery-submission.md` |
