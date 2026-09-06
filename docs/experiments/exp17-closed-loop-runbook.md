# EXP-17 闭环 runbook：认知臂的部署、金丝雀与全量

判据见台账 [EXP-17](README.md#exp-17-认知前缀慢系统在决定之前显式写出来路与进度) 与 EXP-14 的 14-C 准入。
**闭环只在决策级四道门都过之后跑**：认知质量、承重测试、决定改善、保持门（`preservation_generated ≥ 0.98`、
`nonpixel_on_normal ≤ 0.005`、`stop_false_alarm_normal ≤ 0.002`）。

## 0. 部署机制（`scripts/evaluation/rpc_model_server.py --system2_cognition_arm`）

每次慢系统调用：

1. AMB3R 未建图（`pose_ready=false`）或无历史 → **关掉适配器**走认证 native 路径，响应带 `cognition_applied=false`、
   `cognition_skip_reason=amb3r_map_warmup`。与 PPA 部署的暖机行为一致。
2. 否则，决定通道：训练同款提示（`construct_input_stage2`，conjunction `you can see `，8 个位姿哨兵放在当前帧之后），
   AMB3R 相对位姿经 `geometry` 模块变成 token 嵌入，微调后的慢系统贪心生成第一轮：认知前缀 + 决定。
   剥掉前缀（`parse_cognition_prefix`），决定按 native 规则解析：转向 / STOP 直接执行；`↓` 则用同一模型再生成第二轮拿像素坐标。
3. 像素目标的 Z⁰ 通道：**关掉适配器**，用认证 native 提示（`build_native_messages` + `append_native_lookdown_turn`）
   加 native 格式答案 `"v u"` + `<eos>` 取 TRAJ latent，走原封不动的 System1。给定同一个像素目标，这条路与 native 逐位相同。
4. 响应字段：`ppa_runtime=system2-cognition-arm-v1`（客户端据此校验）、`cognition_applied`、`cognition_generated_text`、
   `cognition_prefix_fields`、`cognition_decision_text`、`cognition_second_turn_text`、`z0_source=native_adapters_disabled`。
   `--cognition_audit_native` 会在每次就绪调用上再解一次 native 首轮并记录 `native_audit_agrees`（慢系统开销翻倍，只用于金丝雀）。

客户端（`r2r_val_unseen.py`）在每集结果里写 `cognition_applied_calls`；启动脚本用它做完成检查。

## 1. 金丝雀（600 集，约 48 卡时，网站提交）

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

R=/mnt/afs/liwenhao/agent/370910109
export COG_EVAL_CHECKPOINT=$R/model/exp17_cognition_prefix/exp17b/run_XXXXXXXX_XXXXXX/checkpoints/best.pth   # 填实际 run
export COG_EVAL_CONFIG=$R/HeatmapVLN/configs/ablation/exp17b_c3_geometry_prefix_stop_lora_8gpu.yaml
export COG_EVAL_OUTPUT_ROOT=$R/model/eval_system2_cognition_exp17b_canary600_r2r_val_unseen_8gpu
export COG_EVAL_ARM=system2_cognition_exp17b_canary600
export COG_EVAL_MAX_EPISODES_PER_SHARD=75
export COG_EVAL_AUDIT_NATIVE=1
export COG_EVAL_PROTOCOL_SEED=42

bash scripts/run_system2_cognition_r2r_val_unseen_8gpu_mxc500.sh
```

`COG_EVAL_MAX_EPISODES_PER_SHARD=75` 取 8 个锁定分片各自前 75 集（迭代器顺序，确定性），合 600 集。
配对读数：与 native 种子 42 的 `merged/progress.json` 按 episode_id 取交集，用
`scripts/tools/paired_closed_loop_bootstrap.py`；**配对差 < −3 pt 即停**（约 2.3 个标准误）。
同时报：动作序列与 native 完全一致的 episode 比例（`native_audit_agrees` 逐调用为真的集数）、被覆盖成 STOP 的集的 NE 分布、
2 m 严格半径下的 SR。

## 2. 全量两种子（各 144 卡时，网站提交）

去掉 `COG_EVAL_MAX_EPISODES_PER_SHARD` 与 `COG_EVAL_AUDIT_NATIVE`，`COG_EVAL_PROTOCOL_SEED` 分别取 42 与 1337，
输出根各自独立。判据：EXP-07 同款，两种子合并 SR 差的 95% CI **下界 > −1.5 pt** 为非劣；
早停否决项：从未进 3 m 圈就停的比例相对 native 上升超过 0.5 pt 判否。native 两个种子已有，不重跑。

## 3. 位姿域

训练输入是 Habitat 真值位姿，部署是 AMB3R。封存 DAgger 回填不了 AMB3R（台账 §5 第 26 条）。顺序：

1. 先做零成本的噪声读数（开发机三卡分片）：
   `eval_system2_cognition_prefix.py ... --pose-noise-translation-m 0.2 --pose-noise-rotation-deg 10`，
   与干净读数比较停/转向指标。掉 < 5 pt → 直接进金丝雀。
2. 掉 ≥ 5 pt → 先在台账登记 exp17c（`configs/ablation/exp17c_c3_geometry_prefix_stop_posenoise_lora_8gpu.yaml`，
   训练期 EXP-15 噪声增广），网站提交 `EXP13_ARMS="exp17c"`，复读后再进金丝雀。

## 4. 启动后先确认的三件事

`runtime/<stamp>/logs/model_0.log` 里：

```
System2 cognition arm runtime enabled: {"cognition_prefix": true, "mode": "geometry", "num_tokens": 8, "placeholder_position": "after_current", ...}
```

客户端日志里第一批调用应有 `cognition_skip_reason=amb3r_map_warmup`（暖机），之后 `cognition_applied=true`。
`merged/progress.json` 每行 `cognition_applied_calls > 0`、`history_pose_source == amb3r_vo_da3`。
