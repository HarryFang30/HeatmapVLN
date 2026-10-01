# EXP-20 消融表：复现命令（RTX 4090，容器 `fjl-habitat`）

判据与设置见台账 [EXP-20](README.md#exp-20-论文消融表完整模型与三个消融臂两种子)。本文件只写怎么跑。
所有路径都是容器路径（宿主机 `/home/fangjialei` = 容器 `/workspace`）。

## 0. 代码与权重

- 代码：每次用一个提交的 `git archive` 副本，放在 `/workspace/exp20/src_<SHA>`，不碰部署检出 `/workspace/HeatmapVLN`。
  A0 用的是 `src_d9ef314`。A1 需要 `--ppa_bridge_off`，所以要用含这个开关的提交。
- 权重：

| 臂 | `PPA_EVAL_CHECKPOINT` | `PPA_EVAL_CONFIG` | 其他 |
|---|---|---|---|
| A0 完整模型 | 默认 `/workspace/weights/ppa_refine_v2_best.pth` | 默认 `configs/ppa_action_refine_v2_8gpu.yaml` | — |
| A1 桥关 | 同 A0 | 同 A0 | `PPA_EVAL_BRIDGE_OFF=1` |
| A2 无约束桥 | `/workspace/weights_exp20/exp05_v1_unconstrained_bridge_best_deployment_full.pth` | `$REPO/configs/ppa_action_refine_8gpu.yaml` | — |
| A3 惩罚退回 v1 | `/workspace/weights_exp20/exp09c_stage3_ablation_best.pth` | 同 A0 | — |

- sha256（2026-09-30 拷上去后核对过，与 C500 原件一致）：
  - `ppa_refine_v2_best.pth`：`0b5a0644…6c69`
  - `exp05_…_deployment_full.pth`：`5e6f0995…acacc`
  - `exp09c_…_best.pth`：`61ed694b…922d`
- 两份配置比对过：
  - 权重里嵌的模型字段与对应 yaml 逐项相同。
  - 两份 yaml 在评测相关的字段上只差 `past_plan_action.max_delta_ratio`（v2 是 0.05；v1 没有这个字段，即不截断），其余差别都是训练损失。

## 1. 一次运行

```bash
docker exec -d fjl-habitat bash -lc "cd /workspace/exp20/src_<SHA> && \
  PPA_EVAL_REPO=/workspace/exp20/src_<SHA> PPA_EVAL_GPU_DEVICES=<gpus> PPA_EVAL_PROTOCOL_SEED=<42|1337> \
  PPA_EVAL_ARM=exp20_<arm>_seed<seed> PPA_EVAL_OUTPUT_ROOT=/workspace/eval_runs/exp20_<arm>_seed<seed>_4090 \
  PPA_EVAL_MODEL_PORT_BASE=<52400+10k> PPA_EVAL_VO_PORT_BASE=<52500+10k> PPA_EVAL_DISPLAY_BASE=<360+10k> \
  <上表的臂变量> nohup bash scripts/run_ppa_r2r_val_unseen_cuda.sh > /workspace/exp20/logs/<arm>_seed<seed>.out 2>&1"
```

- **几个运行同时跑**：端口和显示号要错开，第 k 个运行各加 10k。脚本发现显示号被占用会直接退出。
- **一个种子一个输出目录**：续跑（`--resume`）会跳过目录里已有的集。中断后用同一组变量重启即可。
  2026-09-30 A0 种子 42 就是这样从单卡改成两卡续跑的，前 2 集与 09-28 金丝雀逐调用相同。
- **只有 A0 种子 42 开 `PPA_EVAL_TIMING=1`**：开计时不改动作，但会在每个请求上同步 CUDA。
- **停止运行**：对启动脚本的 PID 发 TERM（`docker exec fjl-habitat kill -TERM <pid>`），它会自己收掉服务端、客户端和 Xvfb。不要 `pkill -f`。
- **跑完**：在宿主机上 `chown -R 1015:1015`（容器以 root 写文件）。

## 2. 开全量前的金丝雀（A1 / A2 / A3 各一次，种子 42）

在上面的命令里加 `PPA_EVAL_SHARDS=0,1 PPA_EVAL_MAX_EPISODES_PER_SHARD=2`，输出目录用 `exp20_<arm>_canary`。
这 4 集就是 09-28 金丝雀的 4 集。

**通过条件**（开跑前写死）：

1. 服务端启动日志里有 PPA 预检证据；A1 还要有 `PPA bridge off (EXP-20 A1)` 这一行。缺了脚本会自己退出。
2. 4 集都跑完、没有报错，每个分片的 `ppa_applied_calls > 0`。
3. 与 A0 金丝雀（`/workspace/eval_runs/canary_cuda_seed42`）逐调用比对。每集在第一次就绪调用之前的调用（里程计预热，没有 PPA）必须完全相同：
   - A1 只换了快系统的输入，所以第一次就绪调用之前不可能有差别；
   - A2、A3 的预热段也走原生路径，同样要求相同。
   就绪调用之后有分歧是预期的，只记录第一次分歧在哪次调用。

金丝雀的 SR 等数字只有 4 集，不作任何判断。

## 3. 自动调度（2026-09-30 起）

用户要求：哪个臂的金丝雀通过，就直接开它的全量。容器里常驻
`/workspace/exp20/orchestrate.sh`（仓库副本 `scripts/exp20/orchestrate.sh`），日志在 `/workspace/exp20/logs/orchestrate.out`。

- **任务顺序**：A1 / A2 / A3 金丝雀 → A1 / A2 / A3 种子 42 → A0 种子 1337 → A1 / A2 / A3 种子 1337。
- **占卡**：每个任务占一张空卡。空卡的标准是没有别的进程、显存占用不到 500 MiB；GPU 0 放宽到 4000 MiB 以下，因为用户同意与上面那个别人的空闲进程共用。
- **金丝雀判定**：由 `scripts/exp20/canary_check.py` 按 §2 的条件判定，结果写到 `logs/canary_<臂>.verdict.json`。
  - 通过的臂接着开全量；没通过的臂，它的全量标成 `state/<任务>.blocked`，不开。
- **状态与续跑**：任务结束（不论退出码）写 `state/<任务>.exit`，不会自动重跑；重启调度脚本会跳过已结束、已跳过和仍在运行的任务。
- **端口与显示号**：第 k 个任务用模型端口 52600+10k、里程计端口 52700+10k、显示号 390+10k，与手动起的 A0（52400 / 52500 / 360）错开。
- **停止**：对 `orchestrate.sh` 的 PID 发 TERM。它停下后，已开的运行照常跑完。

## 4. 分析

- 每个臂对 A0，按同平台、同种子配对：`scripts/tools/paired_closed_loop_bootstrap.py`，2000 次重采样，两种子合并报 ΔSR / ΔSPL / ΔNE。
- 判据只看配对 CI（台账 EXP-20）。论文表的取数规则单独写在台账里，不影响判读。
