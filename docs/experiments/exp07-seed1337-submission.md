# EXP-07 提交物：主表第二个协议种子（1337），两臂全量 1839 集

台账条目见 [README.md → EXP-07](README.md#exp-07-主表第二个种子parity--ne-是否可复现长路径增益是否存在)。判据在那里，开跑前已 commit；本文件只放复现命令。

两臂只改协议种子（42 → 1337），其余与种子 42 的两次认证评测完全一致。种子决定 NextDiT 采样噪声（`heatmapvln-nextdit-sha256-v1` 协议下按 episode/step 派生），不影响 System2 贪心解码。

## 臂 1：完整方法（修复栈，桥在线）

`scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh` 自 2026-09-03 起把种子和 arm 标签暴露为
`PPA_EVAL_PROTOCOL_SEED` / `PPA_EVAL_ARM`（默认 42 / `ppa_stage2_online_amb3r`，即 EXP-01 的用法）。
第二个种子必须用**新的输出根**：脚本按 `--resume` 续跑，同一目录混两个种子会被当成已完成的 episode 跳过。

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export PPA_EVAL_CHECKPOINT=/mnt/afs/liwenhao/agent/370910109/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth
export PPA_EVAL_CONFIG=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/configs/ppa_action_refine_v2_8gpu.yaml
export PPA_EVAL_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu_seed1337
export PPA_EVAL_PROTOCOL_SEED=1337
export PPA_EVAL_GPU_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh
```

成功标志同 EXP-05：`[ppa-eval] COMPLETE` + 自检 `"status": "passed"`。

## 臂 2：native（认证复刻栈，锁定代码副本）

认证复刻栈不在 git 里，且它的 `manifests/runtime_code.sha256` 把 4 个 HeatmapVLN 运行时文件
锁在 2026-08-02 的版本上；其中 3 个此后改过（`rpc_protocol.py` 协议号 v2→v3、
`input_constructor.py`、`scripts/training/utils.py`），启动脚本还断言协议号必须是 v2。
所以 native 臂**不能跑在现行 HEAD 上**，做法是（已在共享存储上准备好，未运行）：

- 锁定代码：`HeatmapVLN_native_lock_bd5ead1/` = `git archive bd5ead1`（2026-07-16，认证运行前最后一个
  commit）。`navigation_metrics.py` / `rpc_protocol.py` / `input_constructor.py` 三个文件与
  锁定 hash **逐字节相同**；`scripts/training/utils.py` 的锁定版本是当时服务器工作区里未提交的状态
  （后来并入 e6bcf18），git 里没有——bd5ead1 版本与之的差异只在 LoRA 张量计数与训练损失构造器，
  不在 native 评测路径上；该行 hash 已按 bd5ead1 版本重算（**边界**，写进台账）。
- plan 副本：`evaluation_plans/internnav_native_r2r_val_unseen_8gpu_seed1337/`，tools/configs/README
  与金参照逐字节相同（plan_code 12 项 hash 全部通过），`cohorts/` 由 `build_shards.py` 在运行时
  确定性重建。启动脚本相对金参照只有 5 行差异：

```diff
-FJL_ROOT=/mnt/afs/lixiaoou/intern/fjl
-REPO="$FJL_ROOT/HeatmapVLN"
+FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
+REPO="/mnt/afs/liwenhao/agent/370910109/HeatmapVLN_native_lock_bd5ead1"
-PLAN="$FJL_ROOT/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802"
+PLAN="/mnt/afs/liwenhao/agent/370910109/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_seed1337"
-OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$FJL_ROOT/model/eval_internnav_native_r2r_val_unseen_8gpu_rpcv2_x11bundle_v4}"
+OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$FJL_ROOT/model/eval_internnav_native_r2r_val_unseen_8gpu_rpcv2_x11bundle_v4_seed1337}"
-PROTOCOL_SEED=42
+PROTOCOL_SEED="${EVAL_PROTOCOL_SEED:-1337}"
```

- 两份 manifest 的路径已改到新根/锁定目录，hash 除上述两处（启动脚本自身、utils.py）外原样保留；
  `sha256sum -c` 两份全部通过。金参照目录一个字节未动（mtime 仍为 2026-08-02）。
- 导入检查已用启动脚本同款环境跑过：server/client 均从副本目录导入，`rpc_protocol` 解析到锁定
  目录且协议号为 v2，`sys.modules` 里没有任何模块来自现行 `HeatmapVLN/`。

```bash
cd /mnt/afs/liwenhao/agent/370910109/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_seed1337

export EVAL_PROTOCOL_SEED=1337
export EVAL_GPU_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_8gpu_rpc_eval.sh
```

启动脚本先做静态预检 + 8 路各 1 集的 smoke，再跑全量；成功标志 `[eval] COMPLETE`，结果在
`model/eval_internnav_native_r2r_val_unseen_8gpu_rpcv2_x11bundle_v4_seed1337/merged/result.json`。

### 2026-09-04：首次提交失败与修复（已修，命令不变）

首次提交在静态预检阶段就退出（输出根只建了空的 `runtime/logs`，没有 `eval_contract.json`）。
原因：plan 副本里 **`manifests/internnav_model.sha256` 仍是迁移前的旧路径**——我当时只重写了
`runtime_code.sha256` 和 `plan_code.sha256`，漏了这份模型 manifest，于是
`validate_inputs.py` 报 closure mismatch（14 个文件"missing 新路径 / extra 旧路径"）。

修复只改路径、**不动任何 hash**，并逐项验证过：

| 检查 | 结果 |
|---|---|
| `sha256sum -c internnav_model.sha256`（改路径后，读 ~16 GB） | 14/14 全部 OK —— 模型闭包与认证运行逐字节相同 |
| `plan_code.sha256` / `runtime_code.sha256` | 全部 OK（模型 manifest 的条目已同步重算） |
| `validate_inputs.py` | `"status": "passed"`，`tensor_count: 1338` |
| `build_shards.py` | `union_exact: true`，1839 集；shard 00/03/07 与金参照**逐字节相同** |
| server / client 导入断言（协议 v2） | 均通过 |
| 金参照目录 | 未改动（mtime 仍为 2026-08-02） |

> 金参照自己的 `plan_code.sha256` 现在 `sha256sum -c` 一条也过不了——它列的是迁移前的旧路径，
> 文件已不存在。这正是副本必须重写路径的原因，不是金参照被动过。

上面的提交命令原样重提即可（预检产物 `cohorts/`、`input_validation.json` 会被幂等重建；
失败运行留下的 `.eval.lock` 是空文件，flock 不受影响）。

### 2026-09-04 第二次失败：容器里没有 `git`（已修，命令仍不变）

第二次仍在预检退出，网站日志给出了确切位置：

```
[eval] locked runtime-code hashes passed
[eval] locked evaluation-plan hashes passed
X11 host ABI compatible: architecture=x86_64, glibc 2.35
[eval] X11/Mesa bundle hash verification passed
Traceback ... validate_inputs.py line 139, in main
    source_commit = git_head(args.internnav_repo)
FileNotFoundError: [Errno 2] No such file or directory: 'git'
```

`validate_inputs.py` 用 `subprocess` 调 `git` 验证 InternNav 源码 commit 与工作树干净；
**空白容器里没有 git**（开发机上有 `/usr/bin/git`，所以开发机复现不出来）。

**为什么不能"用 conda 环境的 git"**：`envs/qwen25`、`envs/vlnce` 都没有 git，`/opt/conda` 里也没有
（`/opt/conda/bin/git` 与 `pkgs/git-*` 都不存在）；而且按 `CLAUDE.md` §1.1，容器只保证挂载
`/mnt/afs` 与 `/opt/maca-*`，**`/opt/conda` 在容器里本来就不可见**。所以唯一可行的方向是把 git
放到共享存储上——即本 plan 早已在用的 X11 bundle 那套办法。

**做法**：`$PLAN/bin/` 里放一个自包含 git（wrapper + 3.7 MB 二进制 + libpcre2/libz，glibc 用宿主的，
本 plan 已经为 X11 bundle 卡了 ≥ 2.35 这道门），启动脚本前置 `export PATH="$PLAN/bin:$PATH"`，
**只多这一条命令**。验证：

| 检查 | 结果 |
|---|---|
| `env -i PATH=$PLAN/bin git --version` | `git version 2.34.1`（空环境、PATH 只有 bundle 也能跑） |
| 同上跑 `rev-parse HEAD` | `7a5c624…` = `EXPECTED_SOURCE_COMMIT` |
| 同上跑 `status --porcelain` | 空（工作树干净） |
| `validate_inputs.py`（PATH 里**没有**系统 git） | `"status": "passed"`，`tensor_count: 1338` |
| 三份 manifest 复验 | 16 / 30 / 14 全 OK（新增的 4 个 bundle 文件也已 pin 进 `plan_code`） |

副本启动脚本现在相对金参照共 3 处改动：5 行路径/种子、1 行 PATH 前置、1 段自记录
（`runtime/<stamp>/logs/launcher.log`，下次再失败我直接读它，不用再找网站日志）。

## 跑完后的判读

对每个种子分别、再对两种子合并（把两个 progress.jsonl 各自配对后拼接）跑
`scripts/tools/paired_closed_loop_bootstrap.py --geodesic-min 10`，按台账 EXP-07 的六档判据读。
