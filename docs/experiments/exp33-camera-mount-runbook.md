# EXP-33 Go2 相机位姿：复现命令（RTX 4090，容器 `fjl-habitat`）

判据与设置见台账 [EXP-33](README.md#exp-33-go2-装得上的相机位姿低机位下倾单相机代俯视仿真里掉多少)。下面都是容器路径（宿主机 `/home/fangjialei` = 容器 `/workspace`）。

## 开关

`scripts/run_ppa_r2r_val_unseen_cuda.sh` 新增三个变量，只影响客户端，服务端不变：

| 变量 | 客户端参数 | 不设时 |
|---|---|---|
| `PPA_EVAL_CAMERA_HEIGHT_M` | `--camera_height_m` | 1.25 m |
| `PPA_EVAL_CAMERA_PITCH_DEG` | `--camera_pitch_deg`（负 = 下倾） | 0° |
| `PPA_EVAL_LOOKDOWN_MODE` | `--lookdown_mode rotate\|same_frame` | `rotate` |

- 三个都不设时，启动脚本不传任何新参数。
- 设了任意一个，日志多一行 `camera_height_m=… camera_pitch_deg=… lookdown_mode=…`。
  客户端打印 `camera mount (EXP-33): {...}`，每集开始核对传感器的实际高度与俯仰，每行 `progress` 带 `camera_mount`。
- 续跑时，目录里行的 `camera_mount` 与本次设置不一致就拒绝（两个方向都拒）。

## 代码副本

和 EXP-21 一样，用 `git archive` 导出的副本跑，不动部署检出。本地：

```bash
git archive --format=tar.gz -o /tmp/exp33_src.tgz <commit>
scp /tmp/exp33_src.tgz 6024_fjl:/home/fangjialei/
```

容器里：

```bash
mkdir -p /workspace/exp33/src_<commit7> && tar -xzf /workspace/exp33_src.tgz -C /workspace/exp33/src_<commit7>
```

## 渲染核验（不占卡）

```bash
cd /workspace/exp33/src_<commit7>
Xvfb :690 -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac &  # 加 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
DISPLAY=127.0.0.1:690.0 CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
  HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1 /opt/conda/bin/python scripts/exp33/render_check.py \
  --data_path /workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/cohorts/dataset_shard_00.json.gz \
  --out /workspace/exp33/render_check
```

结束后按 PID 停掉 Xvfb。

## 调度

`scripts/exp33/orchestrate.sh`，在容器里 `nohup setsid` 常驻：

- 8 个任务：A0 的 V0、V1、V3、V2，然后 A1 同序。
- 每个任务占一张空卡（显存不到 1000 MiB；GPU 0 放宽到 6000 MiB 以下），范围限在 `ORCH_GPUS` 内。判空和启动在同一轮里做。
- 端口与显示号：第 k 个任务用模型端口 53100+10k、里程计端口 53200+10k、显示号 600+10k，与 EXP-21、EXP-32 错开。
- `/workspace/exp33/HOLD` 存在时不起新任务；`ORCH_START_WHEN` 命令失败时也不起。正在跑的任务不受影响。
- 日志在 `/workspace/exp33/logs/`，状态在 `/workspace/exp33/state/`。结束的任务不自动重跑；重启调度脚本会跳过已结束和仍在跑的任务。
- 输出：`/workspace/eval_runs/exp33_<a0|a1>_<v0…v3>_seed42_4090/`。

```bash
cd /workspace/exp33 && ORCH_SRC=/workspace/exp33/src_<commit7> ORCH_GPUS="0 4 5" \
  ORCH_START_WHEN='grep -q " END" /workspace/exp32/auto/STATUS' \
  nohup setsid bash /workspace/exp33/src_<commit7>/scripts/exp33/orchestrate.sh > /workspace/exp33/logs/orchestrate.out 2>&1 &
```

停止：先 `touch /workspace/exp33/HOLD`，再按 `state/<任务>.running` 里的 PID 停。
那个 PID 是包着启动脚本的子 shell，启动脚本自己的 PID 用 `ps --ppid <它>` 找。对启动脚本发 TERM，它的清理函数会停掉本次的服务端和 Xvfb。

## 分析

```bash
cd /workspace/exp33/src_<commit7> && /opt/conda/bin/python scripts/exp33/analyze.py \
  --runs /workspace/eval_runs --subset /workspace/exp21/subset500 \
  --cohorts /workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/cohorts \
  --reference /workspace/eval_runs/exp20_a0_seed42_4090/merged/progress.jsonl --out /workspace/exp33/analysis
```

- 某次运行不满足有效性条件（集不全、`camera_mount` 不对、不是 AMB3R 位姿、不是种子 42、桥从未生效），脚本直接拒绝。
- 中途读数：加 `--only a0_v0 a0_v1 …`。只有 8 次运行齐全时才给 H1–H3 的判定。
