# EXP-19 复现命令（闭环行为可视化）

判据见台账 [README.md](README.md) 的 EXP-19 条目。**开跑前把判据再读一遍，跑完只填结果，不改判据。**

GPU 部分在**开发机**上跑（≤ 3 卡，台账 §0 第 7 条；启动前 `mx-smi` 查卡空闲，启动脚本也会拒绝有进程的卡）；
选集、重渲染、真值、指标与作图都是纯 CPU。代码不从共享检出跑（§5 第 27 条）：把要跑的 commit 原样
`git archive` 到产物根下的独立目录。

```bash
R=/mnt/afs/liwenhao/agent/370910109
EXP=$R/model/exp19_behavior_viz           # 产物根：cases/ runs/ renders/ records/ metrics/ figures/ logs/
SHA=<本次正式运行的 commit>
SRC=$EXP/src_$SHA
```

远程命令一律走 login shell（`CLAUDE.md` §3.1）；后台长任务用 `ssh -n -f … 'bash -lc "setsid nohup … &"'`。

---

## 0. 暂存源码

本地：

```bash
git archive $SHA | ssh finn_cci_c500 "mkdir -p $SRC && tar -x -C $SRC && echo $(git rev-parse $SHA) > $SRC/.exp19_git_sha"
```

（`$(git rev-parse …)` 在本地展开成完整 sha 再写进远端文件；启动脚本要求 `.exp19_git_sha` 存在。）

## 1. 选集（纯 CPU，约 1 分钟）

```bash
ssh finn_cci_c500 'bash -lc "cd $SRC && PYTHONDONTWRITEBYTECODE=1 $R/envs/qwen25/bin/python -m scripts.exp19.select_cases --out-dir $EXP/cases --num-gpus 3"'
```

产物：`cases/candidates.json`（15 个候选、各类池大小与中位数、每条谓词的文字定义、输入文件 sha256）、
`cases/episode_lists/gpu{0,1,2}.json`（客户端 `--episode_list` 格式，按种子 42 步数做最长处理时间优先分片）、
`cases/eval_log_reference/<ep_key>.json`（主表种子 42 客户端日志里该集的逐调用记录，复跑保真与代码等价门用）。
`--self-check` 只核对计数、不写文件。

## 2. 冒烟（1 卡，非候选集）

```bash
cat > $EXP/smoke/smoke_list.json <<'J'
{"cohort_name": "exp19_smoke_noncandidate", "episodes": [{"scene_id": "zsNo4HB9uLZ", "episode_id": 1}]}
J
ssh -n -f finn_cci_c500 'bash -lc "cd $SRC && EXP19_SRC=$SRC EXP19_GPUS=0 EXP19_RUN=smoke3 \
  EXP19_RUNTIME_ROOT=/tmp/exp19_runtime EXP19_SERVER_START_TIMEOUT_S=10800 \
  EXP19_SMOKE_LIST=$EXP/smoke/smoke_list.json setsid nohup bash scripts/exp19/run_smoke.sh > $EXP/logs/smoke3.out 2>&1 < /dev/null &"'
```

冒烟看三件事（都在 `runs/smoke3/gpu0/`）：每次客户端调用都有 `trace/<ep>/call_XXX.json`；
轨迹调用的 `actions_match` 全为真；第 0 次调用的慢系统文本与动作块与主表种子 42 日志逐字相同。

`EXP19_DRY_RUN=1` 先跑一遍很便宜：做全部检查、打印全部命令、不启动任何进程。

## 3. 正式复跑（3 卡）

```bash
ssh -n -f finn_cci_c500 'bash -lc "cd $SRC && EXP19_SRC=$SRC EXP19_GPUS=0,1,2 EXP19_RUN=main \
  EXP19_RUNTIME_ROOT=/tmp/exp19_runtime EXP19_SERVER_START_TIMEOUT_S=10800 \
  setsid nohup bash scripts/exp19/run_rerun.sh > $EXP/logs/rerun_main.out 2>&1 < /dev/null &"'
```

每卡一组：Xvfb（`:380+j`）+ 追踪模型服务（`scripts/exp19/rpc_model_server_trace.py`，端口 `52640+j`）+ AMB3R VO 服务
（`52740+j`）+ 客户端（与主表评测逐项同参，另加 `--step_state_trace_dir`，**绝不加** `--save_trajectory_steps`）。
完成后写 `runs/main/DONE`（各客户端退出码与逐集完整性检查）。运行名不可复用：MACA 前向不可重复，一次运行从不续跑或覆盖。

网站提交（空白容器）时去掉 `EXP19_RUNTIME_ROOT`（缓存落在运行目录里）：

```bash
cd /mnt/afs/liwenhao/agent/370910109/model/exp19_behavior_viz/src_<sha>
export EXP19_SRC=$PWD EXP19_RUN=main EXP19_GPUS=0,1,2 EXP19_SERVER_START_TIMEOUT_S=10800
bash scripts/exp19/run_rerun.sh
```

## 4. 重渲染（纯 CPU，Xvfb + llvmpipe）

```bash
ssh -n -f finn_cci_c500 'bash -lc "cd $SRC && EXP19_RUN=main setsid nohup bash scripts/exp19/run_render.sh > $EXP/logs/render_main.out 2>&1 < /dev/null &"'
```

先跑强制自检（在一个 `r2r_panoramic_data_v2` clip 的存储位姿上重渲染，深度须逐帧一致），再把每集每个调用步的真值状态
渲染成 4 视角 256×256、HFOV 90 的 RGB 与前视深度（`renders/<ep_key>.{npz,json}`）。

## 5. 记录、真值、指标与判定（纯 CPU）

```bash
ssh finn_cci_c500 'bash -lc "cd $SRC && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC $R/envs/qwen25/bin/python -m scripts.exp19.build_records --run main"'
```

产物：`records/<ep_key>.json`（逐调用量、保真、关键时刻）、`records/<ep_key>_bundle.{json,npz}`（作图唯一输入）、
`metrics/{metrics.json, summary.md, calls.jsonl}`。判定（H1/H2/H3）与有效性门写在 `metrics.json` 的 `verdicts` / `validity`，
阈值与台账逐字对应。退出码：3 = 有效性门不过（全部判定 `void`），4 = H1 与 `validate.py` 累加器自检不一致。

## 6. 作图（纯 CPU，开发机字体）

```bash
ssh finn_cci_c500 'bash -lc "cd $SRC && EXP19_FIG_MAIN=1 bash scripts/exp19/run_figures.sh"'
```

输出 `figures/<类>/<序号>_<ep_key>_{en,zh}.{pdf,png}` + caption、`figures/main/main_{T,F}_{en,zh}.*`、`figures/manifest.json`。
图注里关于 H1/H2/H3 的话只由 `metrics.json` 的判定生成（预注册措辞规则）；批次无效时拒绝出图（`EXP19_FIG_ALLOW_VOID=1` 才出，且每页盖"VOID BATCH"）。

**作图口径**：图里不出现位姿、VO、里程计或位姿臂对比；热力统一叫 affordance map；左 / 右 / 后扇区压灰注明"未输入模型（仅展示）"；
俯视图不画朝向箭头；数据流不画"未来图 → 动作"。

## 7. 图 v2（论文级改版：在线时间线 + 动画；运行记录 4）

```bash
# 在线时间线数据（呈现用，写 records_v2/，不改 records/、metrics/）
ssh finn_cci_c500 'bash -lc "cd $SRC && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC $R/envs/qwen25/bin/python -m scripts.exp19.build_timeline --run main --self-check"'
# 逐集页 + 主图 T/F（+ EXP19_FIG_ANIM=1 时生成主案例 MP4/GIF 动画；动画需要开发机的 /opt/conda/bin/ffmpeg）
ssh -n -f finn_cci_c500 'bash -lc "cd $SRC && EXP19_FIG_ANIM=1 setsid nohup bash scripts/exp19/run_figures_v2.sh > $EXP/logs/figures_v2.out 2>&1 < /dev/null &"'
```

输出 `figures_v2/`（逐集页、`main/`、`anim/`、`manifest.json`）。热力场为显示做了高斯平滑（环视条带 σ = 2°、时间线方位向 σ = 3.5°，
按峰值重标），圆点标的是未平滑的峰值；图注已写明。2026-09-25 的正式 v2 输出由改版后的工作树代码在开发机本地盘渲染、
逐文件 sha256 校验后拷入 `figures_v2/`（当时 AFS 满盘）。

## 测试

```bash
cd <private copy> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/rpc/src:$PWD $R/envs/qwen25/bin/python -m pytest tests/test_exp19_*.py -q
```

`test_exp19_trace_server.py` 需要 `$R/rpc/src` 在 `PYTHONPATH` 上，否则整文件跳过。私有副本不要直接放在 `/tmp` 下
（仓库根有 `__init__.py`，pytest 会上溯到 `/tmp/__init__.py`，那里有别人的无关包），放 `/tmp/<dir>/repo`。

## 已知环境问题（2026-09-24）

AFS（FUSE）小文件元数据极慢时：`xdpyinfo` 在 5 s 超时内加载不完（Xvfb 就绪探测改为 TCP 连接）；服务端 import torch
约 1 分钟、transformers 约 3.5 分钟，缓存写 AFS 会让启动超过 1 小时（故开发机上用 `EXP19_RUNTIME_ROOT=/tmp/...`、
启动超时放宽到 3 小时）。大文件顺序读正常（13–65 MB/s）。
