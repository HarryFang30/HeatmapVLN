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

## 8. RTX 4090 部署机上重做主案例（C500 停用后；运行记录 5）

机器与容器见 EXP-18 分支的 `docs/ops/deploy_rtx4090.md`：ssh `6024_fjl`，容器 `fjl-habitat`，宿主 `/home/fangjialei` 挂在容器的
`/workspace`，`/data0/dataset` 挂在 `/dataset`，唯一的解释器 `/opt/conda/bin/python`。**只用空卡**（`nvidia-smi`；启动脚本也会拒绝
有计算进程或已用 > 500 MiB 的卡——容器里的 `nvidia-smi` 看得到宿主上所有容器的进程）。下面 `H` 是宿主路径，`C` 是同一目录的容器路径：

```bash
H=/home/fangjialei/exp19_behavior_viz_4090      # 宿主
C=/workspace/exp19_behavior_viz_4090            # 容器
SHA=<本次运行的 commit>
```

**一次性准备**（宿主上，属主是宿主用户）：

```bash
ssh 6024_fjl "mkdir -p $H/logs $H/checks"     # 下面 docker exec -d 的输出都重定向到 $C/logs；目录不在时命令静默不启动
# 源码（容器连不上 GitHub，从本地推过去）
git archive $SHA | ssh 6024_fjl "mkdir -p $H/src_$SHA && tar -x -C $H/src_$SHA && echo $(git rev-parse $SHA) > $H/src_$SHA/.exp19_git_sha"
# 重渲染用的采集器配置：VLN-CE 3b0c5c0（C500 工作树的提交状态），场景链到容器的 /dataset/mp3d
git -C <VLN-CE 克隆> archive 3b0c5c0 | ssh 6024_fjl "mkdir -p $H/support/habitat/VLN-CE && tar -x -C $H/support/habitat/VLN-CE \
  && mkdir -p $H/support/habitat/VLN-CE/data/scene_datasets && ln -sfn /dataset/mp3d $H/support/habitat/VLN-CE/data/scene_datasets/mp3d"
# 论文字体（容器里没有，宿主有）
ssh 6024_fjl "mkdir -p $H/support/fonts && cp /usr/share/fonts/opentype/urw-base35/NimbusSans-*.otf \
  /usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf $H/support/fonts/"
```

**候选文件与集列表**（容器内，纯 CPU，秒级）：

```bash
docker exec fjl-habitat bash -c "cd $C/src_$SHA && /opt/conda/bin/python -m scripts.exp19.rebuild_cases \
  --dataset /workspace/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz --scenes-dir /dataset \
  --out-dir $C/cases --num-gpus 2"
```

默认列出 5 个 #0（F2 那集 500 步，单独一张卡）。

**冒烟门**（1 卡，约 5 分钟）：追踪复跑金丝雀跑过的 zsNo4HB9uLZ_0001，与金丝雀日志逐调用比较。三步分开执行：

```bash
# 1. 列表（1 集）与启动
docker exec fjl-habitat /opt/conda/bin/python -c "import json, os; os.makedirs('$C/cases/smoke_lists', exist_ok=True); \
  json.dump({'cohort_name': 'exp19_smoke', 'episodes': [{'scene_id': 'zsNo4HB9uLZ', 'episode_id': 1}]}, \
  open('$C/cases/smoke_lists/gpu0.json', 'w'))"
docker exec -d fjl-habitat bash -c "mkdir -p $C/logs && cd $C/src_$SHA && EXP19_SRC=\$PWD EXP19_ROOT=$C EXP19_RUN=smoke4090 EXP19_GPUS=4 \
  EXP19_LISTS=$C/cases/smoke_lists bash scripts/exp19/run_rerun_cuda.sh > $C/logs/rerun_smoke4090.out 2>&1"
# 2. 先确认 $C/logs/rerun_smoke4090.out 出现（-d 会吞掉启动 shell 的报错），再等 $C/runs/smoke4090/DONE 出现
#    （launcher 的最后一行是 COMPLETE 或 ERROR）
# 3. 比较（输出写在运行目录之外；运行没完成时工具会拒绝）
docker exec fjl-habitat bash -c "cd $C/src_$SHA && /opt/conda/bin/python -m scripts.exp19.compare_run_to_log \
  --run-dir $C/runs/smoke4090 --out $C/checks/smoke4090_vs_canary.json --require-call0 --require-neutral \
  --client-log /workspace/eval_runs/canary_cuda_seed42/runtime/20260928_212453_1550/logs/client_shard_00.log \
  --progress /workspace/eval_runs/canary_cuda_seed42/workers/shard_00/progress.json"
```

参照文件的 sha256 记在运行记录（5）（日志 `7c077a23…a501d`、`progress.json` `f4cf62a5…83b4`）。

**正式复跑**（2 卡，F2 那张卡约 20 分钟）：

```bash
docker exec -d fjl-habitat bash -c "mkdir -p $C/logs && cd $C/src_$SHA && EXP19_SRC=\$PWD EXP19_ROOT=$C EXP19_RUN=main4090 EXP19_GPUS=4,5 \
  bash scripts/exp19/run_rerun_cuda.sh > $C/logs/rerun_main4090.out 2>&1"
```

**复跑之后**（纯 CPU：俯视图 → 重渲染 → 记录 → 时间线 → 动画）：

```bash
docker exec -d fjl-habitat bash -c "mkdir -p $C/logs && cd $C/src_$SHA && EXP19_ROOT=$C EXP19_RUN=main4090 \
  bash scripts/exp19/run_post_cuda.sh > $C/logs/post_main4090.out 2>&1"
```

输出 `$C/{topdown,renders,records,metrics,records_v2}/` 与 `$C/figures_v2/anim_<run>/`（5 个主案例的 MP4 + GIF，中英，`manifest.json`，
`preview/` 为每段 4 帧 JPEG）。`metrics/` 在结构上是 void（没有主表日志），只作描述。容器以 root 写文件，收尾时在容器里
`chown -R 1015:1015 $C`，宿主用户才能删改。

**顺延**：某类 #0 复跑后不满足谓词时（`run_post_cuda.sh` 的记录阶段会停下并点名），
按预注册顺延到 #1（再不满足则 #2）：

```bash
# 只写顺延集的列表，candidates.json 不动（--num-gpus 不大于顺延集数）
docker exec fjl-habitat bash -c "cd $C/src_$SHA && /opt/conda/bin/python -m scripts.exp19.rebuild_cases --lists-only \
  --dataset /workspace/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz --out-dir $C/cases \
  --lists-dir $C/cases/episode_lists_fb1 --num-gpus 1 --episodes <#1 的 ep_key>"
# 另起一次复跑（EXP19_RUN=main4090_fb1 EXP19_LISTS=$C/cases/episode_lists_fb1），完成后合并读取：
docker exec fjl-habitat bash -c "cd $C/src_$SHA && /opt/conda/bin/python -m scripts.exp19.merge_runs \
  --exp-root $C --out main4090_all main4090 main4090_fb1"
# 之后的 run_post_cuda.sh 用 EXP19_RUN=main4090_all（从头走一遍各阶段；重渲染逐集重新核对输入）
```

## 测试

```bash
cd <private copy> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/rpc/src:$PWD $R/envs/qwen25/bin/python -m pytest tests/test_exp19_*.py -q
```

4090 容器里：`PYTHONPATH=/workspace/rpc/src:$PWD /opt/conda/bin/python -m pytest tests/test_exp19_*.py -q`，并设
`EXP18_FONT_DIR` / `EXP18_CJK_FONT` 指向 `support/fonts`（`test_exp19_figures_v2.py` 里有按字宽判断"放得下"的断言，没有中文字体会误挂一条）。
`test_exp19_trace_server.py` 需要 `$R/rpc/src` 在 `PYTHONPATH` 上，否则整文件跳过。私有副本不要直接放在 `/tmp` 下
（仓库根有 `__init__.py`，pytest 会上溯到 `/tmp/__init__.py`，那里有别人的无关包），放 `/tmp/<dir>/repo`。

## 已知环境问题（2026-09-24）

AFS（FUSE）小文件元数据极慢时：`xdpyinfo` 在 5 s 超时内加载不完（Xvfb 就绪探测改为 TCP 连接）；服务端 import torch
约 1 分钟、transformers 约 3.5 分钟，缓存写 AFS 会让启动超过 1 小时（故开发机上用 `EXP19_RUNTIME_ROOT=/tmp/...`、
启动超时放宽到 3 小时）。大文件顺序读正常（13–65 MB/s）。
