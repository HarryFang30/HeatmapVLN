# 部署在 RTX 4090 服务器（ssh 别名 `6024_fjl`）

2026-09-28 部署并跑通。机器是多人共用的：6 张 RTX 4090（48 GB），Ubuntu 24.04，CUDA 12.8 驱动，没有 conda；环境在 docker 容器 `fjl-habitat` 里。**开跑前先看 `nvidia-smi`，只用空着的卡**（部署时 1–3 号卡有别人的进程）。

## 1. 布局

容器把 `/home/fangjialei` 挂在 `/workspace`，把 `/data0/dataset` 挂在 `/dataset`。下表都是容器内路径。

| 物 | 路径 | 来源 |
|---|---|---|
| 代码 | `/workspace/HeatmapVLN` | 本仓库（部署时 `fc836bf` 起） |
| RPC 工具 `vla_rpc` | `/workspace/rpc` | 迁移包 `rpc_repo_with_git.tgz`：`21cc2fc` + 未提交改动，内容与 `b35e060` 相同 |
| AMB3R | `/workspace/amb3r` | 上游 `92c4081`（codeload 源码包）+ `amb3r_local.patch` + `amb3r_untracked.tgz`，各一个本地提交 |
| DA3 权重 | `/workspace/amb3r/checkpoints/DA3NESTED-GIANT-LARGE` | hf-mirror `depth-anything/DA3NESTED-GIANT-LARGE@8615eef`，`model.safetensors` sha256 `8899faf9…2ddd` |
| 慢系统 / 快系统底座 | `/workspace/InternNav_Model` | InternVLA-N1 发布权重（完整版） |
| 部署权重 | `/workspace/weights/ppa_refine_v2_best.pth` | sha256 `0b5a0644…6d69`（v2 信赖域桥） |
| 锁定评测计划 | `/workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802` | 迁移包 `evaluation_plans_all.tgz` |
| R2R val_unseen | `/workspace/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz` | sha256 `1767a407…d167c3`，与锁定计划一致 |
| MP3D 场景 | `/dataset/mp3d` | 11 个 val_unseen 场景齐全 |

**Python**：容器里只有一个 `/opt/conda/bin/python`（3.11）。它同时装了 torch 2.11+cu128、transformers 4.51.0（`runtime_compat` 要求必须是这个版本）、habitat-sim 0.1.7 和可编辑安装的 habitat-lab 0.1.7（源码就是 `/workspace/habitat-lab`，**不能删**），所以服务端和客户端共用这一个解释器。

部署时补装了两个包：`utils3d 0.0.2`（AMB3R 自带 wheel）和 `addict 2.4.0`（阿里云镜像）。

## 2. 跑评测

在宿主机上：

```bash
docker start fjl-habitat
```

然后进入容器跑启动脚本（下面是全量、种子 42、用 4、5 号卡）：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/HeatmapVLN && PPA_EVAL_GPU_DEVICES=4,5 PPA_EVAL_PROTOCOL_SEED=42 bash scripts/run_ppa_r2r_val_unseen_cuda.sh'
```

- 协议和客户端参数与 C500 上认证过的 8 卡评测相同：同一份 8 分片锁定队列、确定性采样、在线 AMB3R、llvmpipe CPU 渲染。
- 可以用 1–8 张卡。8 个分片轮流分给各张卡，每张卡依次跑自己分到的分片。
- 模型服务端和里程计服务端默认放在同一张卡上，实测空载约 36 GB、运行中见到最高约 37 GB（48 GB 的卡）。显存不够时用 `PPA_EVAL_VO_GPU_DEVICES` 把里程计放到别的卡。
- 冒烟 / 金丝雀：加 `PPA_EVAL_SHARDS=0,1 PPA_EVAL_MAX_EPISODES_PER_SHARD=2`，只打印汇总，不合并。
  **带上限的运行每次都要换一个空的 `PPA_EVAL_OUTPUT_ROOT`，而且不能开 `PPA_EVAL_SHARD_RETRIES`。**
  上限是对"新跑的集"计数的（`r2r_val_unseen.py` 的 `_eval_limit`），`--resume` 又会跳过已跑的集，
  所以往已有结果的目录里再跑一次，是**另外**再跑 2 集新的——金丝雀就不是原来那 4 集了，而
  `progress.json` 里看不出哪几行是后加的。现在这两种情况启动脚本会直接拒绝（exit 2），
  真要续跑就显式加 `PPA_EVAL_ALLOW_CAPPED_RESUME=1`。要固定具体哪几集，用
  `scripts/tools/make_episode_lists_from_run.py` 把参照运行的集钉成集表，配
  `PPA_EVAL_EPISODE_LISTS_DIR` 用，然后不设上限（见 `deploy_ascend_910b.md` §6）。
- 输出默认在 `/workspace/eval_runs/ppa_refine_v2_seed<种子>/`（金丝雀在 `canary_seed<种子>/`）。**换种子一定换输出目录**，否则 `--resume` 会跳过另一个种子已跑的集。
- 跑满 8 个分片且不设上限时，会用锁定计划的 `merge_shards.py` 合并并自检，最后打印 `"status": "passed"`。
- **不设上限的运行（含认证全量）跑完后还会核对一遍"记录下来的集"与"集表里的集"完全一致**：少一集、多一集、或同一集出现两行，启动脚本就失败。合并那条路本来就有这个自检，而不合并的子集运行以前**没有**——那是唯一一种能短一集或多一集还打印 `passed` 的跑法。
- 长任务在宿主机上用 tmux 或 `docker exec -d … > log 2>&1` 挂起，ssh 断开不会影响。

**速度**：每张卡约 1–1.3 秒一步（CPU 渲染 + 在线建图）。全量 1839 集在 2 张卡上大约一天。

## 3. 部署验证（2026-09-28）

- 模型服务端单独启动约 40 秒。日志里有 `Formal PPA online AMB3R runtime enabled … tensors={'heatmap': 79, 'future': 11, 'bridge': 10}`。
- 金丝雀：4、5 号卡两个槽位，分片 0、1 各 2 集。结果：4 集全部成功，SPL 87.0%，NE 0.91 m，注入生效 52 次，位姿全部来自 `amb3r_vo_da3`。

| 分片 | 集 | 成功 | SPL | NE (m) | 步数 | 注入生效 |
|---|---|---|---|---|---|---|
| 0 | zsNo4HB9uLZ / 1 | 1 | 0.995 | 0.27 | 49 | 7 |
| 0 | zsNo4HB9uLZ / 25 | 1 | 0.820 | 1.94 | 105 | 17 |
| 1 | zsNo4HB9uLZ / 2 | 1 | 0.741 | 0.93 | 89 | 12 |
| 1 | zsNo4HB9uLZ / 26 | 1 | 0.924 | 0.51 | 94 | 16 |

- **与 C500 的数字不会逐集相同**。快系统噪声由 CUDA 生成器按调用播种，bf16 算子也因平台而异。要判断部署是否等价，就跑全量两种子，与台账 §4 的 62.81% / 61.17% 比。

## 4. 这台机器的坑

- **仓库里大量文件属主是 root**（以前在容器里以 root 跑出来的），包括 `.git/index`，所以 git 操作要在容器里做：`git -c safe.directory='*' …`。
- **容器连不上 GitHub**，宿主机能连但很慢。更新代码的办法：在本地 `git bundle create x.bundle <旧提交>..<新分支>`，scp 上去，在容器里 `git fetch <bundle> <ref>:<ref>` 再快进。
- **HuggingFace 不通**，下载权重用 `https://hf-mirror.com/<repo>/resolve/<commit>/<file>`；pip 用阿里云镜像。
- 容器的 `.bashrc` 设了 `127.0.0.1:7890` 的代理，但代理并不存在。登录 shell 里装包或下载前先 `unset http_proxy https_proxy`（启动脚本已经处理）。
- 容器是 Ubuntu 20.04，bash 5.0，**没有 `wait -n -p`**，所以不能直接用 C500 的启动脚本。
- **`/workspace/habitat-sim` 不能删也不能挪。** site-packages 里的 `habitat_sim/_ext/*.so` 的 RUNPATH 写死为 `/workspace/habitat-sim/build/lib.linux-x86_64-cpython-311/habitat_sim/_ext`，Corrade 等库要从那里加载。真正运行时要用的只有这个 319 MB 的目录；其余的 `.git`（1.6 GB）和编译中间文件（约 2.7 GB）理论上能删，但没有验证过。
- habitat-sim 是 GLX 版，必须有 X 服务。启动脚本为每张卡起一个 Xvfb，并按 TCP 探测就绪。NVIDIA GLX 渲染在这台机器上与 numba 冲突（`troubleshooting-guide.md` §12），所以默认走 llvmpipe。
- 在容器里用 `pkill -f <模式>` 时，模式会匹配到 `bash -c` 自己那一行，把自己的 shell 杀掉。停进程请用 PID。

## 5. 实时性测试（逐阶段计时）

默认关闭。关闭时服务端和客户端不读时钟、不做 CUDA 同步、响应不多任何字段、不写任何文件，请求和动作与部署逐字节相同。

打开：启动脚本加 `PPA_EVAL_TIMING=1`，脚本给模型服务端、里程计服务端和客户端都导出 `HEATMAPVLN_TIMING=1`（单独起服务端时也可以加 `--timing`）。例（模型 4 号卡、里程计 5 号卡，分片 0、1 各 2 集，输出放新目录）：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/HeatmapVLN && PPA_EVAL_TIMING=1 PPA_EVAL_GPU_DEVICES=4 PPA_EVAL_VO_GPU_DEVICES=5 PPA_EVAL_SHARDS=0,1 PPA_EVAL_MAX_EPISODES_PER_SHARD=2 PPA_EVAL_OUTPUT_ROOT=/workspace/eval_runs/latency_seed42 bash scripts/run_ppa_r2r_val_unseen_cuda.sh'
```

- 上面这条命令的输出目录是固定的 `latency_seed42`，所以**第二次跑之前要先把它清掉或换个名字**：
  带上限的运行往已有结果的目录里续跑会被拒绝（见 §2 那条）。否则它会悄悄给你另外 2 集，
  计时统计也就不是同一个样本了。
- 服务端每个阶段前后各做一次 `torch.cuda.synchronize`（同步的是服务端自己用的那张卡：模型的 `--gpu_id`、里程计的 `--device`），量到的是 GPU 真正算完的时间。同步只挪了主机等待的位置，不改任何张量，所以动作应当不变，但整体会慢一点。
- **"动作不变"要先在 GPU 上实测一次**：用 09-28 金丝雀的配置（4、5 号卡两个槽位，分片 0、1 各 2 集，种子 42）加 `PPA_EVAL_TIMING=1` 跑到新目录，用 `scripts/exp19/select_cases.py` 的 `parse_client_log` 读两边的客户端日志、`scripts/exp19/build_records.py` 的 `compare_calls` 逐调用比：每次调用的步号、`kind`、慢系统原文、动作块都相同，§3 表里的步数、注入生效次数、SR/SPL/NE 也相同，才算通过。
  - 通过：带计时的全量评测的 SR/SPL 可以直接当基线用，全量基线和延迟表一次跑出来。
  - 不通过：计时改变了行为，延迟数字一个都不报，SR/SPL 只认不计时的跑法，先查原因。
- 模型和里程计默认同卡，会互相抢 GPU。要干净的数字，用 `PPA_EVAL_VO_GPU_DEVICES` 把里程计放到另一张卡（上例）。
- 渲染走 llvmpipe（CPU），env.step 和全景采集都慢。这是模拟器开销，汇总里单列，**不算模型延迟**。
- 换一个输出目录再跑：`--resume` 会跳过已跑的集，目录里旧的计时日志也还在。启动脚本只汇总本次运行新写的文件。
- 全量 8 分片带计时也能正常合并：锁定计划的 `merge_shards.py` 只读每个分片的 `progress.json` 和 `result.json`（`tools/merge_shards.py:133-134`），不遍历分片目录，多出的 `timing/` 不影响它。

### 输出

- 客户端：`<输出目录>/workers/shard_0X/timing/client_<时间>_<pid>.jsonl`，每次规划调用一行（schema `heatmapvln-latency-v1`）。一行管一次调用加上它返回的动作块执行完为止，在下一次调用开始或该集结束时写出。
- 两个服务端的响应都多两个字段（客户端的校验不拒收多出的键）：
  - `timing_ms`：下表各阶段，毫秒。
  - `cuda_memory_mib`（只在 CUDA 上有）：`peak_allocated`（本进程张量）和 `peak_reserved`（本进程缓存分配器占的池），都是本次请求内的峰值，每个请求前清零；`device_used` 是请求结束时整张卡已用的显存，含所有进程和 CUDA context。
  - 客户端日志里模型的记在 `model_cuda_mib`，里程计的记在 `vo_rpc` 每一项的 `server_cuda_mib`。
- **报部署显存看 `device_used`。** `peak_allocated` 只是 PyTorch 张量，比实际占用小很多：不含缓存池、CUDA context，也不含另一个服务端。
  - 模型和里程计同卡时，`device_used` 就是两者合计（§2 实测空载约 36 GB）。
  - 分卡时把两张卡的 `device_used` 相加。
  - 卡上有别人的进程时，`device_used` 会把它们也算进去，这时改用两个服务端各自的 `peak_reserved` 相加，再每个进程加几百 MB 的 context。
- 跑完自动汇总到 `<输出目录>/runtime/<时间戳>/latency/latency_summary.{md,json}`。手动汇总（目录或文件都行，同一次调用出现两次时取文件顺序里最后一行）：

```bash
python scripts/tools/summarize_latency.py <输出目录>/workers --output-dir <汇总目录>
```

### 各阶段

模型服务端（`scripts/evaluation/rpc_model_server.py`，`timing_ms` 的键）：

| 键 | 内容 |
|---|---|
| `request_decode` | JSON 解析 + 每张 JPEG 的解码和缩放 |
| `system2_turn1_prep` | 慢系统第一轮：拼提示、chat template、分词和图像预处理、搬上 GPU |
| `system2_turn1_generate` | 第一轮贪心解码（最多 128 个新 token） |
| `system2_turn2_prep` / `system2_turn2_generate` | 第一轮答 `↓` 时的第二轮（加俯视图） |
| `ppa_history_memory` | 历史头：历史帧 + 当前帧 → 记忆 token（只在位姿就绪时） |
| `system1_condition_latents` | 带 latent query 的前向，得到快系统条件 |
| `ppa_bridge` | cond_projector + 桥，得到修正后的条件（只在位姿就绪时） |
| `system1_nextdit_sampling` | NextDiT 采样（32 条 × 10 步） |
| `trajectory_to_actions` | 平均轨迹 → 离散动作（含 STOP→LEFT） |
| `future_heatmap_diagnostics` | 未来头，只做诊断，不影响动作（只在位姿就绪时） |
| `handler_total` | 整个请求，到序列化响应之前 |

里程计服务端（`scripts/amb3r_vo/rpc_amb3r_vo_server.py`）：`jpeg_decode`；`ingest`（只存帧）、`ingest_map_init`（第 20 帧建图）或 `ingest_map_update`（每 8 帧扩图）；`query` 或 `query_map_update`（查询前先把没建图的尾巴建上）；`total`。

客户端（`scripts/evaluation/r2r_val_unseen.py`，墙钟）：
- `plan_ms`（每次调用）：`pano_capture`、`vo_query`、`lookdown_capture`、`model_encode`（请求里全部 JPEG 的编码）、`model_rpc`（模型 RPC 往返）。
- `step_ms`（动作块执行期间，每次一个值）：`env_step`（每个动作一次）、`pano_capture`（排队动作前的历史全景）、`vo_ingest`（每个新帧）。
- `vo_rpc`：同一时段里每次里程计 RPC 的往返和服务端 `timing_ms`；`cycle_wall_ms`：整段墙钟，含没计时的客户端开销。

### 汇总怎么读

按每次调用在模型服务端实际走的路径分组（看响应的 `kind` 和 `ppa_applied`），再报一个合计。只看 `pose_ready` 分组是错的：慢系统直接出箭头或停止的调用根本不跑快系统。09-28 金丝雀里，就绪调用有 29%、预热调用有 50% 是这种。

| 组 | 条件 | 跑了什么 |
|---|---|---|
| `ppa` | `kind` 为 `trajectory` 且 `ppa_applied` | 慢系统 + 历史头 + 桥 + 快系统 + 未来头 |
| `native_system1` | `kind` 为 `trajectory`，没注入 | 慢系统 + 原生快系统（部署里就是前 20 帧的建图预热） |
| `system2_only` | 其他 `kind`（`native_actions`、`stop`、`fallback_stop`） | 只有慢系统 |
| `all` | 全部 | 用来摊到每个动作 |

- **相对 native 的额外开销**：直接读 `ppa` 组的 `ppa_history_memory` + `ppa_bridge`，再加里程计一栏。`future_heatmap_diagnostics` 只做诊断、不影响动作，单列说明。
- 不要用 `ppa` 组减 `native_system1` 组：预热调用集中在每集开头，慢系统提示里的历史帧更少，两组的慢系统耗时本来就不同。

每组有：

- **每次调用**：`model`（编码 + 模型往返）、`vo`（位姿查询 + 这段里所有帧的写入）、`simulator`（全景、俯视图、env.step）、三者之和 `timed_total`、实测 `cycle_wall` 和差值 `untimed`；`plan_latency` 是从决定重规划到拿到动作的关键路径（只算调用侧各阶段）；`model_server` 是服务端 `handler_total`，`model_rpc_overhead` 是往返减去它（传输和序列化）。
- **每个动作（摊销）**：`model`、`vo`、`simulator`、`timed_total`、`cycle_wall` 各自除以这次调用的动作块实际执行的动作数（没执行动作的调用不计）；`pooled` = 所有调用之和 / 总动作数。
- 模型服务端、里程计服务端、客户端的逐阶段统计。
- 显存：两个服务端的 `peak_allocated`、`peak_reserved`、`device_used`，每项给 n / 中位数 / 最大值。报最大值。

### 计时中立性验证（预注册，2026-09-30，跑之前写）

- **设置**：在单卡 GPU 4 上重跑 09-28 金丝雀的 4 集（分片 0、1 各 2 集，种子 42；模型与里程计同卡，和部署一致），
  打开 `PPA_EVAL_TIMING=1`，代码用 `git archive` 导出的源码副本，不动部署检出。金丝雀当时是两卡两槽；这回单槽，按分片顺序跑。
- **通过条件**，两条都要满足：
  1. 两份客户端日志逐调用相同：用 `scripts/exp19/select_cases.parse_client_log` 解析，比较每次调用的步号、类型、
     慢系统输出和动作块，比法与 `scripts/deploy/nav_agent_habitat_check.py` 的 `compare_logs` 相同。
  2. 每集的结局相同：步数 49 / 105 / 89 / 94，成功与否、NE、SPL、注入次数 7 / 17 / 12 / 16。
- **不通过**：只要有一处不同，就判定计时在 GPU 上不中立。这时不报任何延迟数字，先查原因。
- **完整性**：计时文件的行数必须等于规划调用数，否则这次的延迟数字作废。
- **能说明什么**：通过时，这次运行的延迟数字可以用来填 `docs/deploy/navigation_interface.md` §8。但注意三点：
  - 计时会在每个阶段同步 CUDA，墙钟比不计时略慢；
  - 样本只有 4 集，大约 100 次调用；
  - 机器是多人共用的，CPU 渲染会受别人负载影响。

**结果（2026-09-30，GPU 4 单卡，源码副本 `94146fa`）：通过。**

**比对**：
- 4 集共 99 次规划调用，与 09-28 金丝雀逐调用相同，四集依次为 14/14、30/30、28/28、27/27。
- 每集的成功、SPL、NE、步数、注入次数也逐位相同。
- 汇总同样一致：SR 100，SPL 87.0，NE 0.913，注入 52 次。
- 计时文件共 99 行，等于规划调用数。

**产物**：
- 位置：`6024_fjl:/home/fangjialei/verify_0930/timing_canary/`；汇总表在
  `runtime/20260930_192501_117103/latency/latency_summary.md`，比对结果在 `verify_0930/timing_canary_vs_canary.json`。
- 同步开销测不出来：同样这 4 集，NavAgent 核验时服务端没开计时，注入调用的模型 RPC 往返中位数是 3644 ms；开计时时是 3630 ms。

**逐阶段耗时**：取 52 次带注入的轨迹调用（`ppa` 组），单位 ms，中位数 / P90。模型与里程计同在一张卡上。

| 阶段 | 中位数 | P90 |
|---|---:|---:|
| 慢系统第一轮生成（输出"↓"） | 695 | 696 |
| 慢系统第二轮（准备 + 生成，在俯视图上给像素目标） | 107 + 1034 | 130 + 1037 |
| 历史认知头（`ppa_history_memory`） | 588 | 592 |
| 快系统条件前向（`system1_condition_latents`） | 867 | 868 |
| 注入（桥） | 1.3 | 1.4 |
| NextDiT 采样（32 条 × 10 步） | 203 | 205 |
| 轨迹转动作 | 0.5 | 0.6 |
| 未来头（只做诊断） | 1.8 | 2.0 |
| **模型服务端合计** | **3624** | **3681** |
| 里程计位姿查询（含补建图，`query_map_update`） | 1294 | 1595 |
| 里程计写入（每步，服务端 / 往返） | 7 / 13 | 7 / 15 |
| 规划关键路径（`plan_latency`，含仿真渲染） | 5165 | 5486 |

**其他读数**：
- **第 20 帧首次建图**：1.8–2.5 s，每集 1 次，共 4 次。
- **按执行动作摊销**：`ppa` 组中位 1.43 s/动作，其中渲染 0.17 s；全部调用合计 1.28 s/动作。
- **只有慢系统的调用**（输出箭头或停止，34 次）：服务端中位 817 ms。
- **显存**：模型服务端峰值分配 17.4 GB、预留 21.8 GB；里程计峰值分配 11.3 GB、预留 14.6 GB；整卡 37.4 GB。
- **相对 native 的额外开销**：每次规划多出历史认知头 0.59 s、桥约 1 ms、里程计查询约 1.3 s；每步多一次约 13 ms 的里程计写入；每集多一次首次建图。
- **边界**：只有 4 集、99 次调用，都在同一个场景；机器是多人共用的，CPU 渲染受别人负载影响；这些数不能外推到别的卡或别的场景。

