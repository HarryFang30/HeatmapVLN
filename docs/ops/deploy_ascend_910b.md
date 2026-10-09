# 部署在昇腾 910B（华为云 ModelArts Notebook）

2026-10-09 搭好。这台机器**只跑两个服务端**：模型服务端和 AMB3R 里程计服务端。Habitat
客户端、Xvfb 和锁定评测计划**全部留在 RTX 4090 那台 x86 机器上**（见
`deploy_rtx4090.md`），客户端经 SSH 隧道连到 `127.0.0.1:<端口>`，**客户端程序的参数、协议、
合并流程一行都没改**。启动脚本加了外部模式的核对（见 §6），另外有两条**在本机模式下也生效**：带上限的运行不许重试、也不许续跑进已有结果的目录（见 `deploy_rtx4090.md` §2），以及不设上限的运行跑完后要核对记录的集与集表完全一致。

为什么这样切：habitat-sim 0.1.7 在 aarch64 上没有预编译包，`.so` 的名字和 RUNPATH 都是
x86；而客户端根本不碰加速卡（启动脚本给它的是 `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`，
CPU 渲染）。把客户端留在 x86，就把唯一的硬障碍整个移出了关键路径。

```
4090 机器（容器 fjl-habitat）            910B（ModelArts Notebook）
  Habitat 客户端 + Xvfb  ──SSH 隧道──▶  模型服务端 + 里程计服务端
  锁定计划 / 场景 / 合并                  权重 / NPU
```

> **这台机器上还一个数都不能报。** 还差什么见 §9 和
> [`ascend_910b_open_problems.md`](ascend_910b_open_problems.md)。

---

## 1. 机器

| 项 | 值 |
|---|---|
| 架构 | **aarch64**（鲲鹏），Huawei Cloud EulerOS 2.0，192 核 / 1.5 TB 内存 |
| 加速卡 | **昇腾 910B3 ×8**，每张 HBM 64 GB（`npu-smi` 报 65536 MiB）。看卡用 **`npu-smi info`**，没有 `nvidia-smi` |
| CANN | 8.3.RC1（`/usr/local/Ascend/ascend-toolkit/latest`），驱动 npu-smi 24.1.0.3 |
| 共享盘 | `/home/ma-user/work` 是 SFS Turbo（NFS，4.8 T） |
| 本地盘 | `/home/ma-user` 和 `/tmp` 是本地 overlay（21 TB，空 20 TB）。**跟着实例走，换实例就没了** |
| 外网 | 直连可用（pypi、HuggingFace、GitHub 都通），不需要代理 |
| 接入 | SSH 远程开发，地址和端口每建一个实例都会变，见 `~/.ssh/modelarts_setup.sh`（本地） |

卡是多人共用的，`npu-smi` 上**每张空卡也有约 3.3–3.4 GB 常驻占用**，判断"空卡"的阈值要在它
之上（启动脚本默认 `PPA_NPU_MAX_USED_MIB=4096`）。

这台实例上 `npu-smi` 的 **NPU ID 与 Chip Logic ID 逐一相同**（8 张都是恒等映射），所以启动脚本
用的逻辑编号和 `npu-smi info` 表里的编号是同一个东西。换实例后先 `npu-smi info -m` 再看一眼，
不相同的话"只拿空卡"读的就是另一张卡。

---

## 2. 布局

全部在 `~/work/zhr/zhr_1` 下（用户要求不要写到这个目录外面；唯一例外是每次运行的 TMPDIR，
见 §7 第一条）。

| 物 | 路径 | 来源与校验 |
|---|---|---|
| 代码 | `zhr_1/HeatmapVLN` | 本仓库，从 GitHub 直接 clone |
| RPC 工具 `vla_rpc` | `zhr_1/rpc` | 从 4090 容器打包搬来 |
| AMB3R | `zhr_1/amb3r` | 三个提交：上游 `92c4081`；`df74392`（HeatmapVLN 本地补丁：`amb3r_local.patch` + `memory_bounded_attention.py` + utils3d wheel）；`8985d55`（昇腾 NPU 补丁）。**`df74392^{tree}` = `09f1b2f`，就是 4090 上部署的那棵树**；HEAD 的树是 `d89957c`。校验见 §7 第六条 |
| DA3 权重 | `zhr_1/amb3r/checkpoints/DA3NESTED-GIANT-LARGE` | HF `depth-anything/DA3NESTED-GIANT-LARGE@8615eef`，与 4090 记录的 commit 一致 |
| 慢系统 / 快系统底座 | `zhr_1/InternNav_Model` | HF `InternRobotics/InternVLA-N1-DualVLN@a698a9e`。**15 个文件的 sha256 与 4090 上的逐一相同** |
| 部署权重 | `zhr_1/weights/ppa_refine_v2_best.pth` | sha256 `0b5a0644…6d69`，与 4090 一致 |
| Python 环境 | `zhr_1/envs/ppa` | `conda create --clone PyTorch-2.7.1` 再钉版本（§3） |
| 服务端运行目录 | `zhr_1/servers/<时间戳>/` | 日志、`servers.json`，以及 `STOPPED` / `RETIRED` 标记 |
| 短路径 TMPDIR | `/tmp/ppa-m<槽>.XXXXXX`、`/tmp/ppa-v<槽>.XXXXXX` | 每次运行 `mktemp -d` 新建、退出时删掉，见 §7 第一条 |

权重没走 4090 转发，而是在这台机器上直接从 HuggingFace 下（22 GB 约 1 分钟），再逐文件比
sha256。**这一步是"同一个模型"的唯一凭据**，别省。

---

## 3. 环境

克隆一份 notebook 自带的 `PyTorch-2.7.1` 环境，**不要动别人共用的那个**：

```bash
conda create -y -p ~/work/zhr/zhr_1/envs/ppa --clone PyTorch-2.7.1
```

基础：Python 3.11.10 + torch 2.7.1 + torch_npu 2.7.1（torch 本体是 `+cpu` 版，算子走
torch_npu）。在它之上钉死这些：

| 包 | 版本 | 为什么 |
|---|---|---|
| `transformers` | **4.51.0** | `src/models/runtime_compat.py` 硬 gate，版本不等就抛 |
| `numpy` | **1.26.4** | 见 §7 第二条，numpy 2 会把一堆 C 扩展的 ABI 弄坏 |
| `diffusers` | 0.36.0 | 与 4090 一致 |
| `protobuf` | 6.33.6 | 3.20.2 太老，生成的 `vla_pb2` 要 `runtime_version` |
| 其他 | `qwen-vl-utils` `accelerate` `peft` `safetensors` `timm` `pydantic` `numpy-quaternion` `trimesh` `yacs` `addict` `utils3d`（AMB3R 自带 wheel） | 与 4090 对齐 |

**不装** xformers / flash-attn / triton。注意力走 `sdpa`（由
`configs/ppa_action_refine_v2_8gpu.yaml` 指定），不需要 flash-attn。

ModelArts 的私有 pip 源没有 transformers/diffusers，用公网源（清华镜像快）。

---

## 4. 起服务端

```bash
cd ~/work/zhr/zhr_1/HeatmapVLN
PPA_EVAL_ROOT=~/work/zhr/zhr_1 PPA_NPU_DEVICES=0 \
  bash scripts/ascend/run_ppa_servers_npu.sh
```

- `PPA_NPU_DEVICES` 是每槽一张卡的逻辑编号（这台实例上等于 `npu-smi` 的 NPU ID，见 §1），
  只收不带前导零的十进制（`00` 和 `0` 是同一张卡，却能骗过去重）。一槽 = 一对（模型服务端，
  里程计服务端）。
- **默认两者同卡，这是实测的上限，不是宽裕。** 负载下整卡约 **62.2 / 65.5 GB**（模型
  `peak_reserved` 41.8 GB + 里程计 17.9 GB + 每卡约 3.3 GB 常驻，见 §8），只剩约 3 GB。空载
  读数 26.5 GB 不是部署占用。要分卡用 `PPA_NPU_VO_DEVICES`（每槽一个编号）。
- 启动脚本在起任何进程之前按卡做一次**摆放预算**（按 §8 的数：模型 42、里程计 18、常驻 3 GB）：
  - 一张卡上有模型服务端 + **2 个及以上**里程计 → 拒绝；
  - 一张没有模型的卡上 **4 个及以上**里程计 → 拒绝；
  - 一张没有模型的卡上 2–3 个里程计 → 只警告（这种摆法从没量过）。
  
  这个预算存在是因为下面那道"只拿空卡"检查**看不见本次运行自己**：它在服务端起来之前读表，
  只能看到别人的占用。以前一个重复的里程计编号能过所有检查，跑几个小时后 OOM。
- "只拿空卡"：启动时读一次 `npu-smi info`，每张要用的卡 HBM 已用超过 `PPA_NPU_MAX_USED_MIB`
  就拒绝；**读不出来也拒绝**，并把读到的整张表打出来（解析见 §7 第八条）。
- 启动前先跑 `scripts/ascend/check_amb3r_patch.sh`，AMB3R 树没打补丁就不起（见 §7 第六条）。
- 脚本自己 `source` CANN 的 `set_env.sh`：**非交互式 shell 不一定加载 CANN**，不加载时
  torch_npu 根本看不到卡。
- 两个服务端都只绑 `127.0.0.1`。**gRPC 两端都是无鉴权明文，绝不能对外暴露。**
- 用 `ASCEND_RT_VISIBLE_DEVICES`，**昇腾不认 `CUDA_VISIBLE_DEVICES`**——用错的话每个槽都会
  落到 0 号卡，而日志看起来像是分散的。
- 每次运行发一个 **`SERVER_INSTANCE` 令牌**，第 k 槽的两个服务端都在 `GetServerInfo` 里报
  `heatmapvln-instance:<令牌>/slot<k>`（外加 `heatmapvln-device:npu`、`heatmapvln-timing:<0|1>`，
  里程计再加 `heatmapvln-vo-rng-seed:<种子>`），带令牌启动时端口**不开 SO_REUSEPORT**，第二个
  进程抢同一端口会直接起不来，而不是和第一个分流。就绪探测要求这个令牌，还要求所有槽报的
  `model_version` 相同。
- 起来之后它会一直挂着并盯着各槽：进程死了就整体报错退出；**进程活着但设备坏了**（日志里出现
  毒标记）就只退役那一槽，见 §7 第十条。
- 产出 `servers.json`，schema **`heatmapvln-npu-servers-v2`**：

  | 字段 | 说明 |
  |---|---|
  | `slots` `model_ports` `vo_ports` `npus` `vo_npus` | 布局 |
  | `server_instance` | 本次令牌；第 k 槽实际报的是 `<令牌>/slot<k>` |
  | `device` | 恒为 `npu` |
  | `model_version` | 就绪探测时从活的模型服务端读到的 |
  | `repo_commit` / `repo_dirty` | 40 位 commit（读不到就直接失败，不再写 `unknown`）；工作区有已跟踪文件的改动则 `repo_dirty=1`（未跟踪文件不算） |
  | `vo_rng_seed` `timing` `bridge_off` `num_sample_trajs` `num_inference_steps` | 本次的设置；后两个为空串表示用配置文件的值 |

  客户端怎么核对见 §6。

环境变量与 4090 的启动脚本同名（`PPA_EVAL_TIMING` / `PPA_EVAL_BRIDGE_OFF` /
`PPA_EVAL_NUM_SAMPLE_TRAJS` / `PPA_EVAL_NUM_INFERENCE_STEPS`），所以一套 export 可以同时驱动两边
——**而且必须这样**：客户端会逐项比对，两边不一致就拒绝（§6）。

### 启动必须出现的证据

| 日志行 | 证明什么 | 证明不了什么 | 启动脚本 grep？ |
|---|---|---|---|
| `Model server device: npu:0` / `VO server device: npu:0` | 进程真的落在 NPU 上（见 §7 第三条） | 同一行的 `bf16=True` 是 `is_bf16_supported()`，说的是**这张卡支持 bf16**，不是哪次前向用了 bf16；里程计那行的 `DA3_SDPA_QUERY_CHUNK_SIZE=` 只是把环境变量原样回显 | 是（只认 `npu:0` 那部分） |
| `Formal PPA online AMB3R runtime enabled … tensors={'heatmap': 79, 'future': 11, 'bridge': 10}` | 与 4090 上**完全相同**的预检行 | | 是 |
| `DA3 attention: query_chunk=256 (parsed by DA3), memory_bounded=True, xformers_disabled=True` | 由 DA3 **自己的**解析函数读出来的分块大小（AMB3R 树里 `thirdparty/depth_anything_3/model/utils/memory_bounded_attention.py` 的 `_configured_query_chunk_size()`），以及 dinov2 的注意力层绑定的确实是 `memory_bounded_scaled_dot_product_attention`。不设变量时报 `query_chunk=0`，**所以这条能失败** | **某一次前向到底分没分块。** 只有 `chunk_size > 0` 且 `query_length > chunk_size` 且 `dropout_p == 0` 时才分块，否则直接走普通 SDPA；前向的 dtype 也不在这里，由补丁校验兜（§7 第六条） | 是（整行一次 grep：`query_chunk=256 (parsed by DA3), memory_bounded=True`，两件事必须同时出现在同一行） |
| `NPU op toolchain ready (conv2d … in bf16 on npu:0)` | CANN 算子工具链在启动时初始化过了（见 §7 第一条） | 这是**手工构造的预热算子**（里程计是一个 conv2d，模型是 conv2d + matmul），用的 bf16 是预热自己选的，**不代表任何一次前向的 dtype** | 否 |
| `VO device RNG seed per reset_episode: <种子>` | 里程计按这个种子逐集重播种（§7 第五条） | | 否（客户端在线核对 `heatmapvln-vo-rng-seed`，§6） |
| `PPA bridge off (EXP-20 A1)` / `Sensitivity override (EXP-21): nextdit.<键>` | 对应设置真的生效 | | 只在设了对应变量时 |

**以前这张表里那条 `DA3_SDPA_QUERY_CHUNK_SIZE=256` 是自己证自己**：启动脚本导出变量，服务端
原样回显，启动脚本再 grep 自己导出的值——不管 DA3 拿这个变量做了什么都不可能失败，却被当成
"走了认证过的分块路径"的证据。现在 grep 的是上表第三行。

---

## 5. 隧道与断线

隧道脚本在仓库里：`scripts/ascend/start_tunnel.sh`。在 **4090 容器内**起（容器是 bridge 网络、
没有发布端口，所以隧道必须在容器里起）；容器里的仓库在 `/workspace/HeatmapVLN`，先把它更新到
带这个脚本的 commit：

```bash
docker exec -d fjl-habitat bash -lc \
  'cd /workspace/HeatmapVLN && PPA_TUNNEL_HOST=<notebook 地址> PPA_TUNNEL_PORT=<notebook 端口> \
   bash scripts/ascend/start_tunnel.sh <槽数> >> /workspace/ppa_tunnel/tunnel.log 2>&1'
```

- 参数：`<槽数>`（1–8，起槽 0…n-1），或 `--slot <k>` 只起一槽——**给正在跑的评测加槽用
  `--slot`**，别拿更大的槽数重启整条隧道，那会断掉正在用的转发。
- 环境变量：`PPA_TUNNEL_HOST`、`PPA_TUNNEL_PORT` 必填；`PPA_TUNNEL_DIR`（默认
  `/workspace/ppa_tunnel`）、`PPA_TUNNEL_KEY`（默认 `$PPA_TUNNEL_DIR/id_ed25519`）、
  `PPA_TUNNEL_KNOWN_HOSTS`（默认 `$PPA_TUNNEL_DIR/known_hosts`）、`PPA_TUNNEL_USER`（默认
  `ma-user`）。容器里的私钥文件名不是 `id_ed25519` 就用 `PPA_TUNNEL_KEY` 指过去。
- 用的是**专用密钥**，不是你登录 ModelArts 的那把。它在 910B 的 `authorized_keys` 里带
  `restrict,port-forwarding` 和只允许这些端口的 `permitopen`：**拿它既开不了 shell，也转不到
  别的地方**。要撤销就删掉那一行。私钥只放在 4090 容器的 `/workspace/ppa_tunnel/` 里，**不进
  仓库**；仓库里只有脚本。
- `ExitOnForwardFailure=yes`：本地端口已被占（比如上一条隧道没停）时 ssh 直接退出并在日志里
  反复重连，而不是建一条没有转发的会话。`StrictHostKeyChecking=accept-new`：换实例后若地址
  端口被复用、主机密钥变了，ssh 会拒绝，删掉 `known_hosts` 里那一条。
- 脚本是**永久重连**的死循环。**按 PID 停**（`pkill -f` 会连跑它的 shell 一起匹配上）。
- 第 k 个槽映射 `52400+k`（模型）和 `52500+k`（里程计），两端端口号相同——**和 4090 本机认证
  路径的默认端口是同一组**，这正是 §6 要核对实例令牌的原因。

### 断线会发生什么

**单次 RPC 仍然零重试**：`sync_client.py` 遇到 `grpc.RpcError` 只记一条 warning 返回 `None`，
客户端随即 `raise`，这一片的客户端进程就死了。救它的是**分片级重启**：

- **`PPA_EVAL_SHARD_RETRIES`**（默认 **0**，认证过的 CUDA 运行用的就是 0）：客户端进程死了
  之后在同一片里最多重起这么多次，带 `--resume`，已记录的集保留，代价最多是死时在跑的那一集。
  22 小时的全量建议开（比如 2）。
- **做"同机两遍逐调用一致"那一对时保持 0**：客户端日志是追加写的，重启会在同一个文件里留下
  重复块，`scripts/exp19/select_cases.py` 的 self_check 断言没有重复块。
- **带集数上限的运行不许重试、不许续跑**：`PPA_EVAL_MAX_EPISODES_PER_SHARD` 数的是"还没跑的
  集"，每次重启都原样再传一遍，所以"4 集金丝雀"死一次就悄悄变 5 集、`SHARD_RETRIES=2` 时最多
  12 集，`progress.json` 里还看不出哪几集是补的。现在设了上限时，`PPA_EVAL_SHARD_RETRIES>0`
  直接拒绝，输出目录里已有行也拒绝；确实要这样就设 `PPA_EVAL_ALLOW_CAPPED_RESUME=1`。
  正确做法是**把集钉死，不设上限**（§6）。
- **外部模式下，重启前等的是一次真的 RPC 应答，不是端口开着。** 等待用的是和就绪探测同一个
  `rpc_ready`：连上、`HealthCheck` 是 `SERVING`、`GetServerInfo` 报出本次的实例令牌、设备、
  计时和里程计种子。答了但不是记录里那套服务端 → 立刻放弃这一片；
  `PPA_EVAL_SERVER_START_TIMEOUT_S`（客户端默认 1800 秒）内等不到健康应答 → 也放弃这一片。
  
  为什么不能只看端口：外部模式下接受 TCP 连接的是**本机的 ssh 转发器**，远端死了它照样
  接；而且**昇腾的设备错误不杀进程**——一个吃过 AICPU 超时的服务端连接、`HealthCheck`、
  `GetServerInfo` 全过（§7 第十条）。只看端口的旧等待会立刻返回，重起的客户端在一个算不动的
  服务端上把剩下的重试次数烧光。

---

## 6. 用 910B 的服务端跑评测（在 4090 上）

4090 的启动脚本加一个开关就变成"只跑客户端"：

```bash
PPA_EVAL_EXTERNAL_SERVERS=1 \
PPA_EVAL_EXTERNAL_SERVER_DIR=/workspace/npu_servers \
PPA_EVAL_OUTPUT_ROOT=/workspace/eval_runs/<本次专用目录> \
PPA_EVAL_GPU_DEVICES=4 PPA_EVAL_DISPLAY_BASE=601 \
  bash scripts/run_ppa_r2r_val_unseen_cuda.sh
```

- `PPA_EVAL_EXTERNAL_SERVER_DIR` 是把 910B 的运行目录（`servers.json` + `logs/` + 可能有的
  `STOPPED` / `RETIRED`）拷到 4090 的一份。**每次起服务端都要重新拷**：令牌每次都换。
- **`PPA_EVAL_OUTPUT_ROOT` 在这个模式下是必填的。** 不填会落到 4090 自己的默认输出目录，
  而 `--resume` 会跳过那里已经跑过的集——两个平台的结果就混进同一份结果里了。
- `PPA_EVAL_GPU_DEVICES` 在这个模式下只决定**客户端槽数**和客户端的 `CUDA_VISIBLE_DEVICES`；
  渲染仍然是 llvmpipe。`PPA_EVAL_VO_GPU_DEVICES` 在这个模式下没有意义，设了会报错。
- 客户端不再需要权重、AMB3R 树和两个服务端脚本（那些是服务端侧的输入）。

### 客户端核对什么，为什么端口不够

**端口认不出是哪套服务端。** 远端槽位用 `52400+k / 52500+k`，跟 4090 本机认证路径的默认端口
是同一组；外部模式下连上的又是 ssh 转发器。4090 上只要有一套本机 CUDA 服务端占着这组端口，
"昇腾评测"就会悄悄跑在本机 4090 上，而以前的核对（端口、槽数、`bridge_off`）全过。
`model_version` 也不行：它由路径的一段拼出来（昇腾上是 `ppa-stage2-online-amb3r:zhr_1`，
4090 上是 `...:workspace`），两台布局相同的机器就分不开。所以现在分两层核：

1. **`servers.json`（拷来的那份）**，任何一项不符就拒绝：schema 必须是 v2；槽数够、端口对；
   `bridge_off`、`timing`、`num_sample_trajs`、`num_inference_steps` 与本次客户端一致（计时是
   按进程生效的，客户端开、服务端不开时以前所有检查都过，延迟汇总却只剩客户端阶段）；
   `server_instance` 非空（手工起的服务端没有，不认）；`device` 是 `npu`；`repo_commit` 是 40 位
   commit 且 `repo_dirty` 为 0；`repo_commit` **等于客户端自己的 commit**——两台机器要先拉到同
   一个 commit，确实要混用就设 `PPA_EVAL_ALLOW_COMMIT_MISMATCH=1`。
2. **在线探测**：每槽的两个服务端必须报出 `<server_instance>/slot<k>`、`heatmapvln-device:npu`、
   本次的计时开关，里程计还要报 `servers.json` 里的种子。答了但对不上 → 直接拒绝，提示"是不是
   本机 CUDA 服务端占着这些端口，或隧道指错了地方"。

另外三条拒绝：运行目录里有 `STOPPED`（那套服务端已经停了）、有 `RETIRED`（有槽的 NPU 坏了被
退役，§7 第十条）、或者拷来的日志里有毒标记 `NPU device unusable after a failed request`。

### 金丝雀：把集钉死，不要设上限

金丝雀要和 4090 的 4 集是**同一批集**。用 4090 的参照运行生成逐片的集列表：

```bash
python3 scripts/tools/make_episode_lists_from_run.py \
  --reference /workspace/eval_runs/canary_cuda_seed42 \
  --cohorts <锁定计划>/cohorts --shards 0,1 --expect-per-shard 2 \
  --out /workspace/eval_runs/ascend_canary_lists
```

参照运行本身续跑过、重启过，或集数对不上，它都拒绝生成。然后 `PPA_EVAL_SHARDS=0,1`、
`PPA_EVAL_EPISODE_LISTS_DIR=<上面的 --out>`、**不设** `PPA_EVAL_MAX_EPISODES_PER_SHARD`。这样
重启只能补完列表里的集，可以放心开 `PPA_EVAL_SHARD_RETRIES`；跑完后启动脚本核对记录下来的集
**恰好就是**列表里的集，多一集少一集都报错。

计时打开时，如果服务端没开计时，`scripts/tools/summarize_latency.py` 仍然写出汇总，但开头带
WARNING、JSON 里有 `server_timing_missing`、退出码 3，启动脚本会说明"只有客户端侧，不许引用
端到端延迟"。

---

## 7. 这台机器的坑

**① TMPDIR 必须短（AF_UNIX 108 字节），而且每次运行要独占。** CANN 第一次跑卷积时会初始化
它的算子仓库，这个过程走 `multiprocessing.Manager()`，要在 `TMPDIR` 下绑一个 AF_UNIX socket，
路径还要再加约 32 字节的 `/pymp-XXXXXXXX/listener-XXXXXXXX`。AF_UNIX 总长上限 108 字节，放在带
时间戳的运行目录下会超。超了之后报出来的**不是**路径太长，而是
`AclSetCompileopt(ACL_PRECISION_MODE) error code 500001` + `GEInitialize failed`，看着像 CANN
装坏了。更坑的是：模型服务端的 `model/tmp` 比里程计的 `vo/tmp` 长 3 个字符，于是**只有模型
服务端挂，里程计正常**，像是模型侧的问题。

以前用 `zhr_1/tmp/{m,v}<槽>`，只看槽号不看运行：两次"槽 0"的运行共用一个 TMPDIR，在 NFS 上
撞过（退出时 `Errno 39 Directory not empty`），还泄漏了 12 个 `pymp-*` 和 3 个
`amb3r_vo_cfg_*`，没人清。现在每槽每次运行 `mktemp -d` 一个 `/tmp/ppa-m<槽>.XXXXXX` /
`/tmp/ppa-v<槽>.XXXXXX`（本地盘、短、天然唯一），服务端停掉之后由退出 trap 删掉，长度断言还在。
所以每次运行都从一个空的算子仓库开始。`kill -9` 启动脚本时 trap 不跑，目录会留下，确认没有
服务端在跑之后手工删。万一以后某个实例的 `/tmp` 太小，用 `PPA_NPU_TMP_ROOT` 指到一个**短**
路径（比如 `zhr_1/tmp`，bootstrap 会建好它）。

**② numpy 必须是 1.x。** 一次装包留下的孤儿 pip 进程事后把 numpy 升到了 2.4.6，于是所有按
numpy 1 编译的 C 扩展（`cv2`，以及 CANN 自己算子工具链要 import 的那些）ABI 全坏。症状同样
不是启动报错，而是**服务端正常起来、通过所有检查，然后第一个真实请求死掉**，客户端只看到
`RPC model server returned no response`。启动预检现在会查 numpy 版本并真的 import 一次 cv2。
**装完包一定回头确认 `numpy.__version__`**，并且别用 `pkill` 杀 pip（它的子进程会活下来接着装）。

**③ 设备绝不能静默回落 CPU。** 原来模型服务端用 `torch.cuda.is_available()` 推设备，在非
CUDA 的 torch 上会悄悄变成 CPU：8.3 B 的模型照样加载、照样通过健康检查、照样回答，只是用的是
另一个设备的随机数、慢上千倍。现在 `--device {cuda,npu,cpu}` 显式指定，设备不可用就报错，
CPU 只在显式写明时才用。里程计服务端的 `--device` 以前根本没传到模型加载（`load_model("da3")`
不接受设备，会用 DA3 自己的 `.to("cuda")`），现在直接构造 `DA3(device=...)`。

**④ 计时/显存仪表以前是 CUDA 专用的。** 非 CUDA 设备上它**静默变 no-op**：同步被跳过，量到的
是下发时间而不是计算时间，而数字看着完全正常。现在 `src/utils/latency.py` 同时支持
`torch.cuda` 和 `torch.npu`，并且**点名一个不可用的加速器会直接报错**，而不是什么都不计。
响应里的字段仍然叫 `cuda_memory_mib`，这样两个平台的日志和汇总脚本可比。

**⑤ 里程计必须显式播种。** DA3 前向里有三处**没播种**的设备随机采样
（`utils/alignment.py` 的 `randperm`、`model/da3.py` 的两处 `randint`）。CUDA 上之所以跑两遍
结果一样，是因为 CUDA 默认生成器的默认种子是个常量；**昇腾上不是**——实测两个新进程报出的
`torch.npu.initial_seed()` 不同。所以 NPU 上必须给里程计服务端 `--rng-seed`，在每个 episode
的 `reset` 时重新播种。这么做想要的副作用是：每集的位姿不再依赖进程之前跑过多少集，于是 `--resume` 续跑和一口气
跑完应当一致（**这一点 CUDA 上原本并不成立**）。⚠️ **但这条在昇腾上还没验证过**：没有设确定性算子，归约顺序没钉住，而 EXP-22 的 P1/P2 比的是两个新进程按同一顺序跑，
并不覆盖"已经跑过别的集的进程"。它是 A0 的一个未测前提，不是保证。`--rng-seed` 是启动脚本加的（默认
0）；手工起的里程计服务端没有它，也没有实例令牌，客户端会拒绝。

**⑥ AMB3R 树里有两处只能改第三方代码，以前没有任何东西校验补丁打没打。** 补丁在
`scripts/ascend/amb3r_npu.patch`，跟着仓库走：
- `slam/pipeline.py` 的 `autocast(device_type='cuda')` 写死——非 CUDA 上 torch 只是**警告**然后
  按 fp32 跑，是一次静默的精度变化（激活显存还翻倍，而卡上只剩约 3 GB）；
- `thirdparty/depth_anything_3/api.py` 用 `torch.cuda.is_bf16_supported()` 定 dtype——非 CUDA 上
  它是 False，于是**悄悄降到 fp16**。现在按张量所在设备判断，不支持 bf16 就报错。

两处都**不打也不报错、日志里不留痕迹**。以前 bootstrap 只查文件存在，启动脚本的证据行也不覆盖
这棵树，干净 clone 和打过补丁的树在所有检查下一模一样。现在 `scripts/ascend/check_amb3r_patch.sh`
在**启动脚本起任何服务端之前**和 **bootstrap** 里各跑一次（只读，从不改树）：

1. 树和两个文件可读（读不到报"没挂载 / 根目录不对"，与"没打补丁"区分开）；
2. 两个文件里必须**有**补丁加的那两行，缺一行就拒绝，并打出打补丁的命令：
   ```bash
   git -c safe.directory=<树> -C <树> apply <仓库>/scripts/ascend/amb3r_npu.patch
   ```
3. 原来的 cuda 写法还在 → 只警告（可能是回退了，也可能是补丁没覆盖的第二处）；
4. 树是 git checkout 时：`git apply --reverse --check` 要能过（过不了只警告：两处修复在，但
   上下文漂了，该重新生成补丁）；**两个补丁文件之外的一切必须和 `09f1b2f` 一致**，否则拒绝。
   checkout 里没有 `09f1b2f`（比如一份只有上游的 clone）或不是 git checkout 时只警告，只验了
   两个文件。

**别拿整棵树的哈希去比 `09f1b2f`**：`09f1b2f` 是 `df74392^{tree}`，即**不带**昇腾补丁的那棵
（4090 部署的就是它），打过补丁的 HEAD 树是 `d89957c`。树哈希等于 `09f1b2f` 恰恰说明昇腾补丁
没打。2026-10-10 在真机的树上跑过，通过。

**⑦ 有算子会回落 CPU。** 日志里会看到 `aten::linalg_inv_ex.inverse` 和
`aten::_transformer_encoder_layer_fwd` "not currently supported on the NPU backend and will fall
back to run on the CPU"。能跑，只是慢，且这部分计算的数值环境与 CUDA 不同。别把这些警告当故障。

**⑧ `npu-smi` 的表：芯片行最后一个 `|` 段里有两对 `已用 / 总量`，要的是第二对。**

```
| 0                         | 0000:C1:00.0  | 0           0    / 0          65521/ 65536         |
```

前一对是 Memory-Usage（这个驱动上恒为 `0 / 0`），后一对才是 HBM。取第一对就是以前那个 bug：
**每张卡都读成 0 MiB**，包括占满 65521 MiB 的卡，"只拿空卡"的保护等于不存在。再往前一版按
空格硬切，把整行数字拼成一个数（空卡读成"100000 MiB 已用"）。现在解析在
`scripts/ascend/npu_hbm_used_mib.awk`，它要求芯片行的形状对（芯片号、Bus-Id）、最后一段**恰好**
两对、HBM 总量大于 0 且已用不超过总量、同一编号只出现一次；**任何一条不符就什么都不打印**，
启动脚本把空输出当"读不出"，打出整张表后拒绝启动。换驱动版本后它一拒绝，就是表的布局变了，
看表改解析，别绕过去。

**⑨ 不要用 `torch_npu.contrib.transfer_to_npu`。** 它把 `torch.cuda.*` 全局改写成 NPU，等于把
上面那些 CUDA 假设**藏起来**而不是修掉。

**⑩ 昇腾的设备错误不杀进程：坏掉的服务端照样过所有探活。** 证据：一个吃过 AICPU 超时
（ACL 507017）的里程计服务端，一小时后去探它，连接、`HealthCheck`、`GetServerInfo`、
`reset_episode` 和 19 次写入全部正常，第一次真正跑 DA3 mapping 前向的调用**立刻**失败：
`ACL stream synchronize failed, error code:507017`。也就是说：**这个服务端不可能再服务了**，而
部署当时有的每一种存活检查（TCP 连接、`HealthCheck`、`GetServerInfo`）它都过。

另一张卡上一个健康的里程计服务端，在机器上没有任何别的任务时，拿 21 帧**合成的随机噪声图**
做 mapping，也吃了同样的 AICPU 超时：日志里超时出现在请求开始后约 9 分钟，之后每个请求都立刻
以 507017 失败。**输入是合成噪声，不是 Habitat 帧**，所以这既不能说明原来那次故障是并发造成
的，也不能说明不是。它说明的是：这个平台上 mapping 前向会挂住，CANN 自己的超时要几分钟后才
报出来。

部署现在怎么应对：

- **服务端自己**（只在 NPU 上；CUDA 路径不变）：一个请求失败后，对自己的设备做一次
  `torch.npu.synchronize()`。同步也失败，就记一行 `NPU device unusable after a failed request`，
  之后 `HealthCheck` 一直报 `NOT_SERVING`。坏输入、数值错误也会让请求失败，但不会让同步失败，
  所以不会误报。
- **启动脚本**（hold 循环，每 30 秒）：在各槽日志里找这行毒标记，找到就**只退役那一槽**——停掉
  那一对、放出那张卡、往 `$RUNTIME_DIR/RETIRED` 追加 `slot=<k> role=<model|vo> <时间>`——其余槽
  继续服务；所有槽都退役了才整体退出。
- **客户端**：重启等待要的是 `SERVING` 的真应答（§5），所以退役槽的那一片等到超时后放弃，不会
  烧光重试；下次启动时，运行目录里有 `RETIRED` / `STOPPED` 或拷来的日志里有毒标记，直接拒绝。
- **唯一的治法是重起那个服务端**，重连客户端没用。仓库里没有单槽重起的入口：现在的做法是等其余
  槽跑完各自的分片，整套重起，重新拷运行目录，再用 `--resume` 补完那一片。

⚠️ 服务端自检这条路径还**没有被真机上真正的 507017 触发过**。万一同步没抛，就什么都不会记，
`HealthCheck` 照样 `SERVING`。所以跑金丝雀时仍然要人盯里程计日志的 mtime：它停止增长就是这个
情况。

---

## 8. 实测数字（2026-10-09，单槽，模型与里程计同卡）

### 显存

| 项 | 值 |
|---|---|
| 刚起好、还没收请求 | 模型 16.2 GB + 里程计 7.0 GB，含每卡约 3.3 GB 常驻，合计约 **26.5 GB** |
| 跑起来之后（服务端自报） | 模型 `peak_allocated` 26.9 GB / `peak_reserved` **41.8 GB**；里程计 13.3 / **17.9 GB** |
| 整卡 `device_used` | **约 62.2 GB / 65.5 GB** |

**别拿"26.5 GB"当部署占用**——那只是空载读数。跑起来之后缓存分配器的池会涨到合计约 59.7 GB，
整卡只剩 3 GB 余量。而且跑完之后空载，那张卡仍然是 65521 / 65536 MiB：缓存分配器不归还，
**服务端进程退出之前这张卡是满的**。含义：

- **一张卡放一槽（模型 + 里程计）就是上限**，不要往同一张卡上再塞东西；启动脚本的摆放预算
  （§4）会拒绝模型 + 2 个里程计同卡；
- 真实占用（两个服务端 `peak_allocated` 合计约 40 GB）比**保留**量小得多，差的是碎片和缓存，
  所以 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 这类设置**值得试**，但还没试过；
- 要更稳就用 `PPA_NPU_VO_DEVICES` 把里程计分到另一张卡，8 卡变成 4 个并发槽。

### 速度

一次规划调用的中位数，**与 4090 全量（A0 种子 42）对比**：

| 阶段 | 4090 | 910B | 倍数 |
|---|---|---|---|
| 模型服务端整个请求 | 3594 ms | 15251 ms | 4.2× |
| 里程计 `vo_query` | 1359 ms | 2263 ms | 1.7× |

910B 上模型服务端内部的分布（中位数，合计 15251 ms）：

| 阶段 | ms | 占比 |
|---|---|---|
| `system2_turn2_generate`（慢系统第二轮解码） | 4491 | 29% |
| `system1_condition_latents` | 3967 | 26% |
| `system2_turn1_generate`（慢系统第一轮解码） | 2915 | 19% |
| `ppa_history_memory`（历史头） | 2901 | 19% |
| `system1_nextdit_sampling` | 420 | 3% |
| 解码 + 两次 prep | 约 525 | 3% |

**这张表是打开计时测的，计时会插设备同步，所以比真实部署慢。** 不打开计时的金丝雀实测是
**约 11.2 秒一次调用**（910B 上那一集 50 步、279 秒 / 25 次调用），对 4090 的 5.26 秒是 **约 2.1 倍**。
按 8 槽折算，一次全量（56236 次调用）约 **22 小时**；4090 上拿 2 张空卡是约 41 小时。
**延迟数字一个都不许进台账**——计时中立性要先在这台机器上重做（见 §9）。

### 已经拿到的优化

**融合注意力路径会回落 CPU。** torch 的 `_transformer_encoder_layer_fwd` 没有 NPU 实现，
torch_npu 会把它搬到 CPU 上跑：实测单次前向 1.35 秒，普通路径 0.0037 秒，**差 365 倍**。
关掉 fastpath 之后 `system1_nextdit_sampling` 从 1372 ms 降到 420 ms（3.3 倍），整个请求降 6%。
只有 NextDiT 采样器真的走了那条路径，历史头和条件编码器没走，所以整体只拿到 6%。

### 还能往哪优化（都没做）

剩下的 4 倍集中在三处，每一处都要动 NPU 专用的算子选择，**会改变数值，要单独重新认证**：

- 慢系统两轮解码合计 7.4 秒（4090 上两轮约 1.9 秒）。Qwen2.5-VL 的增量解码在 torch_npu 上
  每步要下发大量小 kernel；可以试昇腾的融合注意力（`npu_fusion_attention`）、静态 KV cache。
- `system1_condition_latents` 4.0 秒、历史头 2.9 秒，两处都是视觉编码器加注意力。
- 纸面算力上 910B3 的 bf16 稠密算力并不低于 4090，所以这 4 倍**是软件而不是硅片**——有余量，
  但这是独立的工程项，不是调个开关。

### 里程计的注意力

4090 上之所以要 `DA3_SDPA_QUERY_CHUNK_SIZE=256`，是因为不分块要物化一个 20.6 GB 的 bf16 分数
矩阵。**昇腾上不分块反而又快又省**：20 视角约 2.07 万 token，不分块 **0.034 秒 / 0.23 GiB**，
分块 256 是 **0.258 秒**，慢约 7 倍（torch_npu 用的是融合算子）。为了与认证路径一致目前仍设
256；改成不分块是一次独立的变更，要重新验证。启动时 DA3 自报的 `query_chunk=` 只说明 DA3 读到
了什么，不说明哪次前向分了块（§4）。

### 其他

| 项 | 值 |
|---|---|
| 模型服务端启动 | 约 30 秒到监听（含 16 GB 权重从 NFS 加载） |
| 隧道上一次 gRPC 连接 + `get_server_info` | 约 330–380 ms |
| 每次 RPC 的隧道开销 | 模型 245 ms；里程计查询 37 ms、写入 61 ms |

## 9. 还没做的事

> 完整的问题清单（真机证据、每条的状态和处理顺序）在
> [`ascend_910b_open_problems.md`](ascend_910b_open_problems.md)。开跑之前先读那一份。

**那份清单初稿里"下一次开跑前必须处理"的七处（初稿编号 §2.1–2.5、§7.1、§7.2，现在列在它的
§0）在代码里都已经关了：**

| 问题 | 怎么关的 | 在真机上验过？ |
|---|---|---|
| 空卡检查每张卡都读成 0 | `npu_hbm_used_mib.awk`，读不出就拒绝（§7 第八条） | 测试用的是真机表格的原行；真机上还没随一次正式启动跑过 |
| 重启 + 集数上限 = 悄悄多跑集 | 设上限时拒绝重试与续跑；改用钉死的集列表（§5、§6） | 否 |
| 外部模式分不清远端昇腾和本机 CUDA | 实例令牌 + `servers.json` v2 + 在线探测（§4、§6） | 否 |
| 端口开着不等于服务端活着 | 重启等待要真 RPC 应答；服务端自检 + 退役槽（§5、§7 第十条） | "坏掉的服务端过所有探活"验过；自检路径没被真 507017 触发过 |
| TMPDIR 按槽号命名会撞 | 每次运行 `mktemp -d` 在本地 `/tmp`，trap 删（§7 第一条） | 否 |
| 分块注意力证据自己证自己 | 改 grep DA3 自报的行（§4） | 是：设 256 报 `query_chunk=256`，不设报 `query_chunk=0` |
| AMB3R 补丁没有校验 | `check_amb3r_patch.sh`（§7 第六条） | 是：在真机树上通过 |

**还要机器才能关的：**

- **4 集金丝雀没跑完**（只有 1 / 4 集），所以这台机器上**还没有任何 SR/SPL/NE 数字**。按 §6 用
  钉死的集列表重跑，机器上不跑别的。
- **同机跑两遍逐调用一致**还没验证。这是预注册判据里最严格的一条，`--rng-seed` 是为它准备的，
  但"准备好了"不等于"过了"。这一对 `PPA_EVAL_SHARD_RETRIES=0`（§5）。
- **延迟数字一个都不能报。** 计时中立性（打开计时不改变动作）是在 CUDA 上验证过的，昇腾上要
  **重做一次**才能报这里的延迟。
- **多槽并发没压过。** 只做过单槽验证；唯一一次两套服务端同时在机器上，就撞上了那次 AICPU
  超时（§7 第十条，因果没定）。
- **跨平台可比性**：噪声来源、bf16 算子、求和顺序都与 4090 不同，所以不要求也不可能逐集相同。
  能主张的是：昇腾上自己跑两遍一致，加上总数 SR 与 4090 同种子的差落在预注册的 ±2pt 平台带内。
  昇腾上的任何对比臂都只能和**昇腾上的 A0** 比。

---

## 10. 换一个新实例

实例是会换的，换了之后**只需要一步**：`zhr_1` 整个目录（代码、权重、AMB3R、6.4 GB 的 Python
环境、23 GB 合计）都在 SFS Turbo 共享盘上，跟着盘走；环境里写死的前缀就在盘内，所以照样可用。
CANN 由镜像提供。真正跟着旧实例消失的只有本地 overlay 上的东西：`~/.ssh/authorized_keys` 里
那条隧道公钥的授权、pip 的配置，以及 `/tmp` 下的东西（每次运行的 TMPDIR 本来就是用完即删）。

镜像要用同一个（这个环境是按它装的）：

```
pytorch_ascend:pytorch_2.7.1-cann_8.3.rc1-py_3.11-hce_2.0.2509-aarch64-snt9b-...
```

新实例起来以后：

```bash
bash ~/work/zhr/zhr_1/HeatmapVLN/scripts/ascend/bootstrap_instance.sh
```

它会：

1. 核对机器（aarch64、`npu-smi` 看得到几张卡、CANN 版本是否还是环境装配时的 8.3.RC1——不是
   的话只**警告**不拦）；
2. 核对共享盘上该有的东西都在（仓库、权重、AMB3R 树、RPC 工具、DA3 权重），**并跑
   `check_amb3r_patch.sh` 核对 AMB3R 树是打过昇腾补丁的那棵**（§7 第六条），不是就停；
3. 核对环境里每个钉死的版本（transformers 4.51.0、numpy 1.x、torch、bf16 可用、cv2 真的能
   import）——**这一步就是 §7 第二条那个坑的防线**；
4. 把隧道公钥重新写进 `~/.ssh/authorized_keys`，带 `restrict,port-forwarding` 和只允许 8 个槽
   那 16 个端口的 `permitopen`；公钥存在 `zhr_1/tunnel/tunnel_key.pub`（公钥不是秘密，所以可以
   放在盘上，这正是新实例不需要碰 4090 上那把私钥的原因）。旧授权按**密钥本体**认，不按注释
   子串（注释是自由文本，按子串删会把别人的 key 一起删掉），先写到临时文件再一次 `mv` 换上，
   中间没有"这把钥匙不被授权"的窗口；
5. 建好 `zhr_1/{tmp,logs,servers}`（`tmp` 只在把 `PPA_NPU_TMP_ROOT` 指回共享盘时才用，见 §7
   第一条）。

**bootstrap 还从没在一个真正的新实例上跑过。** 现在这台是手工装起来的，脚本是按"应该这样"写的，
不是按"这样试过"写的；第一次换实例要留返工时间。

然后：本地把 SSH 别名指到新实例的地址和端口（`~/.ssh/modelarts_setup.sh '<控制台 SSH 地址>'`），
先 `npu-smi info -m` 确认 NPU ID 与 Chip Logic ID 仍然相同（§1），按 §4 起服务端，按 §5 在
4090 容器里起隧道（新地址第一次连会被 `accept-new` 记进 `known_hosts`）。

**万一环境本身丢了**（共享盘重建过），按
`scripts/ascend/requirements-ascend.txt` 开头的三行命令重建，再跑一次 bootstrap。那个文件里也
写了三个不能动的版本以及为什么。
