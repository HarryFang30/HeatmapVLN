# 部署在昇腾 910B（华为云 ModelArts Notebook）

2026-10-09 搭好。这台机器**只跑两个服务端**：模型服务端和 AMB3R 里程计服务端。Habitat
客户端、Xvfb 和锁定评测计划**全部留在 RTX 4090 那台 x86 机器上**（见
`deploy_rtx4090.md`），客户端经 SSH 隧道连到 `127.0.0.1:<端口>`，**客户端一侧的参数、协议、
合并流程一行都没改**。

为什么这样切：habitat-sim 0.1.7 在 aarch64 上没有预编译包，`.so` 的名字和 RUNPATH 都是
x86；而客户端根本不碰加速卡（启动脚本给它的是 `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`，
CPU 渲染）。把客户端留在 x86，就把唯一的硬障碍整个移出了关键路径。

```
4090 机器（容器 fjl-habitat）            910B（ModelArts Notebook）
  Habitat 客户端 + Xvfb  ──SSH 隧道──▶  模型服务端 + 里程计服务端
  锁定计划 / 场景 / 合并                  权重 / NPU
```

---

## 1. 机器

| 项 | 值 |
|---|---|
| 架构 | **aarch64**（鲲鹏），Huawei Cloud EulerOS 2.0，192 核 / 1.5 TB 内存 |
| 加速卡 | **昇腾 910B3 ×8**，每张 HBM 64 GB。看卡用 **`npu-smi info`**，没有 `nvidia-smi` |
| CANN | 8.3.RC1（`/usr/local/Ascend/ascend-toolkit/latest`），驱动 npu-smi 24.1.0.3 |
| 共享盘 | `/home/ma-user/work` 是 SFS Turbo（NFS，4.8 T）。**`/home/ma-user` 和 `/tmp` 是本地 overlay** |
| 外网 | 直连可用（pypi、HuggingFace、GitHub 都通），不需要代理 |
| 接入 | SSH 远程开发，地址和端口每建一个实例都会变，见 `~/.ssh/modelarts_setup.sh`（本地） |

卡是多人共用的，`npu-smi` 上**每张空卡也有约 3.3 GB 常驻占用**，判断"空卡"的阈值要在它之上
（启动脚本默认 4096 MiB）。

---

## 2. 布局

全部在 `~/work/zhr/zhr_1` 下（用户要求不要写到这个目录外面）。

| 物 | 路径 | 来源与校验 |
|---|---|---|
| 代码 | `zhr_1/HeatmapVLN` | 本仓库，从 GitHub 直接 clone |
| RPC 工具 `vla_rpc` | `zhr_1/rpc` | 从 4090 容器打包搬来 |
| AMB3R | `zhr_1/amb3r` | 上游 `92c4081` 从 GitHub clone + 两个本地补丁提交。**树哈希 `09f1b2f` 与 4090 上部署的那棵树逐字节相同** |
| DA3 权重 | `zhr_1/amb3r/checkpoints/DA3NESTED-GIANT-LARGE` | HF `depth-anything/DA3NESTED-GIANT-LARGE@8615eef`，与 4090 记录的 commit 一致 |
| 慢系统 / 快系统底座 | `zhr_1/InternNav_Model` | HF `InternRobotics/InternVLA-N1-DualVLN@a698a9e`。**15 个文件的 sha256 与 4090 上的逐一相同** |
| 部署权重 | `zhr_1/weights/ppa_refine_v2_best.pth` | sha256 `0b5a0644…6d69`，与 4090 一致 |
| Python 环境 | `zhr_1/envs/ppa` | `conda create --clone PyTorch-2.7.1` 再钉版本（§3） |
| 服务端运行目录 | `zhr_1/servers/<时间戳>/` | 日志、`servers.json` |
| 短路径 TMPDIR | `zhr_1/tmp/m<槽>`、`zhr_1/tmp/v<槽>` | 必须短，见 §7 第一条 |

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

- `PPA_NPU_DEVICES` 是每槽一张卡的逻辑编号；一槽 = 一对（模型服务端，里程计服务端）。
  默认两者同卡（实测合计约 26.5 GB，64 GB 的卡很宽裕），要分卡用 `PPA_NPU_VO_DEVICES`。
- 脚本自己 `source` CANN 的 `set_env.sh`：**非交互式 shell 不一定加载 CANN**，不加载时
  torch_npu 根本看不到卡。
- 两个服务端都只绑 `127.0.0.1`。**gRPC 两端都是无鉴权明文，绝不能对外暴露。**
- 用 `ASCEND_RT_VISIBLE_DEVICES`，**昇腾不认 `CUDA_VISIBLE_DEVICES`**——用错的话每个槽都会
  落到 0 号卡，而日志看起来像是分散的。
- 起来之后它会一直挂着并盯着两个服务端；任一个死掉就整体报错退出，避免客户端往死端口灌请求。
- 产出 `servers.json`（槽数、端口、仓库 commit、`vo_rng_seed`、`bridge_off`），客户端那边会拿它核对。

环境变量与 4090 的启动脚本同名（`PPA_EVAL_TIMING` / `PPA_EVAL_BRIDGE_OFF` /
`PPA_EVAL_NUM_SAMPLE_TRAJS` / `PPA_EVAL_NUM_INFERENCE_STEPS`），所以一套 export 可以同时驱动两边。

### 启动必须出现的证据

| 日志行 | 含义 |
|---|---|
| `Model server device: npu:0` / `VO server device: npu:0` | 真的落在 NPU 上（见 §7 第三条） |
| `NPU op toolchain ready` | CANN 算子工具链在启动时就初始化过了（见 §7 第一条） |
| `Formal PPA online AMB3R runtime enabled … tensors={'heatmap': 79, 'future': 11, 'bridge': 10}` | 与 4090 上**完全相同**的预检行 |
| `DA3_SDPA_QUERY_CHUNK_SIZE=256` | 走的是认证过的分块注意力路径 |

启动脚本自己 grep 这几行，缺一条就不让跑。

---

## 5. 隧道

在 **4090 容器内**起隧道（容器是 bridge 网络、没有发布端口，所以隧道必须在容器里起）：

```bash
docker exec -d fjl-habitat bash -lc \
  'PPA_TUNNEL_HOST=<notebook 地址> PPA_TUNNEL_PORT=<notebook 端口> \
   /workspace/ppa_tunnel/start_tunnel.sh <槽数> > /workspace/ppa_tunnel/tunnel.log 2>&1'
```

- 用的是**专用密钥**，不是你登录 ModelArts 的那把。它在 910B 的 `authorized_keys` 里带
  `restrict,port-forwarding` 和只允许这些端口的 `permitopen`：**拿它既开不了 shell，也转不到
  别的地方**。要撤销就删掉那一行。私钥只放在 4090 容器的 `/workspace/ppa_tunnel/` 里。
- 脚本是**永久重连**的死循环。客户端对 RPC 失败零重试（`sync_client.py` 遇到 `grpc.RpcError`
  只记一条 warning 返回 `None`，客户端随即 `raise`），**隧道断一次就会杀掉整个分片**。
- 第 k 个槽映射 `52400+k`（模型）和 `52500+k`（里程计），两端端口号相同。

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

- `PPA_EVAL_EXTERNAL_SERVER_DIR` 是把 910B 的运行目录（`servers.json` + `logs/`）拷到 4090
  的一份。预检证据从这里读，脚本还会核对端口、槽数和 `bridge_off` 跟本次运行一致。
- **`PPA_EVAL_OUTPUT_ROOT` 在这个模式下是必填的。** 不填会落到 4090 自己的默认输出目录，
  而 `--resume` 会跳过那里已经跑过的集——两个平台的结果就混进同一份结果里了。
- `PPA_EVAL_GPU_DEVICES` 在这个模式下只决定**客户端槽数**和客户端的 `CUDA_VISIBLE_DEVICES`；
  渲染仍然是 llvmpipe。`PPA_EVAL_VO_GPU_DEVICES` 在这个模式下没有意义，设了会报错。
- 客户端不再需要权重、AMB3R 树和两个服务端脚本（那些是服务端侧的输入）。

---

## 7. 这台机器的坑

**① TMPDIR 必须短（AF_UNIX 108 字节）。** CANN 第一次跑卷积时会初始化它的算子仓库，这个过程
走 `multiprocessing.Manager()`，要在 `TMPDIR` 下绑一个 AF_UNIX socket，路径还要再加约 40
字节的 `/pymp-XXXXXXXX/listener-XXXXXXXX`。AF_UNIX 总长上限 108 字节，放在带时间戳的运行目录
下会超。超了之后报出来的**不是**路径太长，而是
`AclSetCompileopt(ACL_PRECISION_MODE) error code 500001` + `GEInitialize failed`，看着像 CANN
装坏了。
更坑的是：模型服务端的 `model/tmp` 比里程计的 `vo/tmp` 长 3 个字符，于是**只有模型服务端挂，
里程计正常**，像是模型侧的问题。现在两个都用 `zhr_1/tmp/{m,v}<槽>`，并且启动前校验长度。

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
的 `reset` 时重新播种。副作用是好的：每集的位姿不再依赖进程之前跑过多少集，`--resume` 续跑
和一口气跑完结果一致（**这一点 CUDA 上原本并不成立**）。

**⑥ AMB3R 树里有两处只能改第三方代码。** 打在 `scripts/ascend/amb3r_npu.patch` 里，跟着仓库走：
- `slam/pipeline.py` 的 `autocast(device_type='cuda')` 写死——非 CUDA 上 torch 只是**警告**然后
  按 fp32 跑，是一次静默的精度变化；
- `thirdparty/depth_anything_3/api.py` 用 `torch.cuda.is_bf16_supported()` 定 dtype——非 CUDA 上
  它是 False，于是**悄悄降到 fp16**。现在按张量所在设备判断，不支持 bf16 就报错。

**⑦ 有算子会回落 CPU。** 日志里会看到 `aten::linalg_inv_ex.inverse` 和
`aten::_transformer_encoder_layer_fwd` "not currently supported on the NPU backend and will fall
back to run on the CPU"。能跑，只是慢，且这部分计算的数值环境与 CUDA 不同。别把这些警告当故障。

**⑧ `npu-smi` 的表不要按空格硬切。** 一行里有多组 `已用 / 总量`，HBM 那组在最后。按 `|` 分段、
取最后一段里的 `N / M` 才对。按空格切会把整行的数字拼成一个巨大的值（实测把空卡读成"100000 MiB
已用"）。

**⑨ 不要用 `torch_npu.contrib.transfer_to_npu`。** 它把 `torch.cuda.*` 全局改写成 NPU，等于把
上面那些 CUDA 假设**藏起来**而不是修掉。

---

## 8. 实测数字（2026-10-09，单槽，模型与里程计同卡）

### 显存

| 项 | 值 |
|---|---|
| 刚起好、还没收请求 | 模型 16.2 GB + 里程计 7.0 GB，含每卡约 3.3 GB 常驻，合计约 **26.5 GB** |
| 跑起来之后（服务端自报） | 模型 `peak_allocated` 26.9 GB / `peak_reserved` **41.8 GB**；里程计 13.3 / **17.9 GB** |
| 整卡 `device_used` | **约 62.2 GB / 65.5 GB** |

**别拿"26.5 GB"当部署占用**——那只是空载读数。跑起来之后缓存分配器的池会涨到合计约 59.7 GB，
整卡只剩 3 GB 余量。含义：

- **一张卡放一槽（模型 + 里程计）已经接近上限**，不要往同一张卡上再塞一槽；
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
**约 11.2 秒一次调用**（49 步一集 279 秒 / 25 次调用），对 4090 的 5.26 秒是 **约 2.1 倍**。
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
256；改成不分块是一次独立的变更，要重新验证。

### 其他

| 项 | 值 |
|---|---|
| 模型服务端启动 | 约 30 秒到监听（含 16 GB 权重从 NFS 加载） |
| 隧道上一次 gRPC 连接 + `get_server_info` | 约 330–380 ms |
| 每次 RPC 的隧道开销 | 模型 245 ms；里程计查询 37 ms、写入 61 ms |

## 9. 还没做的事

- **金丝雀还没跑完**，所以这台机器上还没有任何 SR/SPL/NE 数字。
- **同机跑两遍逐调用一致**还没验证。这是预注册判据里最严格的一条，`--rng-seed` 是为它准备的，
  但"准备好了"不等于"过了"。
- **延迟数字一个都不能报。** 计时中立性（打开计时不改变动作）是在 CUDA 上验证过的，昇腾上要
  **重做一次**才能报这里的延迟。
- **跨平台可比性**：噪声来源、bf16 算子、求和顺序都与 4090 不同，所以不要求也不可能逐集相同。
  能主张的是：昇腾上自己跑两遍一致，加上总数 SR 与 4090 同种子的差落在预注册的 ±2pt 平台带内。
  昇腾上的任何对比臂都只能和**昇腾上的 A0** 比。
- 多槽（8 张卡 8 槽）只做过单槽验证，还没压过并发。

---

## 10. 换一个新实例

实例是会换的，换了之后**只需要一步**：`zhr_1` 整个目录（代码、权重、AMB3R、6.4 GB 的 Python
环境、23 GB 合计）都在 SFS Turbo 共享盘上，跟着盘走；环境里写死的前缀就在盘内，所以照样可用。
CANN 由镜像提供。真正跟着旧实例消失的只有本地 overlay 上的东西：`~/.ssh/authorized_keys` 里
那条隧道公钥的授权，和 pip 的配置。

镜像要用同一个（这个环境是按它装的）：

```
pytorch_ascend:pytorch_2.7.1-cann_8.3.rc1-py_3.11-hce_2.0.2509-aarch64-snt9b-...
```

新实例起来以后：

```bash
bash ~/work/zhr/zhr_1/HeatmapVLN/scripts/ascend/bootstrap_instance.sh
```

它会：

1. 核对机器（aarch64、`npu-smi` 看得到几张卡、CANN 版本是否还是环境装配时的 8.3.RC1）；
2. 核对共享盘上该有的东西都在（仓库、权重、AMB3R 树、RPC 工具、DA3 权重）；
3. 核对环境里每个钉死的版本（transformers 4.51.0、numpy 1.x、torch、bf16 可用、cv2 真的能
   import）——**这一步就是 §7 第二条那个坑的防线**；
4. 把隧道公钥重新写进 `~/.ssh/authorized_keys`，带 `restrict,port-forwarding` 和只允许 8 个槽
   那 16 个端口的 `permitopen`；公钥存在 `zhr_1/tunnel/tunnel_key.pub`（公钥不是秘密，所以可以
   放在盘上，这正是新实例不需要碰 4090 上那把私钥的原因）；
5. 建好 `zhr_1/{tmp,logs,servers}`（`tmp` 必须短，见 §7 第一条）。

然后：本地把 SSH 别名指到新实例的地址和端口（`~/.ssh/modelarts_setup.sh '<控制台 SSH 地址>'`），
按 §4 起服务端，按 §5 在 4090 容器里起隧道。

**万一环境本身丢了**（共享盘重建过），按
`scripts/ascend/requirements-ascend.txt` 开头的三行命令重建，再跑一次 bootstrap。那个文件里也
写了三个不能动的版本以及为什么。
