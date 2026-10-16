# 实机部署说明：4090 服务器 + 宇树 Go2

给要把导航模型接到宇树 Go2 上的同事。一页纸的摘要见 [`README.md`](README.md)。

> **状态（2026-10-15）**：没有在任何真机上跑过。机器人侧客户端 `NavAgent` 接 4090 上的真服务端时，
> 在 4 集仿真上与部署的评测客户端逐调用相同（附录 C）；Go2 一侧（§6）只有没上过机的示意代码。

| 代码 | 作用 |
|---|---|
| `src/deploy/nav_agent.py` | 机器人侧客户端 `NavAgent`，纯 Python，不要 GPU、不要 torch |
| `src/deploy/fake_servers.py` | 不要 GPU 的假服务端，用来先把机器人一侧跑通 |
| `scripts/deploy/start_nav_servers_cuda.sh` | 4090 侧：在一张卡上起模型服务端和里程计服务端 |

**目录**

1. 系统组成
2. 输入与输出 ← 接口契约，接机器人前先读这一节
3. 4090 侧：启动服务端
4. 网络：让机器人侧电脑连到 4090
5. 机器人侧电脑：环境与自检
6. 接到 Go2
7. 第一次上机的顺序
8. 延迟、带宽与显存
9. 失败、超时与重置
- 附录 A 没有验证过的 · 附录 B 与仿真评测客户端的差异 · 附录 C 验证记录 · 附录 D 出处与待实测

---

## 1. 系统组成

这是一个视觉语言导航模型：给它一句英文指令，再一步一步给它机器人看到的 RGB 图，它每步回一个离散动作
（前进 0.25 m、左转 15°、右转 15°、相机下视、停止），直到它说停止。只用 RGB：不要深度、不要里程计、
不要定位，自身的相对位姿由服务器上的里程计服务端从 RGB 序列在线估计。只在 Habitat 仿真（R2R-CE）里评测过。

```
4090 宿主机 · docker 容器 fjl-habitat · 一张 48 GB 卡
    模型服务端    :52400   慢系统（给像素目标或箭头）+ 快系统（轨迹）+ 历史认知
    里程计服务端  :52500   AMB3R 在线建图，给历史帧相对当前帧的位姿
        ▲
        │  gRPC（vla_rpc：JSON + JPEG，无加密、无认证），走 ssh -L 隧道（§4）
        │
机器人侧电脑（x86 笔记本 / NUC，背在狗上，Python 3.10，一个进程）
    NavAgent  ·  两台相机的读取  ·  Go2 运动控制
        │                                   │
        │ 网线 192.168.123.x（DDS）          │ USB
        ▼                                   ▼
Go2 EDU 本体（192.168.123.161）         前视相机（水平）+ 下视相机（下倾 30°）
```

模型里三部分的叫法，后文会用到：

- **慢系统**：一个多模态大模型，读指令和图像（当前图、最多 8 张历史图、下视图），输出图上要去的像素点、几个箭头动作，或者停止。
- **快系统**：拿到像素点后生成一段轨迹，再转成最多 4 个离散动作。
- **历史认知**：模型里用历史帧和它们相对当前帧的位姿（里程计给的）的那部分；位姿就绪前（前 20 帧）跳过。字段名里的 `ppa` 指的就是它。

**状态放在哪：**

| 状态 | 放在哪 |
|---|---|
| 历史帧、动作队列、步数、每次调用的随机种子 | `NavAgent`（每集 `reset()` 清空） |
| 地图与轨迹 | 里程计服务端（一个会话，每集 `reset()` 时重置） |
| 无 | 模型服务端（每次调用自带全部输入，调用之间互不相关） |

**一对服务端只能服务一台机器人。** 里程计服务端同一时刻只有一个会话，第二个 `reset()` 会顶掉第一个；
两个服务端都单线程串行处理请求。

---

## 2. 输入与输出

### 2.1 最小循环

```python
from src.deploy import Action, NavAgent

with NavAgent("127.0.0.1:52400", "127.0.0.1:52500") as agent:   # 模型服务端、里程计服务端
    agent.reset("Walk past the sofa and stop at the kitchen door.", scene_id="lab3f", episode_id=17)
    pitch = 0.0
    while not agent.done:
        front = robot.capture(pitch)            # 当前位姿、按 pitch 拍的一张图（见 2.2）
        action = agent.act(front, robot.capture_lookdown, robot.capture_level)
        robot.execute(action)                   # 见 2.3；做完、停稳再进下一轮
        pitch = action.camera_pitch_deg         # LOOK_DOWN 为 −30，其余为 0
```

`robot.*` 是你们自己的函数。用两台固定相机时，`capture(pitch)` 在 `pitch < 0` 时读下视相机、否则读前视相机（§6.3）。

### 2.2 输入

**每集一次：`reset(instruction, *, scene_id, episode_id)`**

| 参数 | 类型 | 说明 |
|---|---|---|
| `instruction` | 非空 `str` | 英文自然语言指令。末尾的一个 `.`、`!` 或 `?` 会被去掉（与评测相同）。评测只用过 R2R 的英文指令 |
| `scene_id` | 非空 `str`，不含 `/` | 例如 `"go2lab"` |
| `episode_id` | 非负 `int` | `(scene_id, episode_id, 调用序号)` 决定快系统的随机噪声：同一组键、同样的输入会复现同样的动作。**每次实跑换一个 `episode_id`** |

`reset()` 同时重置里程计会话（丢掉地图）。参数不合法抛 `ValueError`。

**每步一次：`act(front_rgb, lookdown_fn, level_fn) -> Action`**

| 输入 | 什么时候要 | 是什么 |
|---|---|---|
| `front_rgb` | 每次 `act()` | 上一个动作做完、停稳后，在**当前位姿**拍的图。俯仰 = 上一个动作的 `camera_pitch_deg`：通常水平；上一个动作是 `LOOK_DOWN` 时是下倾 30° |
| `lookdown_fn()` | 只在这一步要规划时调，每步最多一次（约每 1–4 步一次） | 同一位姿、**相对水平下倾 30°** 的图。慢系统在这张图上找像素目标，快系统也拿它当输入，不能省，也不能拿前视图冒充 |
| `level_fn()` | 只在一种情况下调：上一个动作是 `LOOK_DOWN`，而这一步从队列里取到 STOP、当场重新规划。每步最多一次，先于 `lookdown_fn()` | 同一位姿、**水平**的图 |

用一台带云台的相机时，`lookdown_fn` / `level_fn` 返回前要把相机恢复水平。用两台固定相机时，
`lookdown_fn` 读下视相机，`level_fn` 读前视相机，不用管恢复（§6.3）。

**三张图的格式都一样：**

| 项 | 要求 | 说明 |
|---|---|---|
| 类型 | `numpy.ndarray`，`dtype=uint8`，形状 `(480, 640, 3)`，**RGB** 顺序 | 不是数组抛 `TypeError`，形状或 dtype 不对抛 `ValueError`。**不会替你缩放**。OpenCV 读出的是 BGR，要先 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，否则不报错但结果错 |
| 视场 | 水平 79°（4:3 时垂直约 63.5°），针孔模型 | 仿真值。真实相机要先去畸变，再居中裁到水平 79°、4:3，最后缩放到 640×480（`to_model_frame`，§6.3）。水平小于 79°、或垂直小于 63.5° 的相机补不回来；16:9 的画面受限的是垂直方向，水平要 ≥ 约 95.5° |
| 安装 | 离地 1.25 m，朝正前方；前视水平，下视下倾 30° | 仿真值。装不到时记下实际高度和俯角 |
| 曝光等 | 无要求 | 仿真里没有运动模糊、自动曝光；真机效果未知 |

### 2.3 输出

`act()` 每次返回一个 `Action`（`IntEnum`，值就是服务端的动作码）：

| `Action` | 值 | 机器人做什么 | 下一帧 `front_rgb` 怎么拍 |
|---|---|---|---|
| `FORWARD` | 1 | 沿当前朝向直行 0.25 m | 水平 |
| `TURN_LEFT` | 2 | 原地逆时针转 15° | 水平 |
| `TURN_RIGHT` | 3 | 原地顺时针转 15° | 水平 |
| `LOOK_DOWN` | 5 | 车体不动 | 下倾 30°（两台相机时：改读下视相机） |
| `STOP` | 0 | 停下，本集结束（`agent.done` 变真） | 不再调用 `act()` |

`action.camera_pitch_deg` 给出下一帧的俯仰（`LOOK_DOWN` 为 −30.0，其余 0.0）；常量
`FORWARD_STEP_M = 0.25`、`TURN_STEP_DEG = 15.0`、`LOOK_DOWN_DEG = 30.0` 也可以从 `src.deploy` 导入。

执行约定：

- **一个动作做完、停稳后，再拍下一帧、再调 `act()`。** 模型没有时间概念，只看这一串图。
- **LOOK_DOWN 只影响它之后的第一次拍摄**，也就是下一次 `act()` 收到的那一帧。同一步里的 `level_fn`、
  `lookdown_fn` 和之后的动作都从水平起算；连续两个 `LOOK_DOWN` 也还是 −30°，不会叠成 −60°。
  这是照着仿真来的：仿真里相机每拍一次就复位成水平。
- **动作是开环的**：0.25 m / 15° 走不准时模型不知道，只能从下一帧看出来。碰撞在仿真里是贴墙滑动，真机要自己处理。
- 左/右转、`LOOK_DOWN` 都算一步，也都会往里程计送一帧。
- 服务端只会返回上表五种动作；返回别的值（例如 4 = 抬头）时 `act()` 抛 `RuntimeError`。

**`STOP` 有四种情况。** 先看 `agent.last_step.source`；是 `terminal` 时再看 `agent.last_call.kind`（各种 `kind` 见 §2.4）：

| `last_step.source` | `last_call.kind` | 情况 | 真机上怎么记 |
|---|---|---|---|
| `terminal` | `stop` | 模型输出了停止 | 模型认为到了 |
| `terminal` | `fallback_stop` | 慢系统的输出解析不出任何动作，服务端兜底判停 | 失败 |
| `empty` | — | 服务端返回空动作块 | 失败 |
| `step_cap` | — | 到步数上限 `max_steps`（默认 500） | 超时。仿真评测到上限只是不再走，真机需要一个明确的停 |

动作块里的 STOP 在 `NavAgent` 内部表示"在这里重新规划"，不会返回给机器人。

### 2.4 一集里的调用顺序

一次规划返回最多 4 个动作：第一个当场返回，其余排队，之后每次 `act()` 取一个。队列空了、或取到 STOP 时，
在当前这一步重新规划。下面是一个**示意**（动作块是编的，什么时候调哪个回调是真的规则）：

| 步 | `act()` 收到的 `front_rgb` | `NavAgent` 内部 | 调了哪个回调 | 返回 |
|---|---|---|---|---|
| 0 | 前视（水平） | 队列空 → 规划，得到 `[1, 1, 2, 0]` | `lookdown_fn` | `FORWARD`，队列剩 `[1, 2, 0]` |
| 1 | 前视 | 取队列 | — | `FORWARD` |
| 2 | 前视 | 取队列 | — | `TURN_LEFT` |
| 3 | 前视 | 取到 STOP → 当场规划，得到 `[5, 0, 0, 0]` | `lookdown_fn` | `LOOK_DOWN`，队列剩 `[0, 0, 0]` |
| 4 | **下视**（上一个是 `LOOK_DOWN`） | 取到 STOP → 当场规划，这次规划的当前图要水平 | `level_fn`，再 `lookdown_fn` | 新块的第一个动作 |

每次 `act()` 都先把收到的 `front_rgb` 送进里程计（第 4 步送的就是那张下视图，与仿真一致）。
`LOOK_DOWN` 如果是动作块的最后一个（例如 `[1, 1, 1, 5]`），下一步队列已空，直接拿那张下倾图规划，不调 `level_fn`。

**预热：前 20 帧。** 里程计收满 20 帧才建第一张图。在此之前 `last_call.pose_ready` 为假，模型不用历史认知，
只按原始模型的方式规划，这是设计行为，不是错误。步号从 0 数，第 19 步起（`last_call.step ≥ 19`）规划调用的
`pose_ready` 为真，轨迹调用的 `ppa_applied` 也为真。每集都要重新预热。

**一次规划里发生什么**（同步，一次 RPC）：先跑慢系统（最多生成 128 个 token；第一轮输出 "↓" 时再加一轮，
用下视图）。

| 慢系统输出 | 接着做什么 | `last_call.kind` |
|---|---|---|
| 像素坐标 | 跑快系统（32 条轨迹 × 10 步去噪，取平均），转成最多 4 个动作 | `trajectory` |
| 箭头 | 直接转成动作，不跑快系统 | `native_actions` |
| STOP | 判停 | `stop` |
| 什么都解析不出 | 兜底判停 | `fallback_stop` |

没有独立运行的快系统循环。

### 2.5 诊断信息（不用也能跑）

| 成员 | 内容 |
|---|---|
| `agent.done`、`agent.steps` | 返回过 `STOP`（含到步数上限）后 `done` 为真；`act()` 抛异常后它**仍是假**，循环不能只看它（§9）。`steps` 是执行过的动作数（到步数上限时返回的那个 `STOP` 不计） |
| `agent.last_step` | 上一次 `act()`：`step`（收到的那帧的步号）、`action`、`source`（`queue` 取自队列 / `plan` 新规划的第一个 / `terminal` 服务端判停 / `empty` 空动作块 / `step_cap` 到步数上限）、`call`（这一步做了规划时的那次调用）、`timing_ms`、`vo_server_timing_ms` |
| `agent.last_call` | 最近一次规划：`kind`、`llm_output`（慢系统原文）、`actions`（整个动作块）、`pixel_goal`、`terminal`、`pose_ready`、`ppa_applied`、`vo_frame_id`、`vo_history_frame_ids`、`history_capture_steps`、`client_timing_ms`、`server_timing_ms`、`response`（完整响应） |
| `agent.calls` 等计数 | 本集的规划次数 `calls`、轨迹调用 `trajectory_calls`，以及轨迹调用里带历史认知的 `ppa_applied_calls` 和预热期的 `ppa_warmup_calls` |

`NavAgent(..., on_log=print)` 会打印与评测客户端相同格式的每次调用日志。客户端计时默认跟随环境变量
`HEATMAPVLN_TIMING`，也可以传 `timing=True`；计时不改变请求。

### 2.6 构造参数

```python
NavAgent(model, vo, *, protocol_seed=42, camera=CameraSpec(), timing=None,
         max_steps=500, rpc_timeout_ms=60000, jpeg_encoder=None, on_log=None)
```

| 参数 | 说明 |
|---|---|
| `model`、`vo` | `"host:port"`，或带 `infer_json` 的客户端对象（例如假服务端）。传地址时构造函数当场连上并核对两个服务端的身份和协议版本，不对就抛 `RuntimeError`，所以构造一次就是连通自检（§4） |
| `protocol_seed` | 采样种子，与评测相同取 42 |
| `camera` | 只能是 640×480；其余字段与仿真不同时只打一条 warning |
| `max_steps` | 步数上限，到了返回 `STOP`；里程计会话按 `max_steps + 1` 帧开 |
| `rpc_timeout_ms` | 每次 RPC 的超时，默认 60 s（评测用 600 s） |
| `jpeg_encoder` | `None` 用 vla_rpc 的编码器，与部署是同一个编码函数。接真服务端时不要换。字节完全相同还要求两边 OpenCV 同版本、PyTurboJPEG 都装或都不装（只在 4090 容器里核对过） |

### 2.7 网络上传什么

`NavAgent` 和两个服务端之间只有四种请求：

| 请求 | 发给 | 什么时候 | 内容 | 大小（仿真图实测） |
|---|---|---|---|---|
| `reset_episode` | 里程计 | `reset()` | 会话名 `scene_id/episode_id`、帧数上限 | 很小 |
| `ingest_frame` | 里程计 | 每次 `act()` | `front_rgb` 原图，JPEG 质量 95，640×480 | 约 70 KB |
| `query_relative_poses` | 里程计 | 每次规划 | 当前帧号与最多 8 个历史帧号 | 很小；回：位姿与 `pose_ready` |
| `plan_panoramic` | 模型 | 每次规划 | 指令、随机种子信息、位姿；当前与最多 8 个历史前视图（384×384）、1 张下视图（640×480）；当前帧和每个历史帧还各配 3 张黑色占位图（右、后、左，最多 27 张，见附录 B）；全部 JPEG 质量 90 | 约 350 KB；回：动作块与诊断字段 |

所以每步至少一个往返；规划的那一步再加两个往返（位姿查询、模型调用），共约 420 KB 上行。

---

## 3. 4090 侧：启动服务端

服务器是 ssh 别名 `6024_fjl` 那台（主机地址和账号向负责人要，仓库公开，不写在这里）：6 张 RTX 4090，
是 48 GB 显存的版本（普通 4090 是 24 GB，放不下同卡一套），多人共用。环境都在容器 `fjl-habitat` 里，
路径和权重见 [`../ops/deploy_rtx4090.md`](../ops/deploy_rtx4090.md) §1。

### 3.1 先决条件

- **代码**：部署检出 `/workspace/HeatmapVLN` 停在 `148e398`（2026-09-28），**没有** `scripts/deploy/` 和 `src/deploy/`。两种办法：
  1. 用 09-30 验证过的源码副本 `/workspace/verify_0930/src_94146fa`（附录 C 的逐调用比对就是用它跑的）。
  2. 按 `deploy_rtx4090.md` §4 的 git bundle 办法，把部署检出更新到含这两个目录的提交。

  不管用哪份，启动时都要把 `NAV_REPO` 设成它：启动脚本从 `NAV_REPO`（默认 `/workspace/HeatmapVLN`）
  加载两个服务端的代码。不设的话，两个服务端会从 `148e398` 那份旧检出加载：它的协议和身份与 `NavAgent` 一致，
  启动、构造、调用都不会报错，但跑的是 09-28 的服务端代码（没有之后的改动，例如服务端计时），从报错上看不出来。
- **显卡**：开跑前看 `nvidia-smi`，找一张没有别人进程的卡：两个服务端同卡约要 40 GiB（§8）。
  没有空卡时，用 `NAV_VO_GPU` 分到两张卡：模型约 24 GiB、里程计约 17 GiB，各自再留几百 MiB 给 CUDA context。
- **端口**：同一容器里在跑评测时（评测会在同一容器里并行起多套服务端，第 k 套占 52400+k / 52500+k），用 `NAV_MODEL_PORT` / `NAV_VO_PORT`
  换一对没人用的端口。gRPC 在 Linux 上默认开 `SO_REUSEPORT`，端口撞了不报错、而是两个进程分摊连接，
  机器人的 `reset()` 可能打到评测的里程计、顶掉评测正在跑的会话。启动脚本发现端口上已经有人监听时拒绝启动。

### 3.2 启动

在宿主机上开 tmux（前台的 `docker exec` 会随 ssh 断开一起退出），然后：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/verify_0930/src_94146fa && NAV_REPO=$PWD NAV_GPU=4 NAV_HOST=0.0.0.0 bash scripts/deploy/start_nav_servers_cuda.sh'
```

- 看到 `[nav-servers] READY model=0.0.0.0:52400 vo=0.0.0.0:52500` 后再连。启动要几十秒到几分钟（加载权重），超时 30 分钟。
- Ctrl-C 同时停两个服务端。日志在 `/workspace/nav_servers/<时间>_<pid>/` 的 `model.log`、`vo.log`（启动时会打印目录）。
- `NAV_HOST=0.0.0.0` 是为了让机器人侧经 ssh 隧道连进来（§4）；只在容器里自测时可以去掉。

| 变量 | 默认 | 作用 |
|---|---|---|
| `NAV_REPO` | `/workspace/HeatmapVLN` | 服务端代码从哪份检出加载（见 3.1） |
| `NAV_GPU` | `0` | 模型服务端的卡，也是里程计的默认卡 |
| `NAV_VO_GPU` | 同 `NAV_GPU` | 里程计的卡 |
| `NAV_HOST` | `127.0.0.1` | 两个服务端的绑定地址 |
| `NAV_MODEL_PORT` / `NAV_VO_PORT` | `52400` / `52500` | 端口 |
| `NAV_RUNTIME_DIR` | `/workspace/nav_servers/<时间>_<pid>` | 日志和缓存 |
| `NAV_SERVER_START_TIMEOUT_S` | `1800` | 等两个服务端就绪的上限（秒） |
| `HEATMAPVLN_TIMING` | 不设 | 设 `1` 时服务端逐阶段计时，每阶段做一次 CUDA 同步，只用于测延迟 |
| `NAV_CHECKPOINT`、`NAV_CONFIG` 等 | 部署权重 `/workspace/weights/ppa_refine_v2_best.pth`、`configs/ppa_action_refine_v2_8gpu.yaml` | 一般不改，其余见脚本开头的注释 |

**注意力实现**：这个启动脚本按部署配置走 SDPA（`model.llm.attn_implementation: sdpa`），§8 的延迟就是 SDPA 的数。
评测启动脚本从 2026-10-13 起默认 flash-attention 2：4 集实测模型服务端 3.5 s → 1.9 s、一次规划 5.1 s → 3.5 s，
但动作会变。这个启动脚本还没有这个开关。

---

## 4. 网络：让机器人侧电脑连到 4090

gRPC 没有加密和认证。`fjl-habitat` 在 docker 的 bridge 网络里、没有发布端口，所以服务端只绑 `127.0.0.1` 时，
只有容器里的进程连得上。机器人在别处时：

1. 容器里启动时加 `NAV_HOST=0.0.0.0`（§3.2 的命令已经带了）。容器没有发布端口，只有宿主机和同一 docker 网桥能访问，
   不会暴露到公网网卡；但这台机器多人共用，别人也连得到这两个无认证端口，**用完就停**。
2. 在 4090 宿主机上查容器 IP：

   ```bash
   docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fjl-habitat
   ```

3. 在机器人侧电脑上开隧道。先在它的 `~/.ssh/config` 里加一个 `Host 6024_fjl`（HostName、User、端口向负责人要），
   下面和 §5 的 ssh / scp 都用它；换了服务端端口就用换后的：

   ```bash
   ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L 52400:<容器IP>:52400 -L 52500:<容器IP>:52500 6024_fjl
   ```

4. 连通自检（先按 §5 装好环境）：在机器人侧电脑上构造一次 `NavAgent`，它会核对两个服务端的身份和协议版本，不对就抛异常：

   ```bash
   python -c "from src.deploy import NavAgent; NavAgent('127.0.0.1:52400', '127.0.0.1:52500').close(); print('ok')"
   ```

5. 用 `ping` 量一下机器人侧电脑到 4090 的往返：每步多付一次，每次规划再多付两次（§2.7）。

**不要把这两个端口发布或转发到公网网卡。**

---

## 5. 机器人侧电脑：环境与自检

- **机器选择**：
  - 建议背一台 x86 笔记本或 NUC，装 Ubuntu 20.04 / 22.04，网线接 Go2（§6.2），另有无线连 4090。
  - 也可以用 Go2 的扩展坞（Jetson Orin NX 16GB 或 Orin Nano 8GB，JetPack 5.1.1，aarch64，系统自带 Python 3.8），
    但要多两步：另装一个 Python 3.10（例如 miniforge）；宇树 SDK 锁定的 `cyclonedds==0.10.2` 没有 Linux aarch64 的现成包，
    要先从源码编译 C 版 CycloneDDS 0.10.x（`releases/0.10.x` 分支）再设 `CYCLONEDDS_HOME`。连 4090 的无线链路同样要配（§6.2）。
    没有在扩展坞上试过。
- **一个 Python 3.10 环境**，同时装 `NavAgent` 和宇树 SDK。`cyclonedds==0.10.2` 在 x86_64 Linux 上只有 Python 3.7–3.10 的现成包，
  3.11 及以上在任何平台都没有，所以定 3.10。`vla_rpc` 不在 PyPI 上，它在 4090 宿主机的 `/home/fangjialei/rpc`
  （容器里是 `/workspace/rpc`），先拷过来：

  ```bash
  scp -r 6024_fjl:/home/fangjialei/rpc ./vla_rpc_src
  pip install numpy pillow pytest
  pip install -e ./vla_rpc_src                 # 会装 grpcio、grpcio-tools、protobuf、opencv-python
  pip install -e <unitree_sdk2_python 源码目录>
  ```

  本仓库不用装，也不要装 `requirements` 里的训练依赖：`NavAgent` 只导入几个纯 Python 模块，不要 torch。
  检出一个含 `src/deploy/` 的提交（与 4090 上服务端那份同一个提交最好），在仓库根目录运行，或把它加进 `PYTHONPATH`。
- **自检（不要 GPU、不要机器人）**：

  ```bash
  python -m pytest tests/test_nav_agent.py --noconftest -q
  ```

  应为 23 passed。`--noconftest` 是必须的：`tests/conftest.py` 导入了 torch。2026-10-15 在 Python 3.10.17 +
  numpy 2.2.6 + Pillow 12.3 上实测通过（macOS arm64；x86 Linux 上还没跑过）。
- **假服务端**：没有 GPU 时先把机器人一侧跑通：`NavAgent(FakeModelServer(), FakeVOServer())`
  （`src/deploy/fake_servers.py`，按脚本返回动作，不代表模型行为）。还没装 vla_rpc 时加
  `jpeg_encoder=pil_jpeg_encoder`（三者都从 `src.deploy.fake_servers` 导入；PIL 编码的字节和部署不同，只能配假服务端用）。

---

## 6. 接到 Go2

**本节的状态**：依据是宇树官方文档、`unitree_sdk2_python` 源码（`814556d`，2026-09-21）和公开的 Go2 导航部署，
2026-10-15 逐条核对过（附录 D）。**没有在真机上跑过。** 标"待实测"的数和示意代码里的参数，都要在 §7 的步骤里实测后再定。

### 6.1 前提

- **型号**：Go2 **EDU**。AIR、PRO 不开放二次开发接口（规格表里的 Go2 X 写的是"基础支持"，没试过）。
- **固件**：在 Unitree App 里查软件版本。下面都按 **≥ V1.1.6** 写，即 2025-05 发布的 V2.0 运动接口
  （主运控服务叫 `mcf`；更早的固件叫 `sport_mode`，状态码的含义也不同）。`Move` 的范围是 vx −2.5~3.8 m/s、
  vy ±1.0 m/s、vyaw ±4 rad/s，我们只用到 ≤ 0.3 m/s、≤ 0.5 rad/s。
  低于 V1.1.6 时先升级固件：旧固件上 `error_code` 是错误码，示意代码按 `error_code` 判断状态的地方都不成立；
  `Move` 保持 1 s、不滤波这两条也只在 V2.0 文档里写过。
- **控制方式**：只用高层运动服务（`SportClient`）。不要碰底层电机控制：用它得先关掉主运控服务，否则两套控制同时下指令会失控。

### 6.2 网络

- 机器人侧电脑用网线接 Go2，IP 设在 `192.168.123.x`，宇树建议 `.222`（不能用 `.161`，那是 Go2 本体；扩展坞是 `.18`）。
- 宇树写明开发**只支持有线**，所以连 4090 要另走一条：笔记本自带的无线网卡，或在狗上加装路由器。
  公开部署也是这样做的：StreamVLN 的狗经 Wi-Fi 连远端 4090，作者建议在狗上加路由器提高带宽。
- 宇树不支持在 Go2 内置计算机（`.161`）上跑用户程序，SDK 也不支持在 Mac、Windows 上开发。
- **DDS 网卡**：`ChannelFactoryInitialize(0, "<网卡名>")`，网卡名是接 Go2 的那块（用 `ip addr` 查 `192.168.123.x` 那一块）。
  不传就自动选，多网卡的机器上要显式传。

### 6.3 两台相机 → `act()` 的三个输入

两台相机都固定安装、没有云台，所以"把相机压到 −30°"变成"下一帧改从下视相机取"：

| 接口 | 读哪台相机 |
|---|---|
| `act()` 的 `front_rgb` | 上一个动作的 `camera_pitch_deg` 是 0 时读前视相机；是 −30 时（上一个动作是 `LOOK_DOWN`）读下视相机 |
| `lookdown_fn()` | 下视相机 |
| `level_fn()` | 前视相机 |

两台相机的内参、分辨率、处理流程要一致。每张图先去畸变，再经过下面的 `to_model_frame`：

- 去畸变用 `cv2.undistort(img, K, dist)`，不传新的内参矩阵，输出仍按原内参 `K` 成像。
- 这时 `hfov_deg = 2·atan(w / (2·fx))`，`w` 是图宽，`fx` 取自 `K`。不要填产品页上的标称视场：填错不报错，只会裁错比例。
- 这里按图像中心裁剪，主点 `cx`、`cy` 要接近图像中心。

```python
import math

import cv2
import numpy as np


def to_model_frame(rgb: np.ndarray, hfov_deg: float) -> np.ndarray:
    """去畸变后的 RGB → 居中裁到水平 79°、4:3 → 640x480 uint8（§2.2）。视场不够时报错。"""
    h, w = rgb.shape[:2]
    cw = w * math.tan(math.radians(39.5)) / math.tan(math.radians(hfov_deg / 2))
    ch = cw * 3 / 4
    if cw > w or ch > h:
        raise ValueError("camera field of view smaller than 79 x 63.5 deg")
    x0, y0 = int(round((w - cw) / 2)), int(round((h - ch) / 2))
    crop = rgb[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
    return np.ascontiguousarray(cv2.resize(crop, (640, 480), interpolation=cv2.INTER_AREA))
```

保留的宽度占比是 tan 39.5° / tan(HFOV/2)，例如水平 90° 时约 0.824。高度按 4:3 跟着算，所以垂直视场也要够 63.5°：
16:9 的画面水平要 ≥ 约 95.5°；16:10、水平 90° 的画面（例如 VLingNav 用的 D457 1280×800）刚好够，
1280×720、水平 90° 的不够，函数会报错。

**为什么用外接相机，不用 Go2 自带的**：

- **视场对不上**：官方多媒体文档写自带相机 1280×720、水平 100°、垂直 56°、视频 15 Hz（FAQ 和产品页写"120°"，
  没说是哪个方向；内参没有找到）。按垂直 56° 算，裁成 4:3 后水平只剩约 71°，小于模型要的 79°，补不回来。
- **太低**：官方 URDF 里相机在机身原点前方 0.327 m、上方 0.043 m，水平朝前；机身离地约 0.33–0.36 m，
  所以相机离地约 0.37–0.40 m（待实测）。仿真是 1.25 m。
- **公开部署都用外接相机**：StreamVLN、Uni-NaVid 用 RealSense D455，VLingNav 用 D457，DyNaVLM 用 D435。
  给出安装高度的几家都在 0.85–1 m：InternNav（上海 AI Lab 的开源导航项目，我们的慢系统和快系统就是在它发布的模型上做的）
  建议 Go2 上约 1 m、下倾 15°，DyNaVLM 0.85 m、下倾 10°，SparseVideoNav 约 1 m。查过的 7 个部署里，
  没有一个报告装在 1.25 m 水平或下倾 30°（StreamVLN、Uni-NaVid、VLingNav 没有给高度和俯角）。SparseVideoNav 还专门写了
  这类基于大模型的导航对相机高度变化特别敏感。
- **InternNav 作者的经验**（issue #295、#324、#157）：
  - 相机高度、俯角和运动模糊对效果影响很大；动起来拍的模糊图会明显变差。
  - 训练时相邻两帧大约隔 25 cm 或 15°，真机输入要和它一致。我们每个动作做完、停稳再拍，正好对得上。
  - 他们早期也是一台水平、一台下倾的两台相机，下倾那台只在模型输出下视时用（和我们这套一样），
    发现成功率没有明显差别，后来改成只用下倾那台。他们能这样做，是因为训练数据混了多种视角；我们的模型没有，所以两台都要。

**安装建议**：

- 用支架把两台相机装到接近 1.25 m：前视水平，下视下倾 30°，尽量上下对齐，固定牢、避免高频振动。
- 选全局快门或运动模糊小的相机，动作做完、停稳后再拍。
- 装不到 1.25 m 时，记下实际高度和俯角，结果要按"视点与训练不同"来读。
- 相机装在机身上方时调用 `AutoRecoverySet(False)`：官方建议带负载时关掉跌倒自动翻身，免得翻身时压坏相机和支架。

### 6.4 离散动作 → Go2 运动

**接口事实**（V2.0，`unitree_sdk2_python`）：

- **`Move(vx, vy, vyaw)`**：机体坐标系速度（X 向前、Y 向左、Z 向上），发出后不等应答。
  返回 0 只表示消息发出去了，不代表机器人执行了；**没有对端时会阻塞到客户端超时、再返回 3102**，所以每次都要查返回码。
- **指令保持**：最新一条 `Move` 维持 1 s，运控不对它做滤波，官方建议自己滤波后再发；不动时发 `Move(0, 0, 0)` 或 `StopMove()`。
  所以一个动作执行期间要按固定频率重发。官方没给频率：StreamVLN、InternNav 的 Go2 客户端用 10 Hz，宇树的 C++ 示例用 200 Hz。
  官方也没写死区：速度太小时狗可能不动。
- **`StopMove()` 不只是停**：它还会把运动参数恢复成默认值，而且要等应答。所以动作之间用 `Move(0, 0, 0)` 停，
  `StopMove()` 只在一集结束、超时和异常时用。
- **先进入平衡站立再走**：
  - `StandUp()` 是"站立锁定"（`error_code` 1002），官方提醒锁久了电机容易过热。issue #175 报告只调 `StandUp()` 时
    `Move` 返回 0、狗却 4 秒到十几秒不动，`error_code` 一直停在 1002。
  - `BalanceStand()` 从站立锁定切到"平衡站立"（1013）；旧版 AI 运动服务文档也建议调 `Move` 前先调一次它。
  - 狗趴着、坐着或在阻尼状态时用 `RecoveryStand()`，官方写它会恢复到平衡站立。
  - 示意代码等 `error_code` 变成 1013 再发 `Move`。**拿它当"能走了"的信号是我们的推断，官方没有写**，第一次上机时记下实际看到的值。
- **状态**：订阅 `rt/sportmodestate`（`SportModeState_`，官方写 500 Hz）。平面位姿取 `position[0]`、`position[1]`
  和 `imu_state.rpy[2]`（yaw）；`error_code` 在 V2.0 里是当前模式，不是错误码；`stamp` 是时间戳。
- **位姿是腿式 + IMU 融合的里程计**，原点没有文档，所以每个动作都按"与起点之差"算。有第三方测量
  （户外、带激光雷达负载）说它转角准、平移尺度不准；在 0.25 m / 15° 这么小的步长上误差多大，没有数据，待实测。
- **方向**：按坐标系定义和右手定则，vyaw > 0 和 yaw 增大都是逆时针（向左转），但官方没有明写，§7 第 5 步要先确认。

**映射**（每个动作都从**当前实测位姿**起算）：

| `Action` | Go2 做什么 | 到位条件（待实测后调） |
|---|---|---|
| `FORWARD` | 沿当前朝向前进 0.25 m：按与起点的距离做 P 控制，同时用 vyaw 修正航向 | 距离误差 < 0.5 cm，或超时 |
| `TURN_LEFT` / `TURN_RIGHT` | 原地转 +15° / −15°（±0.2618 rad），按 yaw 误差做 P 控制 | 角度误差 < 0.5°，或超时 |
| `LOOK_DOWN` | 不动，下一帧从下视相机取（§6.3） | — |
| `STOP` | `StopMove()`，本集结束 | — |

- 到位后发 `Move(0, 0, 0)`，等机身停稳再拍下一帧。沉降时间待实测（示意代码先取 0.3 s）；公开的 Go2 客户端不等沉降，
  但它们不要求每帧都是静止视点，我们要（§2.3）。
- **要有最小速度**：只用比例控制时，速度随误差一起降到接近 0，每次都停在容差边上，少走的就是整个容差，还会一步步累加
  （只用比例控制、容差 2 cm / 1.5° 时，理想跟踪的模拟里 FORWARD 每次只走 0.230 m、转向只转 13.6°）。所以示意代码给速度设了下限
  （0.05 m/s、0.1 rad/s）、把容差收到 0.5 cm / 0.5°，并限制每个周期的速度变化（官方建议的滤波）。
  最小速度和容差要在 §7 第 5 步一起实测：最小速度太小狗不动，太大会冲过头；容差太紧时，真机有延迟会来回找位置，直到超时。
- 速度先保守：前进 ≤ 0.3 m/s，转向 ≤ 0.5 rad/s（InternNav 的上限是 0.6 m/s、0.5 rad/s，StreamVLN 是 1.0 m/s、1.2 rad/s）。
- **为什么从当前位姿起算**：StreamVLN 在上一个目标位姿上累加，残差不积累，但里程计漂移会带进来；
  InternNav 从当前里程计起算，没走到位的部分直接丢掉。仿真里每个动作都相对当前状态，所以这里按当前位姿写，并把每步实际走了多少记进日志。
- **走一步停一步**：公开的 Go2 部署端到端延迟大多在 0.3–1.3 s，我们每次规划要 2.5–5.3 s（§8），所以只能规划时原地站着等，
  拿到动作再走。StreamVLN 也是规划时站着等。

### 6.5 示意代码

**没有在真机上跑过**，只在一个按真实 SDK 签名写的模拟狗上跑通过（附录 C）。`front_cam` / `down_cam` 是你们自己的相机读取函数，
每次返回一张去畸变后的 RGB 图；`to_model_frame` 见 §6.3，放在同一个文件里。

```python
import math
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from src.deploy import Action, NavAgent

STAND_LOCKED, BALANCE_STAND = 1002, 1013     # V2.0 的 error_code：站立锁定、平衡站立


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def check(code: int, what: str) -> None:
    if code != 0:                            # 3102 = 没有对端、没发出去
        raise RuntimeError(f"{what} returned {code}")


def ramp(target: float, last: float, step: float) -> float:
    """每个周期最多变 step：官方建议自己滤波后再发 Move（§6.4）。"""
    return last + max(-step, min(step, target - last))


def p_speed(err: float, k: float, v_min: float, v_max: float) -> float:
    """比例控制，但不低于 v_min：官方没写死区，太慢狗可能不动（§6.4）。"""
    return math.copysign(min(v_max, max(v_min, k * abs(err))), err)


class Go2Base:
    """离散动作 → Go2 高层速度指令，20 Hz 闭环在 rt/sportmodestate 上。"""

    def __init__(self, nic: str, rate_hz: float = 20.0, settle_s: float = 0.3):
        ChannelFactoryInitialize(0, nic)
        self.dt, self.settle_s = 1.0 / rate_hz, settle_s
        self._msg, self._stamp, self._t = None, None, 0.0
        self._v = self._w = 0.0              # 上一次发出的速度，给 ramp 用
        self._lock = threading.Lock()
        self._sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._sub.Init(self._on_state, 10)
        self.sport = SportClient()
        self.sport.SetTimeout(10.0)
        self.sport.Init()
        check(self.sport.AutoRecoverySet(False), "AutoRecoverySet")   # 背着相机，关掉跌倒自动翻身（§6.3）
        mode = self._latest().error_code
        if mode == STAND_LOCKED:
            check(self.sport.BalanceStand(), "BalanceStand")          # 站立锁定 → 平衡站立
        elif mode != BALANCE_STAND:
            check(self.sport.RecoveryStand(), "RecoveryStand")        # 趴着、坐着、阻尼等 → 平衡站立
        self._wait_mode(BALANCE_STAND, timeout=10.0)  # 拿 1013 当"能走了"是推断，待实测（§6.4）
        self.sport.SetTimeout(1.0)           # 之后 Move 没有对端时最多阻塞 1 s

    def _on_state(self, msg):                # SDK 的读线程里调用，只存最新一帧
        stamp = (msg.stamp.sec, msg.stamp.nanosec)
        with self._lock:
            if stamp != self._stamp:
                self._stamp, self._t = stamp, time.monotonic()
            self._msg = msg

    def _latest(self, timeout: float = 2.0):
        t_end = time.monotonic() + timeout
        while True:
            with self._lock:
                msg = self._msg
            if msg is not None:
                return msg
            if time.monotonic() > t_end:
                raise RuntimeError("no rt/sportmodestate message")
            time.sleep(0.05)

    def _wait_mode(self, code: int, timeout: float):
        t_end = time.monotonic() + timeout
        while True:
            now = self._latest().error_code
            if now == code:
                return
            if time.monotonic() > t_end:
                raise RuntimeError(f"Go2 did not reach error_code {code}, now {now}")
            time.sleep(0.05)

    def pose(self):
        with self._lock:
            msg, t = self._msg, self._t
        if msg is None or time.monotonic() - t > 0.5:   # 时间戳 0.5 s 没往前走：当故障处理（issue #184 的症状）
            raise RuntimeError("rt/sportmodestate is stale")
        return msg.position[0], msg.position[1], msg.imu_state.rpy[2]

    def _send(self, v: float, w: float):
        check(self.sport.Move(v, 0.0, w), "Move")   # 指令只保持 1 s，所以每个周期重发
        self._v, self._w = v, w

    def _settle(self):
        self._send(0.0, 0.0)                 # Move(0,0,0)：停下，但不重置运动参数
        time.sleep(self.settle_s)            # 等停稳再拍下一帧

    def stop(self):
        """一集结束、超时或异常时用。StopMove 会把运动参数恢复成默认值（§6.4）。"""
        self.sport.StopMove()                # 急停路径上不查返回码，免得盖住原来的异常
        self._v = self._w = 0.0
        time.sleep(self.settle_s)

    def forward(self, dist=0.25, v_max=0.3, v_min=0.05, tol=0.005, timeout=6.0):
        x0, y0, th0 = self.pose()
        t_end = time.monotonic() + timeout
        while True:
            x, y, th = self.pose()
            err = dist - ((x - x0) * math.cos(th0) + (y - y0) * math.sin(th0))
            if abs(err) < tol:
                break
            if time.monotonic() > t_end:
                self.stop()
                raise RuntimeError(f"FORWARD timed out, {err:.3f} m left")
            v = ramp(p_speed(err, 1.5, v_min, v_max), self._v, 0.05)
            w = ramp(max(-0.3, min(0.3, 2.0 * wrap(th0 - th))), self._w, 0.1)
            self._send(v, w)
            time.sleep(self.dt)
        self._settle()

    def turn(self, deg, w_max=0.5, w_min=0.1, tol=math.radians(0.5), timeout=6.0):
        _, _, th0 = self.pose()
        target = wrap(th0 + math.radians(deg))
        t_end = time.monotonic() + timeout
        while True:
            err = wrap(target - self.pose()[2])
            if abs(err) < tol:
                break
            if time.monotonic() > t_end:
                self.stop()
                raise RuntimeError(f"TURN timed out, {math.degrees(err):.1f} deg left")
            self._send(0.0, ramp(p_speed(err, 2.0, w_min, w_max), self._w, 0.1))
            time.sleep(self.dt)
        self._settle()


def run_episode(instruction: str, run_id: int, base: Go2Base, front_cam, down_cam, hfov_deg: float):
    front = lambda: to_model_frame(front_cam(), hfov_deg)
    down = lambda: to_model_frame(down_cam(), hfov_deg)
    with NavAgent("127.0.0.1:52400", "127.0.0.1:52500", on_log=print) as agent:   # 超时用默认 60 s（§9）
        agent.reset(instruction, scene_id="go2lab", episode_id=run_id)   # 每次实跑换一个 run_id（§2.2）
        pitch = 0.0
        try:
            while not agent.done:
                frame = down() if pitch < 0 else front()                   # 上一个是 LOOK_DOWN 就读下视相机
                action = agent.act(frame, down, front)                     # lookdown_fn=下视，level_fn=前视
                pitch = action.camera_pitch_deg
                if action is Action.FORWARD:
                    base.forward(0.25)
                elif action is Action.TURN_LEFT:
                    base.turn(+15)
                elif action is Action.TURN_RIGHT:
                    base.turn(-15)
                elif action is Action.STOP:
                    base.stop()
                # LOOK_DOWN：车体不动
        finally:
            base.stop()                      # 任何异常都先停下；本集作废，重新 reset（§9）
```

### 6.6 安全

- **遥控器**：始终有人拿着 Go2 的遥控器跟在旁边。不要调 `SwitchJoystick(False)`：关掉之后推摇杆不再干预程序，遥控器就不能接管了。
- **软急停**：程序里的急停用 `StopMove()`。`Damp()` 让所有关节进入阻尼、优先级最高，官方定位是突发情况下的急停；
  关节只阻尼不保持，机身会趴下，只在意外时用。
- **看门狗**：示意代码里状态时间戳 0.5 s 不往前走就报错停车，单个动作超时 6 s；`Move` 只保持 1 s，程序挂了狗最多再走 1 s。
  有用户在 80 次运行里遇到 1 次运控卡死：里程计时间戳停住、遥控器全部失灵，只能断电（issue #184；他用的是 C++ SDK，
  经避障服务发速度）。所以看门狗要有，也要能断电。
- **碰撞**：模型没有任何避障，仿真里撞墙只是贴墙滑动。前几次在空旷房间、低速、短指令下跑。宇树另有避障服务
  （`unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client.ObstaclesAvoidClient`），但它可能改写或拒绝运动，
  和离散动作的语义不一致，先不要开。
- **步数上限**：`max_steps` 默认 500，到了返回 `STOP`（§2.3）。

---

## 7. 第一次上机的顺序

1. **4090 起服务端**（§3）。
2. **机器人侧装环境**，跑 `python -m pytest tests/test_nav_agent.py --noconftest -q`（§5）。
3. **开 ssh 隧道**，构造一次 `NavAgent` 做连通自检（§4）。
4. **只接相机，狗不动**：用假服务端跑通循环。存几帧经过 `to_model_frame` 的图，检查是不是 RGB、方向对不对、
   水平视场是不是 79°；检查前视、下视两台相机的画面是否上下对齐。
5. **运动标定，不接模型**：先用很小的速度确认 `turn(+15)` 是向左（逆时针）转、`forward` 是向前走。
   再在地上贴 0.25 m 刻度和 15° 角度线，`forward` 和 `turn` 各做 20 次，记录实测位移和转角的误差、
   每个动作的耗时、停稳所需时间，据此调最小速度、容差、沉降时间。
6. **接真服务端，狗不动**：动作只打印不执行，量规划延迟和网络往返，和 §8 比。
7. **接真服务端，开始动**：空旷房间、短指令、低速，有人拿遥控器跟着。
8. **每一步都留记录**，方便事后对照仿真：发给模型的图像、返回的动作；`last_call` 的 `kind`、慢系统原文、
   `pose_ready`；Go2 的里程计位姿、各段耗时。

---

## 8. 延迟、带宽与显存

**实测来源**：4090 上 R2R val_unseen 1839 集的全量评测（2026-10-01 → 10-02，56236 次规划、207970 个动作），
每张卡上模型服务端和里程计服务端同卡（与 §3 的默认一样），SDPA。完整分解见
[`../experiments/exp20-latency-report.md`](../experiments/exp20-latency-report.md)（实验编号 EXP-20）。
这次跑的是评测客户端，不是 `NavAgent`；两者的请求只差服务端不用的三张侧视图（附录 B），所以服务端和里程计的耗时可以直接用。
`NavAgent` 自己 09-30 接真服务端跑过 4 集、99 次调用，带历史认知的规划步 `act()` 中位 5067 ms，与下表一致。

**机器人每次规划要原地等多久**（规划关键路径 = 当前图采集 + 位姿查询 + 下视图采集 + 编码 + 模型调用；中位数）。
两次采集在仿真里是渲染，合计约 0.2 s；真机上换成读两次相机。

| 调用类型 | 次数 | 规划关键路径 | 其中模型服务端 |
|---|---:|---:|---:|
| 带历史认知的轨迹调用（稳态） | 24637 | **5258 ms**（P90 5965） | 3594 ms |
| 预热期轨迹调用（前 20 帧，不用历史认知） | 5443 | 3231 ms | 2965 ms |
| 只有慢系统（箭头或停止，不跑快系统） | 26156 | 2545 ms | 863 ms |

稳态那一行的分解（ms）：

| 阶段 | 中位数 | P90 |
|---|---:|---:|
| 当前图采集（仿真渲染；真机换成读相机） | 117 | 172 |
| 里程计位姿查询（含补建图） | 1359 | 1965 |
| 下视图采集（仿真渲染；真机换成读相机） | 92 | 210 |
| 编码 | 32 | 37 |
| 模型调用往返 | 3602 | 3775 |
| **规划关键路径** | **5258** | **5965** |
| 另：里程计写入（每步一次，往返） | 16 | 20 |

各阶段的中位数不能直接相加；关键路径那一行是逐次求和后再取的中位数。
模型服务端内部：慢系统两轮生成约 1.7 s、历史认知 0.58 s、快系统条件前向 0.85 s、快系统去噪采样 0.2 s。

**对真机意味着什么：**

- 平均每 3.7 个动作规划一次（207970 / 56236）。规划是同步的：机器人每 1–4 步原地等 2.5–5.3 s，拿到动作才能继续。
- 摊到每个动作的计算平均约 1.0 s（模型约 0.61 s、里程计约 0.37 s）。只看稳态的带历史认知调用约 1.3 s；
  EXP-20 报告里的"每动作 1.46 s"还含仿真渲染。
- 再加上：Go2 走 0.25 m / 转 15° 再停稳的时间（§7 第 5 步实测）、网络往返（每步一次，每次规划再加两次）。
  相机读取已按仿真采集的时间（约 0.2 s）算在上表里，真实相机更慢就按差值加。
- 取队列动作的 `act()` 只有一次里程计写入，通常十几毫秒；但第 20 帧那次写入会在服务端建第一张图，
  之后攒满 8 帧时写入也会顺带补建，所以偶尔会慢。
- 采样条数和去噪步数只影响快系统去噪采样那 0.2 s，降它们省不了多少延迟；大头在慢系统和视觉编码上。
  换 flash-attention 2，4 集实测模型服务端 3.5 s → 1.9 s（§3.2 末尾），这个启动脚本还没有这个开关。

**带宽**（Habitat 渲染图实测，真实相机的 JPEG 大小会不同）：每步里程计写入约 70 KB；有 8 帧历史时每次规划请求约 350 KB，
其中 9 张 384×384 前视图各约 24 KB、下视图约 46 KB、27 张黑色占位图各约 3 KB。

**显存**（`nvidia-smi` 显示的是 MiB，除以 1024 是 GiB）：全量里两个服务端每次请求的峰值保留中位数是模型 22.3 GiB
（最高约 23.6 GiB）、里程计 16.8 GiB，合计约 39 GiB，不含两个进程各自几百 MiB 的 CUDA context。
48 GB 的卡放得下一套，24 GB 的卡放不下。

---

## 9. 失败、超时与重置

- **每集开始调用 `reset()`**，它会重置里程计会话（地图丢弃）。
- **`act()` 抛出任何异常，本集就结束了**：机器人先停下，再 `reset()` 开新的一集。不能在同一集里重试：
  里程计那一侧已经记下了这一帧，重发会对不上帧号。之后再调 `act()` 会抛 `RuntimeError`，直到 `reset()`。
  `agent.done` 不会因异常变真，所以循环不能只看它：示意代码（§6.5）用 `try/finally` 先停狗，再把异常抛出去。
- 在 `reset()` 之前、或返回 `STOP` 之后调 `act()`，也抛 `RuntimeError`。
- **超时**：`rpc_timeout_ms` 默认 60 s，评测一直用 600 s。按 §8，模型调用的 P90 是 3.8 s、位姿查询 2.0 s，60 s 有十几倍余量；
  服务端刚启动后的第一次调用没有单独测过，第一次连上时可以先传 `rpc_timeout_ms=600000`。超时或网络错误时 `act()` 抛 `RuntimeError`
  （vla_rpc 对任何 gRPC 错误都只返回空）。服务端报错的具体原因在服务端日志里（§3.2）。
- **服务端重启或 ssh 隧道断开**：里程计会话丢失或连接中断，下一次 `act()` 会失败，同样停下、`reset()`。
- `NavAgent` 会核对服务端的回显（协议版本、采样种子记录、位姿是否就绪、是否只用了前视图），对不上就抛异常，不会带着错误的输入继续走。

---

## 附录 A 没有验证过的

- 任何真实场景：真实图像、光照、运动模糊、动态障碍、动作执行误差、碰撞。
- 与仿真不同的相机：HFOV、分辨率、畸变、安装高度、俯仰；用第二台相机拍下视图（仿真里是同一台相机转动）。
- 里程计在真实相机上的尺度（部署用固定的 `--translation-scale 1.0`），以及起步阶段只原地转圈时是否稳定。
- 超过 500 步的长任务（里程计会话按 501 帧开，服务端上限 4096）。
- 非英文或风格差别大的指令。
- 机器人到服务端的网络延迟和抖动。
- LOOK_DOWN 相关的路径只在假服务端闭环里测过：09-28 那 4 集里没有出现 LOOK_DOWN。
- §6.5 的示意代码只在模拟狗上跑过（附录 C），真实的 Go2 会有延迟、死区和打滑。
- `NavAgent` 在 x86 Linux 的 Python 3.10 上跑测试（3.10 只在 macOS arm64 上跑过，3.11 在 4090 容器里跑过）。

## 附录 B 与仿真评测客户端的差异

`NavAgent` 复刻 `scripts/evaluation/r2r_val_unseen.py` 的评测循环，有意的不同只有这些：

1. 右 / 后 / 左三个视角发黑色占位图。协议仍要求这三张，但部署配置下服务端只用前视图（慢系统提示、历史认知、
   快系统都只取前视图和下视图）；`NavAgent` 每次都核对服务端回显的 `native_front_only`，不为真就报错。
2. 每张前视图只编码一次（仿真客户端每次循环都重渲染、重编码全景）。同一像素、同一编码器，字节相同。
3. 同一步里若规划两次，下视图复用一次拍摄（部署服务端不会返回以 STOP 开头的非终止动作块，这种情况到不了）。
4. 到步数上限返回 `STOP`；仿真客户端只是不再走。
5. 更严的校验：只接受部署的运行时，未知动作码报错，输入图像尺寸不对报错；RPC 超时默认 60 s（评测用 600 s）。

LOOK_DOWN 后当场重新规划不算不同：仿真客户端这时会第二次采集全景，而相机已被第一次采集复位成水平，
`NavAgent` 用 `level_fn()` 取同一张水平图；送进里程计的帧和进历史的帧（同一步只取第一次采集，即 −30° 那张）两边相同。

## 附录 C 验证记录

**不占 GPU 的：**

- `tests/test_nav_agent.py`（假服务端，23 个）：把 09-28 在 4090 上跑过的那 4 集（共 99 次规划调用）的动作块喂回去，每次调用的步号、
  里程计帧号、历史帧号、地图版本号与部署日志逐一相同，总步数相同；另测预热、块内 STOP 重规划、完整轨迹块、LOOK_DOWN、
  LOOK_DOWN 后块内 STOP 的水平重拍、步数上限、8 动作保护、换集、图像检查、服务端违约、计时开关不改请求。
- `tests/test_nav_agent_habitat.py`（4090 容器，CPU）：同一组图像下，`NavAgent` 的规划请求与评测客户端自己构造的请求
  JSON 逐字相同、每张 JPEG 逐字节相同（13 次调用，其中 2 次是 LOOK_DOWN 后的水平重拍），里程计写入也逐字节相同；
  Habitat 闭环里用脚本化假服务端，评测客户端和 `NavAgent` 各跑一遍，2 集时 124 个请求除仿真渲染的侧视图外全部逐字节相同，
  逐调用日志与结局相同。另用一个 LOOK_DOWN 更密的脚本手工跑过 1 集：53 个请求全部相同。

**接真服务端（2026-09-30，GPU 4，源码副本 `94146fa`，SDPA；判据在跑之前写好）：通过。**
用 `scripts/deploy/start_nav_servers_cuda.sh` 起服务端，`scripts/deploy/nav_agent_habitat_check.py` 跑 09-28 那 4 集
（分两个进程各跑 2 集，种子 42），与 `/workspace/eval_runs/canary_cuda_seed42` 的客户端日志逐调用比对：两边都报 `identical`，
4 集依次 14/14、30/30、28/28、27/27 次调用相同（步号、类型、慢系统原文、动作块），每集结局相同，评测成绩（SPL、NE）与 09-28 那次逐位相同。
产物在 `6024_fjl:/home/fangjialei/verify_0930/navagent/shard_0{0,1}/`。所以能说的是："`NavAgent` 接真服务端时，
在这 4 集上与部署的评测客户端逐调用相同。"那 4 集里没有 LOOK_DOWN，这部分只由上面的假服务端闭环覆盖。

`NavAgent` 一侧的代码（`src/deploy/`、两个 RPC 协议文件、里程计客户端）从 `94146fa` 到现在没有改过；服务端代码之后有改动
（昇腾移植、计时、keepalive 等），同样 4 集的评测在之后的提交上也逐调用复现过（`docs/experiments/exp20-latency-report.md` §8）。

**§6.5 的示意代码（2026-10-15，模拟狗）**：把 §6.3、§6.5 的代码原样摘出来，配一个按 `unitree_sdk2_python`（`814556d`）签名写的
模拟 SDK（500 Hz 状态、机体系积分、`Move` 保持 1 s、没有对端时阻塞到超时返回 3102）和本仓库的假服务端，跑一集 35 步（含 5 次 LOOK_DOWN、
2 次水平重拍）：取图路由全对；FORWARD 每次 0.245–0.248 m，转向 14.5–14.8°，方向正确；狗趴着、站立锁定、已在平衡站立三种起始状态都能进入 1013；
到不了 1013、断开对端、状态时间戳冻结三种故障都会报错并停车。模拟狗是理想跟踪，不代表真机。

## 附录 D 出处与待实测

§6 的每条都在 2026-10-15 按下面的原始出处核对过。

- **宇树开发者文档**（doc-cdn 页面，直接给 markdown）：
  - 运动接口 V2.0（≥ V1.1.6）：<https://doc-cdn.unitree.com/6/814/zh/6_814_zh>：`Move` 范围、1 s 保持、不滤波、
    `StandUp` 锁定（默认站高 0.33 m）与 `BalanceStand` 解锁、`RecoveryStand`、`StopMove` 重置参数、
    `error_code` 模式表（1002 站立锁定、1013 平衡站立）、`Damp`、带负载关自动翻身
  - 旧版运动服务（< V1.1.6）：<https://doc-cdn.unitree.com/6/55/zh/6_55_zh>；旧版 AI 运动服务：<https://doc-cdn.unitree.com/6/219/zh/6_219_zh>
    （调 `Move` 前先调一次 `BalanceStand`）
  - 快速开始、网络：<https://doc-cdn.unitree.com/6/45/en/6_45_en>；扩展坞：<https://doc-cdn.unitree.com/6/110/zh/6_110_zh>
  - FAQ：<https://doc-cdn.unitree.com/6/60/en/6_60_en>（只支持有线、状态 500 Hz、EDU 才能开发）；规格：<https://doc-cdn.unitree.com/6/28/zh/6_28_zh>
  - 坐标系：<https://doc-cdn.unitree.com/6/33/zh/6_33_zh>；里程计：<https://doc-cdn.unitree.com/6/72/zh/6_72_zh>
  - 多媒体（相机）：<https://doc-cdn.unitree.com/6/53/zh/6_53_zh>
- **`unitree_sdk2_python`**：<https://github.com/unitreerobotics/unitree_sdk2_python>（`814556d`）
  - `go2/sport/sport_client.py`、`rpc/client_base.py`、`rpc/internal.py`（3102）、`core/channel.py`、
    `idl/unitree_go/msg/dds_/_SportModeState_.py`、`go2/obstacles_avoid/obstacles_avoid_client.py`、`setup.py`（`cyclonedds==0.10.2`）
  - issue #175（`StandUp` 后 `Move` 不动，`error_code` 停在 1002）、#184（运控卡死）
- `unitree_sdk2`（C++，`d1abf00`）：`example/go2/go2_sport_client.cpp`（控制周期 5 ms）、
  `example/go2/go2_stand_example.cpp`（站姿关节角，配 URDF 推算机身离地约 0.36 m）
- PyPI `cyclonedds` 0.10.2 的文件列表：<https://pypi.org/project/cyclonedds/0.10.2/#files>
- Go2 URDF：<https://github.com/unitreerobotics/unitree_ros>（`5994d4f`），`robots/go2_description/urdf/go2_description.urdf` 的 `front_camera_joint`
- **公开的 Go2 导航部署**：
  - StreamVLN `realworld/go2_vln_client.py`（`e48f6ff`）；论文 arXiv 2507.05240；issue #54、#60
  - InternNav `scripts/realworld/`（`7a5c624`）；issue #295、#324、#157
  - Uni-NaVid arXiv 2412.06224 附录 XII；VLingNav arXiv 2601.08665 §6.3.1；DyNaVLM arXiv 2506.15096 §V-A；
    SparseVideoNav arXiv 2602.05827 §IV；NaVILA arXiv 2412.04453 附录 C4
- Go2 里程计精度的第三方测量：arXiv 2506.09548（v2）

**待实测**（文档里查不到）：

- 固件版本；
- 相机实际视场、内参和安装高度；
- vyaw 的符号（应为逆时针为正）；
- `Move` 合适的重发频率和最小有效速度（官方没有写死区）；
- 动作后的沉降时间；
- 0.25 m / 15° 上的里程计误差；
- 机器人到 4090 的网络往返。
