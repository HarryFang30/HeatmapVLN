# 实机部署说明：4090 服务器 + 宇树 Go2

给要把导航模型接到宇树 Go2 上的同事。机器狗一侧走实验室的 ROS 2 底座 `robot_platform_ws`：相机和运动都由底座提供，
本文写的是我们这个项目怎么接上去。一页纸的摘要见 [`README.md`](README.md)。

> **状态（2026-10-23）**：没有在任何真机上跑过。机器人侧客户端 `NavAgent` 接 4090 上的真服务端时，
> SDPA 和 flash-attention 2 下都在 4 集仿真上与部署的评测客户端逐调用相同（附录 C）。接底座的一侧是脚本
> `scripts/deploy/run_go2.py`（§6.5），只在模拟的底座客户端上跑通过。相机（2026-10-23 定）：用 Go2 自带的前视，不加硬件，
> 前视图和下视图都用它的同一帧（§6.2）。仿真里这个视角比训练视角低约 10 pt SR。

| 代码 | 作用 |
|---|---|
| `src/deploy/nav_agent.py` | 机器人侧客户端 `NavAgent`，纯 Python，不要 GPU、不要 torch |
| `src/deploy/fake_servers.py` | 不要 GPU 的假服务端，用来先把机器人一侧跑通 |
| `scripts/deploy/run_go2.py` | 狗上跑一集：取自带前视、调 `NavAgent`、把动作交给底座（§6.5） |
| `scripts/deploy/start_nav_servers_cuda.sh` | 4090 侧：在一张卡上起模型服务端和里程计服务端 |
| `robot_platform.ros_client.ROSPlatformClient` | 底座提供（不在本仓库）：狗一侧的取图、控制权和相对动作 |

**目录**

1. 系统组成
2. 输入与输出 ← 接口契约，接机器人前先读这一节
3. 4090 侧：启动服务端
4. 网络：让 Orin 连到 4090
5. Orin：环境与自检
6. 接到底座和 Go2
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
狗上的 Orin（Ubuntu 20.04 · ROS 2 Foxy · Python 3.8）
    我们的进程：NavAgent + 底座的 ROS 客户端 ROSPlatformClient
        │  ROS 2，同一个 domain
    底座 robot_platform_ws：/platform/cameras/*、/platform/relative_motion、控制权
        │                                   │
        │ 宇树接口（DDS）                    │ USB
        ▼                                   ▼
Go2 EDU：运动 + 自带前视相机 front_camera（约 0.4 m、水平，§6.2）
                                        左侧环视 left_camera（只做记录）
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

`robot.*` 在这里是示意，写的是仿真里的语义（相机能转）。接底座时取图和动作都走底座的 ROS 客户端，相机是狗自带的固定前视：
三个输入传同一帧，`pitch` 用不上，实际的写法是 `scripts/deploy/run_go2.py`（§6.5）。

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
| `lookdown_fn()` | 只在这一步要规划时调，每步最多一次（约每 1–4 步一次） | 仿真里是同一位姿、**相对水平下倾 30°** 的图。慢系统在这张图上找像素目标，快系统也拿它当输入，不能省。用一台已经固定下倾的相机时，就传这一步的同一帧（InternNav 实机的做法，§6.2）；不要传一张水平的图 |
| `level_fn()` | 只在一种情况下调：上一个动作是 `LOOK_DOWN`，而这一步从队列里取到 STOP、当场重新规划。每步最多一次，先于 `lookdown_fn()` | 仿真里是同一位姿、**水平**的图。单台固定相机时也传同一帧 |

三张图怎么取，看相机怎么装：
- 一台带云台的相机（仿真就是这样）：`lookdown_fn` 压到 −30° 拍，`level_fn` 水平拍，返回前都把相机恢复水平。
- 一台固定下倾的相机（接底座的方案，§6.2）：三个输入都传这一步的同一帧，`LOOK_DOWN` 时什么都不用做。

**三张图的格式都一样：**

| 项 | 要求 | 说明 |
|---|---|---|
| 类型 | `numpy.ndarray`，`dtype=uint8`，形状 `(480, 640, 3)`，**RGB** 顺序 | 不是数组抛 `TypeError`，形状或 dtype 不对抛 `ValueError`。**不会替你缩放**。OpenCV 读出的是 BGR，要先 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，否则不报错但结果错 |
| 视场 | 水平 79°（4:3 时垂直约 63.5°），针孔模型 | 仿真值。真实相机要先去畸变，再居中裁到水平 79°、4:3，最后缩放到 640×480（`to_model_frame`，§6.4）。水平小于 79°、或垂直小于 63.5° 的相机补不回来；16:9 的画面受限的是垂直方向，水平要 ≥ 约 95.5° |
| 安装 | 离地 1.25 m，朝正前方；前视水平，下视下倾 30° | 仿真值。实机用狗自带前视，约 0.4 m、水平（§6.2） |
| 曝光等 | 无要求 | 仿真里没有运动模糊、自动曝光；真机效果未知 |

### 2.3 输出

`act()` 每次返回一个 `Action`（`IntEnum`，值就是服务端的动作码）：

| `Action` | 值 | 机器人做什么 | 下一帧 `front_rgb` 怎么拍 |
|---|---|---|---|
| `FORWARD` | 1 | 沿当前朝向直行 0.25 m | 水平 |
| `TURN_LEFT` | 2 | 原地逆时针转 15° | 水平 |
| `TURN_RIGHT` | 3 | 原地顺时针转 15° | 水平 |
| `LOOK_DOWN` | 5 | 车体不动 | 仿真里下倾 30°；单台固定下倾相机时照常取 |
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

- **代码**：服务端要从 `ded4bd3` 或之后的提交加载：`NAV_ATTN_IMPL` 开关在这个提交里，模型服务端的 `--attn_implementation`
  从 `f61e9f7` 起才有。4090 上已经备好这样一份源码副本 `/workspace/nav_src_ded4bd3`（2026-10-15 用 `git archive` 导出，附录 C 的
  核验就是用它跑的）。部署检出 `/workspace/HeatmapVLN`（`148e398`）和 09-30 的副本 `/workspace/verify_0930/src_94146fa` 都太旧。
  要换更新的提交时，在本地检出里：

  ```bash
  git archive <提交> | ssh 6024_fjl 'mkdir ~/nav_src_<提交> && tar -x -C ~/nav_src_<提交>'
  ```

  容器里就是 `/workspace/nav_src_<提交>`。启动时要把 `NAV_REPO` 设成这份副本：启动脚本从 `NAV_REPO`（默认 `/workspace/HeatmapVLN`）
  加载两个服务端的代码。不设的话会加载 `148e398` 那份旧检出，它不认 `--attn_implementation`，模型服务端起不来，启动脚本报
  `model server exited` 并打出日志尾部。
- **显卡**：开跑前看 `nvidia-smi`，找一张没有别人进程的卡：两个服务端同卡约要 40 GiB（§8）。
  没有空卡时，用 `NAV_VO_GPU` 分到两张卡：模型约 24 GiB、里程计约 17 GiB，各自再留几百 MiB 给 CUDA context。
- **端口**：同一容器里在跑评测时（评测会在同一容器里并行起多套服务端，第 k 套占 52400+k / 52500+k），用 `NAV_MODEL_PORT` / `NAV_VO_PORT`
  换一对没人用的端口。gRPC 在 Linux 上默认开 `SO_REUSEPORT`，端口撞了不报错、而是两个进程分摊连接，
  机器人的 `reset()` 可能打到评测的里程计、顶掉评测正在跑的会话。启动脚本发现端口上已经有人监听时拒绝启动。

### 3.2 启动

在宿主机上开 tmux（前台的 `docker exec` 会随 ssh 断开一起退出），然后：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/nav_src_ded4bd3 && NAV_REPO=$PWD NAV_GPU=4 NAV_HOST=0.0.0.0 bash scripts/deploy/start_nav_servers_cuda.sh'
```

- 看到 `[nav-servers] READY model=0.0.0.0:52400 vo=0.0.0.0:52500 attn=flash_attention_2` 后再连。启动要几十秒到几分钟
  （加载权重；2026-10-15 实测约 20 秒），超时 30 分钟。
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
| `NAV_ATTN_IMPL` | `flash_attention_2` | 模型服务端的注意力实现，设 `sdpa` 复现 09-30 的核验。服务端拿不到就拒绝启动；启动脚本还要在日志里看到确认才报 `READY` |
| `HEATMAPVLN_TIMING` | 不设 | 设 `1` 时服务端逐阶段计时，每阶段做一次 CUDA 同步，只用于测延迟 |
| `NAV_CHECKPOINT`、`NAV_CONFIG` 等 | 部署权重 `/workspace/weights/ppa_refine_v2_best.pth`、`configs/ppa_action_refine_v2_8gpu.yaml` | 一般不改，其余见脚本开头的注释 |

**注意力实现**：2026-10-15 起默认 flash-attention 2，与评测启动脚本（`PPA_EVAL_ATTN_IMPL`）相同。同样 4 集上它把模型服务端
从 3.5 s 降到 1.9 s、一次规划从 5.1 s 降到 3.5 s（§8）。它和 SDPA 的 bf16 数值不同，同一集里的走法会变；那 4 集的成败不变。

---

## 4. 网络：让 Orin 连到 4090

我们的进程跑在狗上的 Orin 上（§5）。gRPC 没有加密和认证。`fjl-habitat` 在 docker 的 bridge 网络里、没有发布端口，
所以服务端只绑 `127.0.0.1` 时，只有容器里的进程连得上。从 Orin 连过去：

1. 容器里启动时加 `NAV_HOST=0.0.0.0`（§3.2 的命令已经带了）。容器没有发布端口，只有宿主机和同一 docker 网桥能访问，
   不会暴露到公网网卡；但这台机器多人共用，别人也连得到这两个无认证端口，**用完就停**。
2. 在 4090 宿主机上查容器 IP：

   ```bash
   docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fjl-habitat
   ```

3. 在 Orin 上开隧道。先在 Orin 的 `~/.ssh/config` 里加一个 `Host 6024_fjl`（HostName、User、端口、跳板向负责人要；
   底座的 `examples/remote_inference/ssh_config` 有 PanaV 连远端 GPU 时的跳板写法可以照抄格式），下面和 §5 的 ssh / scp 都用它；
   换了服务端端口就用换后的：

   ```bash
   ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L 52400:<容器IP>:52400 -L 52500:<容器IP>:52500 6024_fjl
   ```

4. 连通自检（先按 §5 装好环境）：在 Orin 上构造一次 `NavAgent`，它会核对两个服务端的身份和协议版本，不对就抛异常：

   ```bash
   python -c "from src.deploy import NavAgent; NavAgent('127.0.0.1:52400', '127.0.0.1:52500').close(); print('ok')"
   ```

5. 量一下经隧道的往返（4090 走跳板时直接 `ping` 量的不是这条路）：每步多付一次，每次规划再多付两次（§2.7）。

   ```bash
   time python -c "from src.deploy import NavAgent; NavAgent('127.0.0.1:52400', '127.0.0.1:52500').close()"
   ```

**不要把这两个端口发布或转发到公网网卡。** 这条从容器外经隧道连进来的路径还没有端到端测过：4090 上的核验（附录 C）都是在容器里连的。

---

## 5. Orin：环境与自检

我们的进程跑在狗上的 Orin 上，也就是底座 `robot_platform_ws` 所在的那台（Ubuntu 20.04、ROS 2 Foxy、Linux 用户 `irmv`，
底座在 `/home/irmv/robot_platform_ws`），和底座在同一个 ROS domain 里。取图和运动都通过底座的 ROS 客户端（§6），
所以这一侧不需要宇树 SDK，也不需要 x86 电脑。

- **Python 用系统的 3.8**：底座的 ROS 客户端依赖 Foxy 的 `rclpy`，Foxy 只支持系统自带的 Python 3.8，`NavAgent` 和它在同一个进程里，
  所以也是 3.8。numpy 1.24 是支持 3.8 的最后一版。
- **环境要和底座一致**：`RMW_IMPLEMENTATION`（底座用 `rmw_cyclonedds_cpp`）和 DDS 网卡（默认 `eth0`）在底座的
  `config/deployment.env` 里；ROS domain 不在里面，`./platform` 沿用当前终端的 `ROS_DOMAIN_ID`（没设就是 0），
  所以启动底座和跑我们进程的终端里 `ROS_DOMAIN_ID` 要一样。最省事的是用底座的 `./platform shell` 开一个已经加载好
  ROS、宇树和底座环境的 shell，在里面跑我们的进程。
- **装法**：单独建一个 venv。rclpy 和底座的包不在系统 site-packages 里，是 `./platform shell` 通过 `PYTHONPATH` 加进来的，
  所以每次都要先进 `./platform shell`，再激活这个 venv，再 `cd` 到本仓库根目录（`./platform shell` 打开后停在底座目录，
  那里也有 `src/` 和 `tests/`）。在普通终端里激活这个 venv 是 import 不到 rclpy 的。`--system-site-packages` 是为了用上
  系统里 apt 装的依赖。`vla_rpc` 不在 PyPI 上，它在 4090 宿主机的 `/home/fangjialei/rpc`（容器里是 `/workspace/rpc`），
  用 §4 第 3 步配好的 `6024_fjl` 拷过来：

  ```bash
  cd /home/irmv/robot_platform_ws && ./platform shell      # ROS、宇树、底座环境都已加载
  python3 -m venv --system-site-packages ~/heatmapvln_venv && source ~/heatmapvln_venv/bin/activate
  python -m pip install --upgrade pip          # 20.04 自带的 pip 20.0 认不出 grpcio 的 aarch64 现成包，会改成从源码编译
  scp -r 6024_fjl:/home/fangjialei/rpc ~/vla_rpc_src
  pip install --upgrade "numpy==1.24.4" "pillow==10.4.0" pytest   # 和附录 C 实测的版本一致；不加 --upgrade 时系统里的旧版不会被换掉
  pip install -e ~/vla_rpc_src                 # 会装 grpcio、grpcio-tools、protobuf、opencv-python
  cd <本仓库根目录>
  ```

  `python3 -m venv` 报 ensurepip 不可用时，要找底座的维护者装 `python3.8-venv`（需要 sudo）。
  本仓库不用装，也不要装 `requirements` 里的训练依赖：`NavAgent` 只导入几个纯 Python 模块，不要 torch。
  检出 `14ac629` 或之后的提交（Python 3.8 上的自检要用到这次对 `fake_servers.py` 的修正；4090 上 `ded4bd3` 那份的
  `nav_agent.py` 与它相同，发出的请求没有差别），在仓库根目录运行，或把它加进 `PYTHONPATH`。
- **自检（不要 GPU、不要机器人）**：

  ```bash
  python -m pytest tests/test_nav_agent.py --noconftest -q
  ```

  应为 23 passed。`--noconftest` 是必须的：`tests/conftest.py` 导入了 torch。2026-10-16 在 Python 3.8.20 + numpy 1.24.4 +
  Pillow 10.4.0 + vla_rpc（grpcio 1.70、protobuf 5.29、OpenCV 5.0）上实测通过，用 vla_rpc 的编码器跑一集假服务端也通过
  （macOS arm64；还没在 Orin 上跑过）。
- **假服务端**：没有 GPU 时先把这一侧跑通：`NavAgent(FakeModelServer(), FakeVOServer())`
  （`src/deploy/fake_servers.py`，按脚本返回动作，不代表模型行为）。还没装 vla_rpc 时加
  `jpeg_encoder=pil_jpeg_encoder`（三者都从 `src.deploy.fake_servers` 导入；PIL 编码的字节和部署不同，只能配假服务端用）。

---

## 6. 接到底座和 Go2

**本节的状态**：机器狗一侧走实验室的 ROS 2 底座 `robot_platform_ws`（下称"底座"）。依据是底座 2026-10-12 版的 README、VALIDATION、
`docs/project_integration.md`、`docs/remote_inference.md`、`config/platform.yaml` 和 `src/robot_platform/robot_platform/ros_client.py`
（底座的工作区，不在本仓库）；相机方案照 InternNav / DualVLN 在 Go2 上的实机做法（附录 D）。底座本身在真机上验证过取图和相对动作，
PanaV 项目用它在真机上走完过一条完整路线；**我们的模型还没有接上去跑过**，§6.5 的脚本只在模拟的底座客户端上跑通过（附录 C）。

### 6.1 底座给什么，我们用什么

| 底座接口 | 我们怎么用 |
|---|---|
| `/platform/cameras/front_camera/frame`（狗自带前视，客户端 `get_frame('front_camera')`） | 裁剪缩放后（§6.4），`front_rgb`、`lookdown_fn()`、`level_fn()` 都用它的同一帧 |
| `/platform/relative_motion`（`move_relative`） | 执行 `FORWARD`、`TURN_LEFT`、`TURN_RIGHT` |
| `/platform/acquire_control`、`heartbeat`、`release_control` | 一集开始拿控制权，结束或出错时释放；心跳由客户端的后台线程每 0.25 s 发一次 |
| `/platform/stop` | 人工急停 |
| `/platform/odom`、`/platform/observation`、环视 `left_camera` | 不进模型：位姿由 4090 上的里程计服务端从图像估计；环视只用于记录 |

- 底座默认只读。只有用 `./platform run --allow-motion` 启动、而且我们拿到了控制权，底座才会让狗动。
- **运动只走底座**：底座独占宇树的运动出口。不要在狗上另跑宇树 SDK、`sport_mode_ctrl` 或旧导航节点去发运动指令，会和底座争控制。
- 同一时刻只有一个项目能持有控制权：PanaV 在跑时我们拿不到，反之亦然。
- 这和 PanaV 的接法一样：图像和动作走底座的 ROS 接口，模型推理走另一条路（PanaV 用 HTTP，我们用 gRPC 隧道，§4）。

### 6.2 相机：用狗自带的前视

2026-10-23 定：直接用 Go2 自带的前视摄像头，不加任何硬件。10-21 定的 D455 支架方案改成备选，放在本节最后。

- 底座已经接好这一路：`get_frame('front_camera')`，1920×1080 的 JPEG，5 fps，打开后预热 5 s。
- 离地约 0.4 m（按 URDF 推算，附录 D），水平朝前，俯角调不了。
- 宇树给的视场是水平约 100°（附录 D）。模型要的是水平 79°、4:3、640×480，所以每一帧都从中间裁一块再缩小（§6.4）。
- 前视、下视、水平重拍三个输入都传同一帧。InternNav 在 Go2 上也是一台相机、同一帧当下视（附录 D）。模型给出 `LOOK_DOWN` 时狗不动。
- 这台相机没有标定过，裁剪按宇树标称的 100° 算。以后标定出来不是 100°，改 `run_go2.py --hfov` 就行。

**仿真对比（EXP-33，2026-10-23）**：在 Habitat 里按 Go2 能装的方式摆相机，跑部署的完整模型，看 R2R-CE val_unseen 上掉多少。

| 相机 | 下视图 | SR | 比 1.25 m 训练视角（同集配对，95% CI） |
|---|---|---|---|
| 1.25 m、水平（训练和评测用的视角，参照） | 原地再下倾 30° 另拍 | 66.8 | — |
| **0.4 m、水平**（Go2 自带前视，现在的方案） | 同一帧 | 57.4 | **−9.8 pt [−14.9, −4.8]** |
| 0.6 m、下倾 15°（InternNav 训练视角 `60cm_15_15`） | 同一帧 | 62.0 | −4.8 pt [−9.6, 0.0] |
| 0.7 m、下倾 15°（InternNav 官方支架） | 同一帧 | — | 只跑了 88 集，和 0.6 m 分不出来 |

- 自带前视在仿真里比训练视角低约 10 pt，比装到 0.6 m、下倾 15° 低 4.6 pt [+0.4, +9.0]。用户 10-23 定：这点损失可以接受，先用自带前视。
- 设置：500 集固定子集，种子 42，4090，flash-attention 2，视角改在仿真客户端里（见 EXP-33 台账）。10-23 提前停，1.25 m 跑了 437 集、0.7 m 跑了 88 集。
- 仿真里的视角不是真图：纹理、光照、运动模糊、这台相机的畸变、行走时机身的晃动都不在里面。

**备选：以后想多拿这几个点时**，加一台相机装高一点、往下斜：Intel RealSense D455（全局快门），离地 0.6–0.7 m、下倾约 15°，固定牢。
可以打印 InternNav 的 `go2_stand.STEP` 加一个 15° 楔块，也可以用一根杆加一个球形云台。D455 接在 `camera2` 的 USB 口，底座配置里关掉
`camera2` 和 `side_panorama`、加一路 640×480 的 `nav_camera`，跑脚本时加 `--camera nav_camera --hfov 79`。
详细的装法和底座配置写在本文 `62db74c` 版的 §6.2。

### 6.3 动作：交给底座的相对动作

底座在 `relative` 模式下自己做闭环（50 Hz、PID，到位容差平移 3 cm、旋转 1°，速度够低持续 0.12 s 才算完成），
我们只提交距离或角度，不用自己写控制器。角度左转为正、右转为负，单位 rad。

| `Action` | 底座调用 | 底座实测（2026-10-08 调好 PID 后，单次） |
|---|---|---|
| `FORWARD` | `robot.move_relative(distance=0.25)` | 走了 0.241 m，用时 2.36 s；侧移约 2.6 cm，航向变了 +0.65° |
| `TURN_LEFT` | `robot.move_relative(angle=math.radians(15))` | 转了 15.008°，2.42 s |
| `TURN_RIGHT` | `robot.move_relative(angle=-math.radians(15))` | 转了 15.202°，2.24 s |
| `LOOK_DOWN` | 不调用；相机是固定的，下一帧照常取 | — |
| `STOP` | 不调用；本集结束，`release()` | — |

- `move_relative` 默认等动作做完再返回（`wait=True`，超时 30 s），返回 `{'task_id', 'state', 'reason'}`。
  `state` 不是 `succeeded` 就按本集失败处理（§9）。
- 底座的限速是平移 0.2 m/s、旋转 0.3 rad/s（在底座 `config/platform.yaml` 的 `motion` 里）；相对动作的平移在底座代码里
  另外限到 0.15 m/s，改配置调不了。
- 动作做完后取下一帧，要连取两次 `get_frame(..., wait_new=True)`、用第二帧。`wait_new` 只保证这一帧是调用之后才收到的，
  不保证是调用之后才拍的：底座每 0.2 s 读一帧、每 0.2 s 发布一次，收到的帧可能是零点几秒前拍的。第二帧晚一个周期，
  加上底座要停稳 0.12 s 才判定动作完成，一般已经是停稳以后的视点，但底座不保证。
- 动作是开环的：底座走到容差内就算完成，差的那点模型只能从下一帧看出来（§2.3）。
- 相机固定，`LOOK_DOWN` 不改变下一帧。模型连续给出 `LOOK_DOWN`（中间没有别的动作），说明它在这个视角下找不到要去的地方；
  脚本连续 3 次就按本集失败处理（§9）。
- 底座的实测是不带负载、一种地面、每种动作几次。我们不加硬件，负载没变；长 episode 和别的地面上的重复性底座还没验证。

### 6.4 图像预处理

底座给的是 1920×1080 的 JPEG，`act()` 要 640×480 的 RGB `uint8`（§2.2）。`scripts/deploy/run_go2.py` 的 `to_model_frame` 每帧做三件事：

1. 用 Pillow 解码成 RGB（不经过 OpenCV，所以不会有 BGR 的问题）。
2. 从中间裁一块水平 79°、4:3 的区域：宽度占比是 tan 39.5° / tan(HFOV/2)，HFOV 取 100° 时是 1920 里的 1328 列，高 996 行。
3. 缩放到 640×480。

没有去畸变：这台相机没有标定，InternNav 用 D455 时也是直接用原图。16:9 的画面要裁出 4:3、79° 的一块，水平视场至少要约 95.5°；
`--hfov` 填小于这个数时脚本直接报错。

### 6.5 运行脚本 `scripts/deploy/run_go2.py`

一次跑一集。要在 `./platform shell` 里、激活 §5 的 venv、`cd` 到本仓库根目录后运行，4090 的服务端和隧道已经起好（§3、§4）。

```bash
python scripts/deploy/run_go2.py --offline                      # 不要狗、不要 4090：检查环境装好了没有
python scripts/deploy/run_go2.py --dry-run -i "Walk to the door and stop."   # 真相机、真模型，狗不动，只打印动作
python scripts/deploy/run_go2.py -i "Walk to the door and stop."             # 狗会动
```

每一步它做的事：

1. 连取两帧 `front_camera`，用第二帧（§6.3），裁剪缩放（§6.4）。
2. 交给 `NavAgent.act()`，前视、下视、水平重拍都传这一帧。
3. `FORWARD` → `move_relative(distance=0.25)`；`TURN_LEFT` / `TURN_RIGHT` → `move_relative(angle=±15°)`；`LOOK_DOWN` 不动；`STOP` 结束。
4. `move_relative` 不成功、连续 3 次 `LOOK_DOWN`、任何异常（包括 Ctrl-C）都结束本集，并且一定释放控制权。

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i` / `--instruction` | — | 英文指令，R2R 那样的说法 |
| `--run-id` | 当前时间（秒） | 每次实跑要不一样，它决定模型采样的随机种子（§2.2） |
| `--dry-run` | 关 | 不拿控制权、不发动作，只取图、规划、打印 |
| `--offline` | 关 | 不连狗、不连 4090：用一张测试图和假服务端跑完一集，最后打印 `offline check OK` |
| `--fake-servers` | 关 | 连狗（或底座的模拟后端），模型换成假服务端，只测接线 |
| `--model` / `--vo` | `127.0.0.1:52400` / `:52500` | 经隧道的两个服务端 |
| `--camera` / `--hfov` | `front_camera` / `100` | 用哪一路相机、它的水平视场 |
| `--max-steps` | 500 | 到了就停 |
| `--out` | `runs/go2/<时间>_<run id>/` | 记录目录 |

每次运行的记录：

- `frames/step_NNNN.jpg`：每一步送进模型前的原始 JPEG，原样保存，没有重新编码。
- `steps.jsonl`：每步一行，写着动作、这一步有没有规划（慢系统原文、动作块、像素目标、位姿是否就绪）、底座返回的结果，以及取图、模型、动作各用了几秒。
- `summary.json`：怎么结束的、走了几步、用了多久。`outcome` 的取值：
  - `model_stop`：模型认为到了；
  - `fallback_stop` 或 `empty_chunk`：模型没给出能执行的动作；
  - `step_cap`：到了步数上限；
  - `failed`：出错或被中断，`error` 里写着原因。

测试：`tests/test_run_go2.py`（附录 C）。没有在真底座和真机上跑过。

### 6.6 安全

- **先在模拟里跑通**：`./platform run --backend sim --allow-motion` 和实机是同一套 ROS 接口，模拟后端会给 `config/platform.yaml` 里每一路已启用的相机（包括 `front_camera`）发 256×192 的占位图（§7 第 4 步）。
- **遥控器**：始终有人拿着 Go2 的遥控器跟在旁边。
- **急停**：`ros2 service call /platform/stop std_srvs/srv/Trigger {}`，立即停止并释放控制权；程序里是 `robot.stop()`。
  这条命令要在另一个 `./platform shell` 里执行（开跑前先开好放着），保证和底座同一个 RMW、网卡和 domain。
- **掉线保护在底座里**：控制租约 1 s（客户端每 0.25 s 发心跳），我们的进程挂掉或断开后，底座在租约到期时清零输出。
  出错时要取消动作、`release()`，并确认 `/platform/status` 里的 `output` 为零。
- **心跳线程会悄悄退出**：底座客户端的心跳只要有一次调用超时（0.5 s）或出错，后台线程就退出，不会重启。1 s 后租约到期，
  底座清零输出、收回控制权，下一个 `move_relative` 返回 `failed`（`reason` 以 `invalid_session:` 开头；动作进行中到期则是
  `lease_expired`），`/platform/status` 的 `last_reason` 是 `lease_expired`。按本集失败处理就行，下一集 `acquire()` 会重新起心跳。
- **碰撞**：模型没有任何避障，底座的第一版也不做绕障。前几次在空旷房间、低速、短指令下跑。
- **不要同时开别的运动程序**（§6.1）。
- **步数上限**：`max_steps` 默认 500，到了返回 `STOP`（§2.3）。

---

## 7. 第一次上机的顺序

不加任何硬件，从第 1 步开始。

1. **4090 起服务端**（§3）。
2. **Orin 装环境**（§5）：先按 §4 第 3 步在 `~/.ssh/config` 里加好 `Host 6024_fjl`，装好以后跑
   `python -m pytest tests/test_nav_agent.py tests/test_run_go2.py --noconftest -q`（31 passed），
   再跑 `python scripts/deploy/run_go2.py --offline`，最后一行是 `offline check OK`。
3. **开 ssh 隧道**，在 Orin 上构造一次 `NavAgent` 做连通自检（§4）。
4. **模拟底座 + 假服务端**：先停掉正在跑的实机底座（两者抢同一把锁），`./platform run --backend sim --allow-motion`，
   再 `python scripts/deploy/run_go2.py --fake-servers -i "Walk to the door."`，确认取图、动作、释放控制权都通。
   模拟后端给的是 256×192 的占位图，这一步只测接线。
5. **实机只读，只取图**：`./platform run`（不带 `--allow-motion`），`./platform cameras --cameras front_camera` 确认出图。
   再跑一次 `--dry-run --fake-servers`，打开 `runs/go2/<这次>/frames/` 里的图看一下：画面是正的、颜色正常、能看到前方的地面和远处。
6. **实机只读 + 真服务端**：`python scripts/deploy/run_go2.py --dry-run -i "..."`，狗不动，动作只打印。
   看 `steps.jsonl` 里每步的取图、模型用时，和 §8 比。底座没开运动不影响取图。
   模型一直输出 `↓` 或 `LOOK_DOWN`、给不出像素目标时，脚本连续 3 次就停；这说明它在这个视角下认不出路，换个起点或更短的指令再试。
7. **接真服务端，开始动**：`./platform run --allow-motion`，`python scripts/deploy/run_go2.py -i "..."`。
   空旷房间、短指令，有人拿遥控器跟着，另一个 `./platform shell` 里备好急停命令（§6.6）。
   底座的相对动作已经在不带负载时实测过（§6.3），我们没加硬件，不用重新标。
8. **每一集的记录**在 `runs/go2/<时间>_<run id>/`（§6.5），跑完整个目录拷走就行。要第三视角，另外录 `left_camera` 或手机拍。

---

## 8. 延迟、带宽与显存

**实测来源**：4090 上 R2R val_unseen 1839 集的全量评测（2026-10-01 → 10-02，56236 次规划、207970 个动作），
每张卡上模型服务端和里程计服务端同卡（与 §3 的默认一样），SDPA。完整分解见
[`../experiments/exp20-latency-report.md`](../experiments/exp20-latency-report.md)（实验编号 EXP-20）。
这次跑的是评测客户端，不是 `NavAgent`；两者的请求只差服务端不用的三张侧视图（附录 B），所以服务端和里程计的耗时可以直接用。
`NavAgent` 自己 09-30 接真服务端跑过 4 集、99 次调用，带历史认知的规划步 `act()` 中位 5067 ms，与下表一致。

**机器人每次规划要原地等多久**（规划关键路径 = 当前图采集 + 位姿查询 + 下视图采集 + 编码 + 模型调用；中位数）。
两次采集在仿真里是渲染，合计约 0.2 s；真机上只在 `act()` 之前读一次相机（连取两帧），下视用同一帧，不另取图。

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
| 下视图采集（仿真渲染；单台固定相机时不另取图） | 92 | 210 |
| 编码 | 32 | 37 |
| 模型调用往返 | 3602 | 3775 |
| **规划关键路径** | **5258** | **5965** |
| 另：里程计写入（每步一次，往返） | 16 | 20 |

各阶段的中位数不能直接相加；关键路径那一行是逐次求和后再取的中位数。
模型服务端内部：慢系统两轮生成约 1.7 s、历史认知 0.58 s、快系统条件前向 0.85 s、快系统去噪采样 0.2 s。

**flash-attention 2（启动脚本的默认）**：上面的全量是 SDPA 跑的。2026-10-15 用默认参数起机器人服务端，`NavAgent` 跑同样那 4 集
（82 次规划，附录 C），规划步的 `act()` 中位数如下（含下视图取图，仿真里约 0.1 s）：

| 调用类型 | 次数 | `act()` | 其中模型调用往返 | 其中位姿查询 |
|---|---:|---:|---:|---:|
| 带历史认知的轨迹调用（稳态） | 45 | **3316 ms** | 1865 ms | 1334 ms |
| 预热期轨迹调用 | 13 | 1789 ms | 1626 ms | 1 ms |
| 只有慢系统 | 24 | 1153 ms | 436 ms | 548 ms |

省下的全在模型那边（SDPA 全量的模型调用往返是 3602 ms），里程计不受影响。这一列只有 4 集、一个场景，和上面全量不是同一批样本；
同样 4 集上评测客户端的对照见 `exp20-latency-report.md` §9（模型服务端 3499 → 1866 ms、规划 5108 → 3472 ms）。

**对真机意味着什么：**

- 平均每 3.7 个动作规划一次（207970 / 56236）。规划是同步的：机器人每 1–4 步原地等，拿到动作才能继续。
  默认的 flash-attention 2 下稳态约 3.3 s（SDPA 是 5.3 s），只跑慢系统的调用约 1.2 s（SDPA 2.5 s）。
- 摊到每个动作的计算：SDPA 全量平均约 1.0 s（模型约 0.61 s、里程计约 0.37 s），flash-attention 2 只缩短模型那部分，约减半。
  EXP-20 报告里的"每动作 1.46 s"是 SDPA 稳态、还含仿真渲染。
- 再加上：底座执行一个动作约 2.2–2.4 s（§6.3，含停稳判断）、动作后连取两帧新图（按 5 fps 约 0.4 s，没实测），以及网络往返
  （每步一次，每次规划再加两次）。仿真采集的时间（约 0.2 s）已经算在上表里。
- 粗算：每个动作约 2.3 s 运动 + 摊下来约 0.7 s 规划（上一条的 1.0 s 里模型那 0.6 s 在 flash-attention 2 下约减半）
  + 约 0.4 s 取图，一集 100 个动作大约 5–6 分钟。
- 取队列动作的 `act()` 只有一次里程计写入，通常十几毫秒；但第 20 帧那次写入会在服务端建第一张图，
  之后攒满 8 帧时写入也会顺带补建，所以偶尔会慢。
- 采样条数和去噪步数只影响快系统去噪采样那 0.2 s，降它们省不了多少延迟；大头在慢系统和视觉编码上，
  flash-attention 2 省的正是这一块（上表）。

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
  `agent.done` 不会因异常变真，所以循环不能只看它：`run_go2.py`（§6.5）用 `try/finally` 先释放控制权（底座随即清零输出），再把异常抛出去。
- 在 `reset()` 之前、或返回 `STOP` 之后调 `act()`，也抛 `RuntimeError`。
- **超时**：`rpc_timeout_ms` 默认 60 s，评测一直用 600 s。按 §8，模型调用的 P90 是 3.8 s、位姿查询 2.0 s，60 s 有十几倍余量；
  服务端刚启动后的第一次调用没有单独测过，第一次连上时可以先传 `rpc_timeout_ms=600000`。超时或网络错误时 `act()` 抛 `RuntimeError`
  （vla_rpc 对任何 gRPC 错误都只返回空）。服务端报错的具体原因在服务端日志里（§3.2）。
- **服务端重启或 ssh 隧道断开**：里程计会话丢失或连接中断，下一次 `act()` 会失败，同样停下、`reset()`。
- **底座那一侧出错**：取图超时（`RosPlatformError: fresh frame timed out`）、动作没有成功（`state` 不是 `succeeded`）、
  控制权丢失，都按本集失败处理：`release()`，确认输出为零，新的一集重新 `reset()`。底座的预检失效（例如桥接失联）时
  底座自己停下并拒绝后续动作，按底座的约定不要自动恢复旧任务。
  控制权已经丢了时 `release()` 会报 `invalid_session`，这本身不说明有问题，原因看 `/platform/status` 的 `last_reason`。
- `NavAgent` 会核对服务端的回显（协议版本、采样种子记录、位姿是否就绪、是否只用了前视图），对不上就抛异常，不会带着错误的输入继续走。

---

## 附录 A 没有验证过的

- 任何真实场景：真实图像、光照、运动模糊、动态障碍、动作执行误差、碰撞。
- **我们的模型还没有接到底座上跑过**。
- **相机视点和训练不同**：训练和评测是 1.25 m 水平前视加同一位置下倾 30° 的下视；实机是狗自带前视，约 0.4 m、水平，下视用同一帧。
  仿真里这样换视角，SR 低约 10 pt（§6.2 的仿真对比）；真实图像上会差多少，只能上机看。
- **自带前视没有标定**：视场按宇树标称的水平 100° 裁到 79°（§6.4），没有去畸变；实际视场和畸变都没量过。
- 里程计在真实相机上的尺度（部署用固定的 `--translation-scale 1.0`），以及起步阶段只原地转圈时是否稳定。
- 超过 500 步的长任务（里程计会话按 501 帧开，服务端上限 4096）。
- 非英文或风格差别大的指令。
- 机器人到服务端的网络延迟和抖动。
- LOOK_DOWN 相关的路径只在假服务端闭环里测过：09-28 那 4 集里没有出现 LOOK_DOWN。
- §6.5 的脚本只在模拟的底座客户端上跑过（附录 C），没有接过真底座。
- Orin 上的 Python 3.8 环境：3.8 只在 macOS arm64 上测过，还没在 Orin（aarch64）上装过。
- 从容器外经 ssh 隧道连服务端：4090 上的核验都是在容器里连的。

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

`nav_agent.py`、两个 RPC 协议文件、里程计客户端从 `94146fa` 到现在没有改过（`fake_servers.py` 只改了一处类型写法，见下面
Python 3.8 一条，不影响发出的请求）；服务端代码之后有改动
（昇腾移植、计时、keepalive 等），同样 4 集的评测在之后的提交上也逐调用复现过（`docs/experiments/exp20-latency-report.md` §8）。

**接 flash-attention 2 的机器人服务端（2026-10-15，GPU 4，源码副本 `/workspace/nav_src_ded4bd3`）：通过。**
用默认参数起 `start_nav_servers_cuda.sh`（日志有 `Attention implementation confirmed: flash_attention_2`，`READY ... attn=flash_attention_2`），
`nav_agent_habitat_check.py` 跑同样 4 集，与 10-13 用评测启动脚本跑的 flash-attention 2 那一臂（`/workspace/lat_1013/arm_fa`，源码 `ace1ff8`；
两边的服务端代码只差 `f61e9f7` 加的开关和核对）逐调用比对：13/13、20/20、28/28、21/21 次调用相同，每集结局和评测成绩相同。
那一臂开着服务端计时、这次没开，动作仍然相同。产物在 `/workspace/nav_servers/fa2check_ded4bd3_navagent/`。
调用数和 09-30 的 SDPA 核验不同（82 对 99），是因为两种注意力本来就走法不同（§3.2）。

**§6.5 的脚本 `run_go2.py`（2026-10-23，Python 3.8.20、numpy 1.24.4、Pillow 10.4.0，macOS arm64）**：
`tests/test_run_go2.py` 8 个和 `tests/test_nav_agent.py` 23 个都通过。用一个按底座 `ros_client.py`（2026-10-12 版）的方法名和返回值
写的替身（1920×1080 的 `front_camera`、`move_relative` 返回 `state`）和本仓库的假服务端：
- 裁剪：1920×1080、水平 100° 时居中裁 1328×996，缩到 640×480，是 RGB；视场不够时报错。
- 一集 35 步（29 个运动动作、5 次 `LOOK_DOWN`、最后 `STOP`）：每步连取两帧、用第二帧；29 个运动动作各对应一次方向正确的 `move_relative`，
  `LOOK_DOWN` 和 `STOP` 不发动作；开头拿控制权、结束释放；每步的原始 JPEG 原样存下。
- `--dry-run`：不拿控制权、不发动作，照样规划完一集。
- 动作返回 `failed`：抛出这个错误，控制权照样释放，`summary.json` 记 `failed`。
- 模型每次都给 `LOOK_DOWN`：连续 3 次后结束本集，没有发任何动作，控制权照样释放。
- `--offline` 一条命令跑完一集，打印 `offline check OK`。

替身不代表真底座的时序和误差。10-21 那版示意代码（D455 方案）的同类核验见本文 `62db74c` 版。

**Python 3.8（2026-10-16，macOS arm64）**：`tests/test_nav_agent.py` 23 个、`tests/test_nav_servers_launcher.py` 4 个都通过
（numpy 1.24.4、Pillow 10.4.0）；装上 vla_rpc（grpcio 1.70、protobuf 5.29、OpenCV 5.0）后用它的编码器跑一集假服务端也通过。
为此把 `src/deploy/fake_servers.py` 里一处只有 3.9 才支持的类型写法改成了 `typing.Dict`（`14ac629`）。

## 附录 D 出处与待实测

- **底座 `robot_platform_ws`**（2026-10-12 版，实验室的工作区，不在本仓库）：
  - `README.md`：相机列表、`CameraFrame`、ROS 接口总览、控制权与模式、相对动作、急停；
  - `VALIDATION.md`：前视相机接入（1920×1080、videohub）、两路环视、相对动作 PID 实测（0.25 m、±15°）、PanaV 实机全路线；
  - `docs/project_integration.md`、`docs/remote_inference.md`、`config/platform.yaml`、`config/deployment.env`；
  - `src/robot_platform/robot_platform/ros_client.py`（`get_frame`、`acquire`、`move_relative`、心跳）、`examples/panav_adapter.py`。
- **宇树开发者文档**：多媒体（自带相机 1280×720、水平 100°、垂直 56°）：<https://doc-cdn.unitree.com/6/53/zh/6_53_zh>；
  FAQ：<https://doc-cdn.unitree.com/6/60/en/6_60_en>；运动接口 V2.0（带负载时建议关掉自动翻身）：<https://doc-cdn.unitree.com/6/814/zh/6_814_zh>
- **Go2 URDF**：<https://github.com/unitreerobotics/unitree_ros>（`5994d4f`），`robots/go2_description/urdf/go2_description.urdf` 的
  `front_camera_joint`（机身原点前方 0.327 m、上方 0.043 m）；`unitree_sdk2`（`d1abf00`）`example/go2/go2_stand_example.cpp`
  的站姿推算机身离地约 0.36 m，官方文档写默认站高 0.33 m，所以相机离地约 0.37–0.40 m。
- **InternNav / DualVLN 的实机做法**（相机方案的依据）：
  - 论文 arXiv 2512.08186 §5.2（Turtlebot4、Go2、G1，D455"装在不同高度、下倾 15°"，远端 RTX 4090）；InternVLA-N1 技术报告 §5.5 同样的说法；
  - 仓库 <https://github.com/InternRobotics/InternNav>（`7a5c624`）：`assets/3d_printing_files/go2_stand.STEP`、`downwards_slope.3mf`（15° 楔块）；
    `scripts/realworld/http_internvla_server.py`（模型输出 `↓` 时用同一帧再跑一次；D455 内参 640×480 下 fx = fy = 386.5）；
    `assets/realworld_sample_data.tar.gz`（38 张下视图都和前一帧逐字节相同）；
    `internnav/dataset/internvla_n1_lerobot_dataset.py`（`125cm_0_30`、`60cm_15_15`、`60cm_30_30` 等视角）、
    `scripts/train/qwenvl_train/train_dual_system.sh`（训练混合这几种视角）；
  - 维护者在 issue 里的说法：#194（支架约 0.7 m、楔块约 15°）、#295（Go2 建议约 1 m；从两台相机改成只用下倾那台）、
    #327（一台小车：约 1 m、要有下倾；另有用户说 Go2 不抬高相机效果还可以）、#175（G1：至少 0.6 m、全局快门、要有下倾）、
    #148（一直输出 ↓：先查 ↓ 之后有没有真的取下视图，也可能是相机太平）、#157（下倾 15°；D435i 运动模糊大，用 D455）、
    #109（D455 全局快门、D435 卷帘快门）、#343（斜装后对不上螺孔，用双面胶贴）、#329（楔块用双面胶贴）、
    #335（装牢防振）、#337（用户相机约 50 cm 时效果差，维护者没确认原因）、#352（多高度数据是为了 Go2、G1 部署）；
    #194 里还有用户反映 `downwards_slope.3mf` 打不出来，维护者建议自己画三角楔块。
- **公开的 Go2 导航部署**（相机型号和安装高度）：StreamVLN `realworld/go2_vln_client.py`（`e48f6ff`）、论文 arXiv 2507.05240；
  Uni-NaVid arXiv 2412.06224 附录 XII；
  VLingNav arXiv 2601.08665 §6.3.1；DyNaVLM arXiv 2506.15096 §V-A；SparseVideoNav arXiv 2602.05827 §IV；NaVILA arXiv 2412.04453 附录 C4。
  给出高度的几家都在 0.85–1 m、下倾 10–15°；查过的 7 家都没报告装在 1.25 m 水平或下倾 30°。

**待实测：**

- 自带前视 1920×1080 画面的实际水平视场和畸变（现在按宇树标称的 100° 裁剪）；
- 底座相对动作 0.25 m / ±15° 在我们场地上的误差和耗时；
- Orin 到 4090 的网络往返；
- Orin 上的 Python 3.8 环境能否装上 vla_rpc 并通过 `tests/test_nav_agent.py`、`tests/test_run_go2.py`。
