# 导航模型部署接口（NavAgent）

给要把导航模型接到真机上的同事。代码：`src/deploy/nav_agent.py`；服务端启动：`scripts/deploy/start_nav_servers_cuda.sh`。

## 0. 这是什么

一个视觉语言导航模型：输入一句自然语言指令和机器人前视相机的 RGB 图像，逐步输出离散动作（前进 0.25 m、左/右转 15°、相机下视、停止），直到它输出停止。只用 RGB，不需要深度、里程计或定位：自身相对位姿由一个视觉里程计服务从 RGB 序列在线估计。模型只在 Habitat 仿真（R2R-CE）里评测过，**从没在真机上跑过**。

## 1. 架构

```
机器人程序 ──act(front_rgb, lookdown_fn, level_fn)──▶ NavAgent（机器人侧，纯 Python，无 GPU）
                                                        │ gRPC（vla_rpc：JSON + JPEG，不加密、无认证）
                                       ┌────────────────┴────────────────┐
                              模型服务端（GPU）                     里程计服务端（GPU）
                 慢系统（生成像素目标/箭头）+ 快系统（轨迹）          AMB3R 在线建图，给出历史帧相对位姿
                            + 历史认知头与注入
```

| 状态 | 放在哪 |
|---|---|
| 历史帧、动作队列、步数、每次调用的随机种子 | NavAgent（每集 `reset()` 清空） |
| 地图与轨迹 | 里程计服务端（一个会话，每集 `reset()` 时重置） |
| 无 | 模型服务端（每次调用自带全部输入，互不相关） |

**一对服务端只能服务一台机器人**：里程计服务端同一时刻只有一个会话，第二个 `reset()` 会顶掉第一个；两个服务端都是单线程串行处理。

## 2. 启动服务端

在 4090 机器的 `fjl-habitat` 容器里（路径、权重同 `docs/ops/deploy_rtx4090.md`）。先决条件：

- 部署检出 `/workspace/HeatmapVLN` 里还没有 `scripts/deploy/`、`src/deploy/`：先按 `docs/ops/deploy_rtx4090.md` §4 的 git bundle 办法把它更新到含这两个目录的提交。
- 开跑前先看 `nvidia-smi`，只用空卡。两个服务端默认同卡，约 36–37 GB 显存（这台机器是 48 GB 版 4090，普通 24 GB 的卡放不下）。`NAV_VO_GPU` 可把里程计放到另一张卡；分两张卡时各自占多少没测过。
- 容器里同时在跑评测时（评测第 k 个槽位占 52400+k / 52500+k），必须用 `NAV_MODEL_PORT` / `NAV_VO_PORT` 换一对没人用的端口。gRPC 在 Linux 上默认开 SO_REUSEPORT，端口撞了不会报错，而是两个进程分摊连接，机器人的 `reset()` 可能打到评测的里程计服务端、顶掉评测正在跑的会话。启动脚本发现端口上已经有人监听时拒绝启动。

在宿主机的 tmux 里跑（前台的 `docker exec` 会随 ssh 断开一起退出）：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/HeatmapVLN && NAV_GPU=4 bash scripts/deploy/start_nav_servers_cuda.sh'
```

- 看到 `[nav-servers] READY model=… vo=…` 后再连；Ctrl-C 同时停两个。
- 要服务端计时（§8）时在命令里加 `HEATMAPVLN_TIMING=1`。服务端会在每个阶段做 CUDA 同步，只用于测延迟。
- 其余参数（路径、超时）见脚本开头的注释。

**机器人怎么连进来。** gRPC 没有加密和认证。服务端默认只绑 127.0.0.1，而 `fjl-habitat` 在 docker 的 bridge 网络里、没有发布端口，所以这时只有容器里的进程能连，宿主机上的 `ssh -L …:127.0.0.1:52400` 也连不上。机器人在别处时：

1. 容器里启动时加 `NAV_HOST=0.0.0.0`。容器没有发布端口，只有宿主机和同一 docker 网桥能访问，不会暴露到公网网卡；但这台机器多人共用，别人也能连到这两个无认证端口，用完就停。
2. 在宿主机上查容器 IP：`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fjl-habitat`。
3. 机器人侧：`ssh -N -L 52400:<容器IP>:52400 -L 52500:<容器IP>:52500 <4090 宿主机>`（换了端口就用换后的），之后 NavAgent 连本机的 `127.0.0.1:52400` 和 `127.0.0.1:52500`。

不要把这两个端口发布或转发到公网网卡。

## 3. 接口

```python
from src.deploy import Action, NavAgent

agent = NavAgent("127.0.0.1:52400", "127.0.0.1:52500")   # 模型服务端、里程计服务端
agent.reset("Walk past the sofa and stop at the kitchen door.", scene_id="lab3f", episode_id=17)

while not agent.done:
    front = robot.capture_front()              # 640x480x3 uint8 RGB，当前位姿
    action = agent.act(front, robot.capture_lookdown, robot.capture_level)
    robot.set_camera_pitch(action.camera_pitch_deg)   # 0 = 水平；LOOK_DOWN 为 -30
    if action is Action.FORWARD:
        robot.drive_straight(0.25)
    elif action is Action.TURN_LEFT:
        robot.rotate_in_place(+15)             # 逆时针
    elif action is Action.TURN_RIGHT:
        robot.rotate_in_place(-15)             # 顺时针
    elif action is Action.STOP:
        robot.halt()                           # 本集结束，agent.done 为真
    # LOOK_DOWN：只动相机，车体不动
    robot.wait_until_still()                   # 动作做完、停稳后再拍下一帧

agent.close()
```

`robot.capture_lookdown()` 和 `robot.capture_level()` 的要求见 §4。

| 成员 | 说明 |
|---|---|
| `NavAgent(model, vo, *, protocol_seed=42, camera=CameraSpec(), timing=None, max_steps=500, rpc_timeout_ms=60000, jpeg_encoder=None, on_log=None)` | `model`、`vo` 是 `"host:port"`；构造时检查两个服务端的身份与协议版本，不对就抛异常。`jpeg_encoder=None` 用 vla_rpc 的编码器，与部署逐字节相同，接真服务端时不要换。`on_log=print` 会打印与评测客户端相同格式的每次调用日志。 |
| `reset(instruction, *, scene_id, episode_id)` | 开始一集。英文指令（评测只用过 R2R 英文指令；骨干 InternVLA-N1 发布权重的训练语料本仓库没有记录）。`scene_id`（不含 `/`）和 `episode_id`（非负整数）连同调用序号决定快系统的随机噪声：同一组键加同样的输入会复现同样的动作，所以每次实跑换一个 `episode_id`。 |
| `act(front_rgb, lookdown_fn, level_fn) -> Action` | 一次调用 = 机器人执行一个动作。输入见 §4，输出见 §5。 |
| `done`、`steps` | 本集是否已结束（返回过 STOP）；已返回的动作数。 |
| `last_step` | 上一次 `act()`：`action`、`source`（`queue` 执行队列里的动作 / `plan` 新规划的第一个 / `terminal` 服务端判停 / `empty` 服务端返回空动作块、执行 STOP / `step_cap` 到步数上限）、`call`、`timing_ms`、`vo_server_timing_ms`。 |
| `last_call` | 最近一次规划调用：`kind`、`llm_output`（慢系统原文）、`actions`（整个动作块）、`pixel_goal`、`pose_ready`、`ppa_applied`、里程计帧号、`client_timing_ms`、`server_timing_ms`、`response`（完整响应）。 |
| `close()` | 关闭连接（也可用 `with NavAgent(...) as agent:`）。 |

没有 GPU 时可以先用假服务端把机器人这一侧跑通：`NavAgent(FakeModelServer(), FakeVOServer())`（`src/deploy/fake_servers.py`，按脚本返回动作，不代表模型行为）。这需要已装 vla_rpc；还没装时用 `NavAgent(FakeModelServer(), FakeVOServer(), jpeg_encoder=pil_jpeg_encoder)`（三者都从 `src.deploy.fake_servers` 导入；PIL 编码的字节和部署不同，只能配假服务端用）。

## 4. 输入要求

**前视图 `front_rgb`**：每次 `act()` 一张，是上一个动作做完后、在当前位姿拍的。

| 项 | 要求 | 说明 |
|---|---|---|
| 格式 | `numpy.ndarray`，`uint8`，形状 `(480, 640, 3)`，RGB 顺序 | 不是数组抛 `TypeError`，形状或 dtype 不对抛 `ValueError`，**不会替你缩放**。OpenCV 读出的是 BGR，要先 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，否则不会报错但结果错。 |
| 视场 | 水平 FOV 79° | 仿真值。其他相机要先去畸变成针孔模型，再居中裁到水平 79°（保留的宽度占比 = tan 39.5° / tan(HFOV/2)，例如 HFOV 90° 时约 0.824）、4:3（对应垂直约 63.5°），最后缩放到 640×480；水平视场小于 79° 的相机补不回来。都没验证过。 |
| 安装 | 离地 1.25 m，朝正前方，水平（俯仰 0°） | 仿真值。 |
| 曝光等 | 无要求 | 仿真里没有运动模糊、自动曝光这些问题，真机效果未知。 |

**下视图 `lookdown_fn()`**：只在需要规划的那一步被调用（约每 1–4 步一次），每步最多一次。

- 返回同一台相机、同一位姿、**相对水平向下 30°** 拍的图，格式同前视图。是相对水平的绝对角度：刚执行过 LOOK_DOWN、相机已经在 −30° 时，就在 −30° 拍。
- 返回前把相机**恢复水平**。
- 慢系统在这张图上找像素目标，快系统也把它当视觉输入，所以不能省，也不能拿前视图冒充。
- 两种做法：云台俯仰（拍前压到 −30°，拍完抬回）；或另装一台固定 −30° 的相机，内参、分辨率、安装高度都要和前视相机一致。仿真里是同一台相机转动，第二台相机的做法**没验证过**。

**水平重拍 `level_fn()`**：只在一种情况下被调用：上一个动作是 LOOK_DOWN，而这一步从队列取到 STOP、当场重新规划（以 ↓ 结尾的短箭头块，如 [5,0,0,0]、[3,5,0,0]）。每步最多一次；同一步里两个都要时，先调 `level_fn` 再调 `lookdown_fn`。

- 返回同一台相机、同一位姿、**水平**拍的图，格式同前视图；返回时相机保持水平。
- 原因：仿真客户端每次重新规划都会再采一次全景，而这一步的第一次采集已经把相机复位成水平，所以这次规划的当前前视图是水平图。`act()` 收到的那张 −30° 图照样发给里程计、进历史。

## 5. 输出与执行约定

| `Action` | 值 | 机器人做什么 | 下一帧怎么拍 |
|---|---|---|---|
| `FORWARD` | 1 | 直行 0.25 m | 相机水平 |
| `TURN_LEFT` | 2 | 原地逆时针转 15° | 相机水平 |
| `TURN_RIGHT` | 3 | 原地顺时针转 15° | 相机水平 |
| `LOOK_DOWN` | 5 | 车体不动，相机压到 −30° | 在 −30° 拍 |
| `STOP` | 0 | 停下，本集结束 | 不再调用 `act()` |

- **每个动作先把相机俯仰设到 `action.camera_pitch_deg`，再动车体**。LOOK_DOWN 只影响它之后的第一次拍摄，也就是下一次 `act()` 收到的那一帧：仿真客户端每次采集全景后都把相机恢复水平（`capture_panoramic_views` 里的 `set_state(..., reset_sensors=True)`），所以同一步里的水平重拍（`level_fn`）、下视图和下一个动作都从水平开始；连续两个 LOOK_DOWN 也还是 −30°，不会叠成 −60°。
- **一个动作做完、停稳后再拍下一帧、再调用 `act()`**。模型没有时间概念，只看这一串图。
- 只有 `act()` 返回 `STOP` 才表示停。动作块里的 STOP 在 NavAgent 内部表示“在这里重新规划”，不会返回给机器人。
- 返回的 `STOP` 不一定表示模型认为到了：`last_call.kind == "fallback_stop"` 表示慢系统输出里既没有箭头也解析不出像素坐标，服务端兜底判停，真机上应记为失败；`kind == "stop"` 才是模型输出了 STOP。
- 到 `max_steps`（默认 500）返回 `STOP`（`last_step.source == "step_cap"`）。仿真评测到上限只是不再走，真机需要一个明确的停。
- 左/右转、LOOK_DOWN 都算一步，也都会产生一帧给里程计。
- 服务端只会返回上表五种动作；返回别的值（例如 4 = 抬头）时 NavAgent 抛异常。
- 动作是开环的：0.25 m / 15° 走不准时模型不知道，只能从下一帧图像里看出来。碰撞在仿真里是贴墙滑动，真机要自己处理。

## 6. 预热：前 20 帧

里程计攒满 20 帧才建第一张图。在此之前 `last_call.pose_ready` 为假，模型走原始 InternNav 路径（不用历史认知），这是设计行为，不是错误。从第 20 帧起（帧号从 0 数，即 `last_call.step` ≥ 19）的规划调用 `pose_ready` 为真，轨迹调用的 `ppa_applied` 也为真。每集都要重新预热。起步阶段只原地转圈时里程计是否稳定，真机上没验证过。

## 7. 重规划节奏

- 一次规划返回最多 4 个动作。第一个立即返回，其余排队，之后每次 `act()` 取一个；队列空了或取到 STOP 时，在当前步重新规划。实际每 1–4 步规划一次。
- 规划是同步的：一次 RPC 里先跑慢系统（最多生成 128 个 token；第一轮输出“↓”时再加一轮，用下视图）。慢系统输出像素坐标时才跑快系统（32 条 × 10 步去噪，取平均轨迹，转成最多 4 个动作，`kind=trajectory`）；输出箭头时直接转成动作（`kind=native_actions`，不跑快系统）；输出 STOP 或解析不出时判停（`kind=stop` / `fallback_stop`）。没有独立运行的快系统循环。
- 取队列动作的 `act()` 只做一次里程计写入 RPC，但不一定快：第 20 帧那次写入会在服务端建图；之后还没进图的帧在下一次规划的位姿查询里补建（攒满 8 帧时在写入时补建），所以规划步的位姿查询也会变慢。以 §8 实测为准。

## 8. 延迟与带宽

**延迟：待计时运行填入。** 下表留空，数字来自 `HEATMAPVLN_TIMING=1` 的实测（服务端与客户端的计时见 `docs/ops/deploy_rtx4090.md` §5；NavAgent 的客户端计时在 `last_step.timing_ms` / `last_call.client_timing_ms`）。**不要用仿真评测的“每步 1–1.3 s”代替**，那包含 CPU 渲染。

| 阶段（每次） | 中位数 (ms) | P90 (ms) | 来源 |
|---|---|---|---|
| 里程计写入（每步） | 待填 | 待填 | `vo_ingest` |
| 里程计位姿查询（每次规划） | 待填 | 待填 | `vo_query` |
| 下视图采集（机器人侧） | 待填 | 待填 | `lookdown_capture` |
| 模型 RPC 往返（每次规划） | 待填 | 待填 | `model_rpc` |
| 其中服务端处理 | 待填 | 待填 | `server_timing_ms.handler_total` |
| 规划步 `act()` 合计 | 待填 | 待填 | `act_total`（`source == plan`） |
| 队列步 `act()` 合计 | 待填 | 待填 | `act_total`（`source == queue`） |

以上是 RTX 4090 上的数；机器人与服务端之间的网络另算。

**带宽**（Habitat 渲染图实测，真实相机的 JPEG 大小会不同）：每步里程计写入约 70 KB（640×480，JPEG 质量 95）；8 帧历史时每次规划请求约 350 KB，其中 9 张 384×384 前视图各约 24 KB、下视图约 46 KB、27 张黑色占位图各约 3 KB（见 §11）。

## 9. 重置、失败与超时

- **每集开始调用 `reset()`**，它会重置里程计会话（地图丢弃）。
- **`act()` 抛出任何异常，本集就结束了**：机器人先停下，再 `reset()` 开新的一集。不能在同一集里重试，因为里程计那一侧已经记下了这一帧，重发会对不上帧号。
- 超时：`rpc_timeout_ms` 默认 60 s，**没有验证过**：最慢的应是服务端启动后的第一次调用和第 20 帧建图，评测一直用 600 s。§8 填数前建议传 `rpc_timeout_ms=600000`。超时或网络错误时 `act()` 抛 `RuntimeError`（vla_rpc 对任何 gRPC 错误都只返回空）。服务端报错的具体原因在服务端日志里（启动脚本打印的目录）。
- 服务端重启后里程计会话丢失，下一次 `act()` 会失败，同样 `reset()`。
- NavAgent 会核对服务端的回显（协议版本、采样种子记录、位姿是否就绪、是否只用了前视图），对不上就抛异常，不会带着错误的输入继续走。

## 10. 没有验证过的

- 任何真实场景：真实图像、光照、运动模糊、动态障碍、动作执行误差、碰撞。
- 与仿真不同的相机：HFOV、分辨率、畸变、安装高度、俯仰；用第二台相机拍下视图。
- 里程计在真实相机上的尺度（部署用固定的 `--translation-scale 1.0`）与起步纯旋转时的稳定性。
- 超过 500 步的长任务（里程计会话按 501 帧开，服务端上限 4096）。
- 非英文或风格差别大的指令。
- 机器人到服务端的网络延迟和抖动。

## 11. 与仿真评测客户端的差异

NavAgent 复刻 `scripts/evaluation/r2r_val_unseen.py` 的评测循环，有意的不同只有这些：

1. 右 / 后 / 左三个视角发黑色占位图。协议仍要求这三张，但部署配置下服务端只用前视图（慢系统提示、历史认知头、快系统都只取前视图和下视图）；NavAgent 每次都核对服务端回显的 `native_front_only`，不为真就报错。
2. 每张前视图只编码一次（仿真客户端每次循环都重渲染、重编码全景）：`act()` 收到的那张一次，LOOK_DOWN 后当场重规划时 `level_fn()` 的水平图一次。同一像素同一编码器，字节相同。
3. 同一步里若规划两次，下视图复用一次拍摄（部署服务端有防死锁，这种情况到不了）。
4. 到步数上限返回 STOP；仿真客户端只是不再走。
5. 更严的校验：只接受 PPA 运行时，未知动作码报错，输入图像尺寸不对报错；RPC 超时默认 60 s（评测用 600 s）。

LOOK_DOWN 后当场重规划不算不同：仿真客户端这时第二次采集全景，相机已被第一次采集复位成水平，NavAgent 用 `level_fn()` 取同一张水平图；里程计帧和历史（同一帧只取第一次采集，即 −30° 那张）两边相同。

## 12. 验证状态

已完成（不占 GPU）：

- `tests/test_nav_agent.py`（本地，假服务端）：把 4090 金丝雀 4 集共 99 次规划调用的动作块喂回去，每次调用的步号、里程计帧号、历史帧号、地图版本号与部署日志逐一相同，总步数相同；另测预热、块内 STOP 重规划、完整轨迹块、LOOK_DOWN、LOOK_DOWN 后块内 STOP 的水平重拍、步数上限、8 动作保护、换集、图像检查、服务端违约、计时开关不改请求。
- `tests/test_nav_agent_habitat.py`（4090 容器，CPU）：
  - 同一组图像下，NavAgent 的规划请求与评测客户端自己的 `_rpc_plan_panoramic` 构造的请求 JSON 逐字相同、每张 JPEG 逐字节相同（13 次调用，其中 2 次是 LOOK_DOWN 后的水平重拍）；里程计写入也逐字节相同。
  - 闭环：在 Habitat 里用脚本化假服务端，评测客户端原样跑一遍、NavAgent 跑一遍。测试默认 1 集（`NAV_AGENT_HABITAT_MAX_EPISODES` 可调），每集 13 次规划、5 次 LOOK_DOWN，含 LOOK_DOWN 后块内 STOP（水平重拍）、LOOK_DOWN 在块尾、块内 STOP、预热到就绪。2 集时 124 个请求除仿真渲染的侧视图外全部逐字节相同（302 张逐字节、618 张侧视图只比布局），逐调用日志与结局相同。
  - 另用一个 LOOK_DOWN 更密的脚本手工跑过 1 集（[3,5,0,0]、[1,1,1,5]、[5,5,1,0]、[5,0,0,0]、[1,5,0,0] 等，不在测试里）：53 个请求全部相同，11/11 次调用相同。
  - 对照（都是手工跑的，不在测试里）：去掉每步的相机复位后同一检查报不同；关掉水平重拍后，LOOK_DOWN 后当场重规划的那几次调用 `current/front` 报不同。

待 GPU 验证（预注册，2026-09-30，跑之前写）：

- **真服务端逐调用比对**：GPU 4 上用 `scripts/deploy/start_nav_servers_cuda.sh` 起服务端，服务端不开计时；NavAgent 只开客户端计时，这不改变请求。
  - 用 `scripts/deploy/nav_agent_habitat_check.py` 跑 09-28 金丝雀的 4 集（分片 0、1 各 2 集，种子 42），分片各用 `--compare-log` 与
    `/workspace/eval_runs/canary_cuda_seed42` 的对应客户端日志比对。
  - **通过**：两个分片都报 `identical`，即每次调用的步号、类型、慢系统输出、动作块都相同，每集结局也相同。这时就可以说
    "NavAgent 接真服务端时，在这 4 集上与部署的评测客户端逐调用相同"。
  - **不通过**：任何一处不同都算，记下第一处不同的调用，先查原因，不改口径。
  - 金丝雀里没有 LOOK_DOWN，所以这一项证明不了 LOOK_DOWN 相关路径；那部分仍只由上面的假服务端闭环覆盖。
- **计时运行**：按 `docs/ops/deploy_rtx4090.md` §5 末尾的预注册跑，通过后再填 §8 的表。

金丝雀里没有 LOOK_DOWN，LOOK_DOWN 相关的路径只由上面的闭环（假服务端）覆盖。

## 13. 对接前要定下来的事

- 机器人底盘和控制方式：离散动作（0.25 m / 15°）够不够，还是要连续路点或速度指令（模型内部有连续轨迹，目前没有返回）。
- 相机：型号、HFOV、分辨率、安装高度；下视图用云台还是第二台相机。
- 算力放在哪：服务端同卡约 36–37 GB 显存（48 GB 版 4090；普通 24 GB 放不下）。机器人上的 NavAgent 只要 CPU、本仓库（只导入几个纯 Python 模块，不需要 torch）、`numpy`、`Pillow` 和 `vla_rpc`。`vla_rpc` 不在 PyPI 上，源码在 4090 容器的 `/workspace/rpc`（`docs/ops/deploy_rtx4090.md` §1），安装时会拉 `grpcio`、`grpcio-tools`、`protobuf`、`opencv-python`。只在 Python 3.11 上测过；`src/deploy/fake_servers.py` 至少要 3.9（ROS Noetic 自带的是 3.8）。
- 网络：机器人到 GPU 服务器的带宽与延迟（§8）。
