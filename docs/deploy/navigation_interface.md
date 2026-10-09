# 导航模型部署接口（NavAgent）

给要把导航模型接到真机上的同事。代码：`src/deploy/nav_agent.py`；服务端启动：`scripts/deploy/start_nav_servers_cuda.sh`。接宇树 Go2 的具体做法见 §14。

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
- 开跑前先看 `nvidia-smi`，只用空卡。两个服务端默认同卡，全量实测峰值合计约 41 GB 显存（§8；这台机器是 48 GB 版 4090，普通 24 GB 的卡放不下）。`NAV_VO_GPU` 可把里程计放到另一张卡；分两张卡时各自占多少没测过。
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

**实测**：RTX 4090，两个服务端同在一张卡上。单位 ms，取中位数 / P90。

- **主表**：A0 种子 42 的全量评测，2026-10-01 → 10-02。R2R val_unseen 1839 集，用两张卡，共 56236 次规划调用、207970 个动作，服务端和客户端都开了计时。
  - 汇总文件：`/workspace/eval_runs/exp20_a0_seed42_4090/runtime/20261001_015310_125250/latency/latency_summary.md`（台账 EXP-20 运行记录 1）。
  - 这次跑的是评测客户端，不是 NavAgent。两者的请求只差服务端不用的三张侧视图：NavAgent 发黑色占位图（§11），其余逐字节相同（§12）。所以服务端和里程计的耗时可以直接用。
  - 表里的图像采集是仿真渲染，真机要另测。
- **NavAgent 自己**：9 月 30 日接真服务端跑过 4 集、99 次调用。带注入的规划步 `act()` 中位 5067 ms，和主表一致（§12）。

**不要用仿真评测的"每步 1–1.3 s"代替这里的数**，那个数包含 CPU 渲染。

带注入的轨迹调用（24637 次）：

| 阶段（每次） | 中位数 | P90 | 汇总里的名字 |
|---|---:|---:|---|
| 里程计写入（每步，往返，客户端计） | 16 | 20 | `step.vo_ingest` |
| 里程计位姿查询（每次规划，含补建图） | 1359 | 1965 | `plan.vo_query` |
| 模型 RPC 往返（每次规划） | 3602 | 3775 | `plan.model_rpc` |
| 其中服务端处理 | 3594 | 3768 | `model_server` |
| 规划的关键路径（位姿查询 + 下视图 + 编码 + 模型 RPC） | 5258 | 5965 | `plan_latency` |

关键路径里的下视图采集是仿真渲染，中位 92 ms。服务端内部各阶段的拆分见 `docs/ops/deploy_rtx4090.md` §5：慢系统两轮约 1.9 s，历史头 0.6 s，快系统 1.05 s，桥约 1 ms。

**按调用类型分开看**（中位数）：

| 类型 | 次数 | 规划关键路径 | 其中服务端处理 |
|---|---:|---:|---:|
| 带注入的轨迹调用 | 24637 | 5258 | 3594 |
| 预热期轨迹调用（前 20 帧，不用历史） | 5443 | 3231 | 2965 |
| 只有慢系统（箭头或停止，不跑快系统） | 26156 | 2545 | 863 |

**对真机意味着什么**：
- 一次规划平均执行 3.7 个动作（207970 / 56236）。
- 规划是同步的：机器人每 1–4 步要原地等 2.5–5.3 s（看调用类型），拿到动作后才能继续。
- 只算计算、不算仿真渲染，平均每个动作约 1.0 s（模型约 0.61 s、里程计约 0.37 s，按动作摊）。还要加上机器人执行动作、停稳的时间（Go2 见 §14）。
- 网络延迟另算：每次规划要多走两个往返（位姿查询、模型调用），每走一步要多走一个（里程计写入）。
- 显存：全量里模型服务端峰值预留 22.8 GB（最高 24.2 GB），里程计 17.2 GB，两者同卡合计最高约 41 GB。

**带宽**（Habitat 渲染图实测，真实相机的 JPEG 大小会不同）：每步里程计写入约 70 KB（640×480，JPEG 质量 95）；8 帧历史时每次规划请求约 350 KB，其中 9 张 384×384 前视图各约 24 KB、下视图约 46 KB、27 张黑色占位图各约 3 KB（见 §11）。

## 9. 重置、失败与超时

- **每集开始调用 `reset()`**，它会重置里程计会话（地图丢弃）。
- **`act()` 抛出任何异常，本集就结束了**：机器人先停下，再 `reset()` 开新的一集。不能在同一集里重试，因为里程计那一侧已经记下了这一帧，重发会对不上帧号。
- 超时：`rpc_timeout_ms` 默认 60 s，评测一直用 600 s。
  - 按 §8 的全量数字，模型调用的 P90 是 3.8 s，位姿查询是 2.0 s，60 s 有十几倍余量。
  - 服务端刚启动后的第一次调用没有单独测过。第一次连上时可以先传 `rpc_timeout_ms=600000`。超时或网络错误时 `act()` 抛 `RuntimeError`（vla_rpc 对任何 gRPC 错误都只返回空）。服务端报错的具体原因在服务端日志里（启动脚本打印的目录）。
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

GPU 验证（预注册，2026-09-30，跑之前写；结果在本小节末尾）：

- **真服务端逐调用比对**：GPU 4 上用 `scripts/deploy/start_nav_servers_cuda.sh` 起服务端，服务端不开计时；NavAgent 只开客户端计时，这不改变请求。
  - 用 `scripts/deploy/nav_agent_habitat_check.py` 跑 09-28 金丝雀的 4 集（分片 0、1 各 2 集，种子 42），分片各用 `--compare-log` 与
    `/workspace/eval_runs/canary_cuda_seed42` 的对应客户端日志比对。
  - **通过**：两个分片都报 `identical`，即每次调用的步号、类型、慢系统输出、动作块都相同，每集结局也相同。这时就可以说
    "NavAgent 接真服务端时，在这 4 集上与部署的评测客户端逐调用相同"。
  - **不通过**：任何一处不同都算，记下第一处不同的调用，先查原因，不改口径。
  - 金丝雀里没有 LOOK_DOWN，所以这一项证明不了 LOOK_DOWN 相关路径；那部分仍只由上面的假服务端闭环覆盖。
- **计时运行**：按 `docs/ops/deploy_rtx4090.md` §5 末尾的预注册跑，通过后再填 §8 的表。

**结果（2026-09-30，GPU 4，源码副本 `94146fa`）：两项都通过。**
- **真服务端逐调用比对**：两个分片都报 `identical`，4 集依次为 14/14、30/30、28/28、27/27 次调用相同，结局相同。
  每集的 SPL、NE 与金丝雀逐位相同，例如 0.9951026954713119、0.26651203632354736。产物在
  `6024_fjl:/home/fangjialei/verify_0930/navagent/shard_0{0,1}/`，每个目录里有 `client.log`、`calls.jsonl`、`progress.json`、`compare_log.json`。
- **计时运行**：通过，见 `docs/ops/deploy_rtx4090.md` §5 末尾。§8 的表已经按这次的数填好。

金丝雀里没有 LOOK_DOWN，LOOK_DOWN 相关的路径只由上面的闭环（假服务端）覆盖。

## 13. 对接前要定下来的事

**已定（2026-09-30）**：
- 机器人是宇树 Go2，装两台相机：一台前视（水平），一台固定下倾 30°。
- 模型和里程计放在 4090 服务器上，Go2 这一侧只跑 NavAgent，通过网络连过去。
- 交付是这份文档；Go2 一侧的程序由使用方按 §14 自己写。

**仍待定，也是风险最大的地方**：
- **相机型号、视场、分辨率、安装高度**：仿真里相机离地 1.25 m、水平，HFOV 79°。Go2 自带相机估计只有约 0.37 m 高，视场也对不上（§14.3）。这种视点从来没测过。
- **Go2 的型号和固件版本**：必须是 EDU；固件是否 ≥ V1.1.6，决定用哪一版运动接口（§14.1）。
- **机器人到 4090 的上行网络**：Go2 的开发接口只支持有线，连 4090 要另配无线或路由（§14.2）。

## 14. 接到宇树 Go2

**本节的状态**：
- 依据是宇树官方文档、`unitree_sdk2_python`（2026-09-21 的 `814556d`），以及公开的 Go2 导航部署代码，出处列在 §14.8。
- **没有在真机上跑过**。标"待实测"的数和代码里的参数，都要在 §14.6 的步骤里实测后再定。

### 14.1 前提

- **型号**：只有 Go2 **EDU** 开放二次开发接口，AIR、PRO 不开放。
- **固件**：在 Unitree App 里查软件版本。
  - ≥ V1.1.6：用 2025-05 发布的 V2.0 运动接口。`Move` 的范围是 vx −2.5~3.8 m/s、vy ±1.0 m/s、vyaw ±4 rad/s。
  - 更早的版本：用旧版接口，范围是 vx ±0.6 m/s、vy ±0.4 m/s、vyaw ±0.8 rad/s。
  - 下面都按 ≥ V1.1.6 写。
- **控制方式**：只用高层运动服务（`SportClient`）。
  - 不要碰底层电机控制：用它得先关掉主运控服务，否则两套控制同时下指令会失控。
  - 主运控服务要一直开着：高层指令要经它转发。

### 14.2 程序跑在哪、怎么连

```
4090 宿主机（容器里两个服务端）
        ▲  ssh -L 隧道（§2），走机器人上另配的无线网
机器人侧电脑（NavAgent + Go2 驱动，同一个 Python 进程）
        │  网线，192.168.123.x 网段
Go2 本体（192.168.123.161）    前视相机、下视相机（USB）
```

- **机器人侧电脑**有两个选择：
  1. Go2 的扩展坞（Jetson Orin，`192.168.123.18`，出厂 JetPack 5.1.1）。
  2. 背在狗上的一台 x86 笔记本或 NUC，装 Ubuntu 20.04 / 22.04，用网线接 Go2，IP 设在 `192.168.123.x`（例如 `.222`，不能用 `.161`）。
  - 宇树不支持在 Go2 内置计算机（`.161`）上跑用户程序，也不支持 Mac、Windows。
  - **建议先用 x86 那台**：`cyclonedds==0.10.2`（宇树 SDK 锁定的版本）只有 x86_64、Python 3.7–3.10 的现成包；扩展坞是 aarch64，要先从源码编译 C 版 cyclonedds 0.10.x，再设 `CYCLONEDDS_HOME`。
- **Python 环境**：一个 Python 3.10 环境，同时装 NavAgent 的依赖和 `unitree_sdk2_python`（在它的源码目录里 `pip3 install -e .`）。
  - **NavAgent 的依赖**：只要 CPU，不需要 torch。装本仓库（只导入几个纯 Python 模块）、`numpy`、`Pillow` 和 `vla_rpc`。
  - `vla_rpc` 不在 PyPI 上，源码在 4090 容器的 `/workspace/rpc`（`docs/ops/deploy_rtx4090.md` §1），安装时会拉 `grpcio`、`grpcio-tools`、`protobuf`、`opencv-python`。
  - `src/deploy/fake_servers.py` 至少要 Python 3.9。
  - NavAgent 只在 3.11 上测过。在 3.10 上先跑一遍 `python -m pytest tests/test_nav_agent.py`，这一步不需要 GPU，也不需要机器人。
- **连 4090**：宇树写明 Go2 的开发接口**只支持有线**，所以机器人侧电脑要另有一条到 4090 的网络（自带无线网卡，或在狗上加装路由器）。之后按 §2 开 ssh 隧道，NavAgent 连本机端口。
  - 每次规划上传约 350 KB，每步约 70 KB（§8）。
  - 网络往返延迟每次规划要多付两次（§8），上线前用 `ping` 量一下。
- **DDS 网卡**：`ChannelFactoryInitialize(0, "<网卡名>")`，网卡名是接 Go2 的那块网卡（用 `ifconfig` 查 `192.168.123.x` 那一块）。不传就自动选，多网卡的机器上建议显式传。

### 14.3 两台相机怎么对应 `act()` 的三个输入

两台相机都是固定的，没有云台，所以 §3、§5 里"把相机压到 −30°"变成"下一帧改从下视相机取"：

| 接口 | 读哪台相机 |
|---|---|
| `act()` 的 `front_rgb` | 上一个动作的 `camera_pitch_deg` 是 0 时读前视相机，是 −30 时（上一个动作是 LOOK_DOWN）读下视相机 |
| `lookdown_fn()` | 下视相机。不用"恢复水平"，因为两台都不动 |
| `level_fn()` | 前视相机 |

**图像格式**按 §4 处理：
- 先去畸变，再居中裁到水平 79°、4:3，缩放到 640×480 的 RGB `uint8`。
- 两台相机的内参、分辨率、处理流程要一致。

**为什么建议外接相机，不用 Go2 自带的**：
- **视场对不上**：官方多媒体文档写自带相机 1280×720、水平 100°、垂直 56°；FAQ 和产品页却写"120°"，没说是哪个方向，内参也没公开。按垂直 56° 算，裁成 4:3 后水平只剩约 71°，小于模型要的 79°，补不回来。
- **太低**：官方 URDF 里相机在机身前方、水平朝前。按默认站高 0.33 m 推算，离地约 0.37 m（待实测）。仿真是 1.25 m。
- **公开部署的做法**：
  - StreamVLN、InternNav、Uni-NaVid、VLingNav 等在 Go2 上跑导航的工作，用的都是外接相机（RealSense D455 / D457 一类）。
  - 给出安装参数的几家（InternNav、DyNaVLM、SparseVideoNav）离地约 0.7–1 m，下倾 10–15°。
  - 没有找到装在 1.25 m、保持水平，或下倾 30° 的部署。
- **InternNav 作者的经验**（他们的 issue 回复）：
  - 相机高度、俯角和运动模糊对效果影响很大；
  - 他们早期也是一台水平、一台下倾的两台相机，下倾那台专门对应 LOOK_DOWN（和我们这套一样），后来改成只用下倾那台。
- **建议**：
  - 用支架把两台相机装到接近 1.25 m，前视水平，下视下倾 30°，尽量上下对齐。
  - 固定牢，避免高频振动。
  - 选全局快门或运动模糊小的相机，动作做完、停稳后再拍。
  - 装不到 1.25 m 时，记下实际高度和俯角，结果要按"视点与训练不同"来读。
- **相机装在机身上方时**，调用 `AutoRecoverySet(False)`：官方建议带负载时关掉跌倒自动翻身，免得翻身时压坏头部的相机和支架。

### 14.4 离散动作怎么变成 Go2 运动

**几条接口事实**（V2.0，`unitree_sdk2_python`）：
- **`Move(vx, vy, vyaw)`**：机体坐标系速度，发出后不等应答。返回 0 只表示消息发出去了，不代表机器人执行了。
- **指令保持**：最新一条 `Move` 维持 1 s，运控不对它做滤波。
  - 所以一个动作执行期间要**按固定频率重发**。官方没给频率；公开实现用 10–25 Hz，宇树的 C++ 示例用 200 Hz。
  - 不动时发 `Move(0, 0, 0)` 或 `StopMove()`。
- **启动顺序**：
  - `StandUp()` 是关节锁定的站立；有用户报告，只调 `StandUp()` 后 `Move` 几秒到十几秒都不动（issue #175）。
  - 先调 `BalanceStand()` 进入平衡站立。还要等 `rt/sportmodestate` 的 `error_code` 变成可行走状态（如 1013 平衡站立）再发 `Move`，这一条是推断，待实测。
- **位姿**：订阅 `rt/sportmodestate`（`SportModeState_`），平面位姿取 `position[0]`、`position[1]` 和 `imu_state.rpy[2]`（yaw）。
  - 这是 Go2 的腿式融合里程计。有第三方测量说它的转角准、平移尺度不准；在 0.25 m / 15° 这么小的步长上误差多大，没有数据，待实测。

**映射**（每个动作在**当前实测位姿**上起算）：

| `Action` | Go2 做什么 | 到位条件（待实测后调） |
|---|---|---|
| `FORWARD` | 沿当前朝向前进 0.25 m：按与起点的距离做 P 控制，同时用 vyaw 修正航向 | 距离误差 < 2 cm，或超时 |
| `TURN_LEFT` / `TURN_RIGHT` | 原地转 ±15°（±0.2618 rad），按 yaw 误差做 P 控制 | 角度误差 < 1.5°，或超时 |
| `LOOK_DOWN` | 不动，下一帧从下视相机取（§14.3） | — |
| `STOP` | `StopMove()`，本集结束 | — |

- 到位后发 `StopMove()`，等机身停稳再拍下一帧。沉降时间待实测，公开实现用 0.2 s。
- 动作做完到拍下一帧之间不能有残余运动：模型没有时间概念，每一帧都要是"这个动作做完后"的视点（§5）。
- 速度先保守：前进 ≤ 0.3 m/s，转向 ≤ 0.5 rad/s。有用户反映，速度压到 0.3 m/s 左右时里程计漂移明显变小。
- **起算点**：公开实现有两种做法。
  - StreamVLN 在上一个目标位姿上累加，残差不积累，但里程计漂移会带进来。
  - InternNav 从当前里程计起算，没走到位的部分直接丢掉。
  - 仿真里每个动作都相对当前状态，所以这里按"当前实测位姿"写，并把每步实际走了多少记进日志。

### 14.5 示意代码

下面是**没有在真机上跑过的示意**。`FrontCam` / `DownCam` 指你们自己的相机读取函数，每次返回一张原始 RGB 图。

```python
import math
import threading
import time

import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from src.deploy import Action, NavAgent


def to_model_frame(rgb: np.ndarray, hfov_deg: float) -> np.ndarray:
    """去畸变后的 RGB → 居中裁到水平 79°、4:3 → 640x480（§4）。垂直视场不够时报错。"""
    h, w = rgb.shape[:2]
    cw = w * math.tan(math.radians(39.5)) / math.tan(math.radians(hfov_deg / 2))
    ch = cw * 3 / 4
    if cw > w or ch > h:
        raise ValueError("camera field of view smaller than 79 x 63.5 deg")
    x0, y0 = int(round((w - cw) / 2)), int(round((h - ch) / 2))
    crop = rgb[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
    return np.ascontiguousarray(cv2.resize(crop, (640, 480), interpolation=cv2.INTER_AREA))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Go2Base:
    """离散动作 → Go2 高层速度指令，20 Hz 闭环在 rt/sportmodestate 上。"""

    def __init__(self, nic: str, rate_hz: float = 20.0, settle_s: float = 0.3):
        ChannelFactoryInitialize(0, nic)
        self.sport = SportClient()
        self.sport.SetTimeout(10.0)
        self.sport.Init()
        self.dt, self.settle_s = 1.0 / rate_hz, settle_s
        self._msg, self._t = None, 0.0
        self._lock = threading.Lock()
        self._sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._sub.Init(self._on_state, 10)
        self.sport.BalanceStand()            # 不要只用 StandUp（§14.4）

    def _on_state(self, msg):
        with self._lock:
            self._msg, self._t = msg, time.monotonic()

    def pose(self):
        with self._lock:
            msg, t = self._msg, self._t
        if msg is None or time.monotonic() - t > 0.5:   # 状态 0.5 s 没更新：当故障处理
            raise RuntimeError("rt/sportmodestate is stale")
        return msg.position[0], msg.position[1], msg.imu_state.rpy[2]

    def stop(self):
        self.sport.StopMove()
        time.sleep(self.settle_s)            # 等停稳再拍下一帧

    def forward(self, dist=0.25, v_max=0.3, tol=0.02, timeout=6.0):
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
            vx = max(-v_max, min(v_max, 1.5 * err))
            vyaw = max(-0.3, min(0.3, 2.0 * wrap(th0 - th)))
            self.sport.Move(vx, 0.0, vyaw)   # 指令只保持 1 s，所以每个周期重发
            time.sleep(self.dt)
        self.stop()

    def turn(self, deg, w_max=0.5, tol=math.radians(1.5), timeout=6.0):
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
            self.sport.Move(0.0, 0.0, max(-w_max, min(w_max, 2.0 * err)))
            time.sleep(self.dt)
        self.stop()


def run_episode(instruction: str, run_id: int, base: Go2Base, front_cam, down_cam, hfov_deg: float):
    front = lambda: to_model_frame(front_cam(), hfov_deg)
    down = lambda: to_model_frame(down_cam(), hfov_deg)
    with NavAgent("127.0.0.1:52400", "127.0.0.1:52500", rpc_timeout_ms=600000, on_log=print) as agent:
        agent.reset(instruction, scene_id="go2lab", episode_id=run_id)   # 每次实跑换一个 run_id（§3）
        pitch = 0.0
        try:
            while not agent.done:
                frame = down() if pitch < 0 else front()                   # §14.3：上一个是 LOOK_DOWN 就读下视相机
                action = agent.act(frame, down, front)
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

### 14.6 安全

- **遥控器**：始终有人拿着 Go2 的遥控器跟在旁边。
  - 不要调 `SwitchJoystick(False)`，否则推摇杆会失效。
  - 公开实现的做法是摇杆一动就让出控制。
- **软急停**：程序里的急停用 `StopMove()`。
  - `Damp()` 让所有关节进入阻尼，优先级最高，但机身会在重力下趴下，只在意外时用。
- **看门狗**：
  - 示意代码里，状态 0.5 s 不更新就报错停车；单个动作超时 6 s。
  - 有用户报告，长时间运行中偶发运控卡死：里程计时间戳停住、遥控器也失灵，只能断电，约 1/80 次（issue #184）。所以看门狗要有，也要能断电。
- **碰撞**：模型没有任何避障，仿真里撞墙只是贴墙滑动（§5）。
  - 前几次在空旷房间、低速、短指令下跑。
  - 宇树另有避障服务（`ObstaclesAvoidClient`），但它可能改写或拒绝运动，和离散动作的语义不一致，先不要开。
- **步数上限**：`max_steps` 默认 500，到了返回 `STOP`（§5）。

### 14.7 第一次上机的顺序

1. **装环境**：在机器人侧电脑上装好环境，跑 `tests/test_nav_agent.py`（不需要 GPU，不需要机器人）。
2. **只接相机**：用假服务端（§3 末尾）跑通循环，机器人不动。
   - 存几帧经过 `to_model_frame` 的图，检查是不是 RGB、方向对不对、水平视场是不是 79°。
   - 检查前视、下视两台相机的画面是否上下对齐。
3. **运动标定**，不接模型：在地上贴 0.25 m 刻度和 15° 角度线，`forward` 和 `turn` 各做 20 次。
   - 记录实测位移和转角的误差、每个动作的耗时、停稳所需时间。
   - 据此调速度、容差、沉降时间。
4. **接真服务端，机器人不动**：动作只打印不执行，量规划延迟和网络往返，和 §8 比。
5. **接真服务端，开始动**：空旷房间、短指令、低速，有人拿遥控器跟着。
6. **每一步都留记录**，方便事后对照仿真：
   - 发给模型的图像、动作；
   - `last_call`（`kind`、慢系统输出、`pose_ready`）；
   - Go2 的里程计位姿、各段耗时。

### 14.8 出处与待实测

**出处**：
- 宇树开发者文档：
  - 运动接口 V2.0：<https://doc-cdn.unitree.com/6/814/zh/6_814_zh>
  - 快速开始、网络：<https://doc-cdn.unitree.com/6/45/en/6_45_en>
  - FAQ：<https://doc-cdn.unitree.com/6/60/en/6_60_en>
  - 多媒体（相机）：<https://doc-cdn.unitree.com/6/53/zh/6_53_zh>
- `unitree_sdk2_python`：<https://github.com/unitreerobotics/unitree_sdk2_python>（`814556d`）
  - `go2/sport/sport_client.py`、`go2/video/video_client.py`、`core/channel.py`
  - issue #175（StandUp 后 Move 不动）、#184（运控卡死）
- Go2 URDF（相机位姿）：<https://github.com/unitreerobotics/unitree_ros>，`robots/go2_description/urdf/go2_description.urdf`
- 公开的 Go2 导航部署：
  - StreamVLN `realworld/go2_vln_client.py`
  - InternNav `scripts/realworld/`，以及 issue #295、#324、#157
  - Uni-NaVid（arXiv 2412.06224 附录 XII）
- Go2 里程计精度的第三方测量：arXiv 2506.09548 表 II

**待实测**（文档里查不到）：
- 固件版本；
- 相机实际视场、内参和安装高度；
- `Move` 合适的重发频率和最小有效速度；
- 动作后的沉降时间；
- `rt/sportmodestate` 的实际频率；
- 0.25 m / 15° 上的里程计误差；
- 机器人到 4090 的网络往返；
- NavAgent 在 Python 3.10 上能否通过测试。
