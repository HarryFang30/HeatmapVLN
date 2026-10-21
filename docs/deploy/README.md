# 实机部署：从这里开始

把导航模型接到宇树 Go2 上：模型跑在 4090 服务器上；狗上的 Orin 跑实验室的 ROS 2 底座 `robot_platform_ws`，
我们的进程也在这台 Orin 上，从底座取图、调用 4090 上的模型、再把返回的离散动作交给底座执行。

> **到 2026-10-21 为止，我们的模型没有在真机上跑过。**
> 下面的数字来自仿真（Habitat R2R-CE）、4090 上的服务端实测和底座自己的验收记录。
> 接底座的代码只在模拟的底座客户端上跑通过。相机方案：一台 D455 装在支架上，约 0.7–1 m、固定下倾 15°，
> 前视图和下视图都用它的同一帧（§6.2，照 InternNav 在 Go2 上的实机做法），还没装。

完整说明在 [`navigation_interface.md`](navigation_interface.md)（下文的 §n 都指它）。

## 1. 全链路

```
4090 宿主机 · docker 容器 fjl-habitat · 一张 48 GB 卡
    模型服务端    :52400   指令 + 图像 → 动作块（最多 4 个动作）
    里程计服务端  :52500   RGB 序列 → 各帧相对位姿
        ▲
        │  gRPC（JSON + JPEG，无加密、无认证），从 Orin 走 ssh -L 隧道（§4）
        │
狗上的 Orin（Ubuntu 20.04 · ROS 2 Foxy · Python 3.8）
    我们的进程：NavAgent + 底座的 ROS 客户端 ROSPlatformClient
        │  ROS 2，同一个 domain
    底座 robot_platform_ws：相机话题、相对动作、控制权
        │                                   │
        ▼                                   ▼
Go2 EDU：运动（自带前视只做记录）       支架上的 D455（约 0.7–1 m、下倾 15°，底座里叫 nav_camera，§6.2）
```

## 2. 输入和输出

模型这一侧只和 `NavAgent` 打交道，狗这一侧只和底座的 `ROSPlatformClient` 打交道：

```python
from robot_platform.ros_client import ROSPlatformClient
from src.deploy import Action, NavAgent

# 示意；完整写法（就绪等待、try/finally 释放控制权、连取两帧）见 §6.5
with ROSPlatformClient() as robot, NavAgent("127.0.0.1:52400", "127.0.0.1:52500") as agent:
    robot.wait_for_motion_ready(timeout=40)
    robot.acquire(owner="heatmapvln", mode="relative")
    agent.reset(instruction, scene_id="go2lab", episode_id=run_id)      # 每集一次
    while not agent.done:
        frame = ...                                                     # nav_camera 这一步的图（§6.4）
        action = agent.act(frame, lambda: frame, lambda: frame)         # 三个输入都是同一帧
        ...                                                             # 交给 robot.move_relative；state 不是 succeeded 就作废本集
```

| | 什么时候 | 内容 | 在底座上从哪来 / 到哪去 |
|---|---|---|---|
| **输入** | 每集一次 | 英文指令；`scene_id`（字符串）、`episode_id`（非负整数），两者决定模型采样用的随机种子，每次实跑换一个 `episode_id` | — |
| **输入** | 每步一次 | `front_rgb`：上一个动作做完、停稳后在当前位姿拍的图，`numpy` 数组，`uint8`，形状 `(480, 640, 3)`，RGB | `robot.get_frame('nav_camera', wait_new=True)` 的 640×480 JPEG（D455 彩色图，水平约 79°），解码转 RGB（§6.4）；动作后连取两帧、用第二帧 |
| **输入** | 只在要规划的那一步 | `lookdown_fn()`：仿真里是下倾 30° 的图 | 返回这一步的同一帧（相机本身已经下倾 15°，InternNav 实机就是这样做的，§6.2） |
| **输入** | 只在一种情况 | `level_fn()`：上一个动作是 `LOOK_DOWN`、且这一步取到动作块里的 STOP（表示"在这里重新规划"）时调；仿真里是水平的图 | 同一帧 |
| **输出** | 每步一次 | 一个 `Action` | `FORWARD` → `move_relative(distance=0.25)`；`TURN_LEFT` / `TURN_RIGHT` → `move_relative(angle=±15°，rad)`；`LOOK_DOWN` 什么都不做（相机是固定的）；`STOP` 结束、`release()` |
| **异常** | 任何一步 | `act()` 抛任何异常，或底座取图超时、`move_relative` 返回的 `state` 不是 `succeeded`、控制权丢失，本集都作废，不能在同一集里重试；`agent.done` 不会因异常变真 | `release()`，底座清零输出；新的一集重新 `reset()` |
| **不需要** | — | 深度、里程计、定位、地图 | 相对位姿由 4090 上的里程计从 RGB 序列估计，不用底座的 `/platform/odom`；D455 的深度也不用 |

图像要求、每个回调什么时候被调、一集里的调用顺序和诊断字段在 §2；相机方案、底座接口和完整示意代码在 §6；异常和超时在 §9。

## 3. 上机顺序

| 步 | 做什么 | 在哪 | 见 |
|---|---|---|---|
| 1 | 用 `/workspace/nav_src_ded4bd3` 这份代码启动两个服务端，看到 `READY ... attn=flash_attention_2` | 4090 | §3 |
| 2 | 先在 `~/.ssh/config` 加好 `6024_fjl`；在 `./platform shell` 里建 venv、装 vla_rpc，用 `14ac629` 或之后的检出跑 `python -m pytest tests/test_nav_agent.py --noconftest -q`，应为 23 passed | Orin | §4、§5 |
| 3 | 开 ssh 隧道，构造一次 `NavAgent` 做连通自检 | Orin | §4 |
| 4 | 打印 InternNav 的 Go2 支架，自己画一个 15° 楔块（官方楔块文件只有拓竹的 G-code），装 D455；换下 `camera2`，底座配置里关掉 `camera2` 和 `side_panorama`、加 `nav_camera`，重启底座 | Orin + Go2 | §6.2 |
| 5 | 停掉实机底座，`./platform run --backend sim --allow-motion`，用假服务端跑通取图、动作、释放控制权（模拟图是 256×192 占位图，只测接线） | Orin | §6.5、§7 |
| 6 | 实机只读（`./platform run`），`./platform cameras --cameras nav_camera` 确认出图，检查图像、俯角，量镜头离地高度 | Orin + Go2 | §6.2、§7 |
| 7 | 实机只读 + 真服务端，动作只打印（去掉 `wait_for_motion_ready`、拿控制权和动作调用）；再 `--allow-motion` 单独试几次 0.25 m / ±15°（装了支架，重心变了） | 全部 | §6.3、§7 |
| 8 | 空旷房间、短指令实跑，有人拿遥控器跟着 | 全部 | §6.6、§7 |

## 4. 要准备什么

| 项 | 要求 | 见 |
|---|---|---|
| 机器人 | 宇树 Go2 **EDU**，装好底座 `robot_platform_ws` 的 Orin | §5、§6 |
| 相机 | 一台 Intel RealSense D455（全局快门），只用彩色图 640×480 | §6.2 |
| 支架 | InternNav 仓库的 `assets/3d_printing_files/go2_stand.STEP`（相机约 0.7 m）加一个自己画的 15° 楔块（官方的 `downwards_slope.3mf` 只有拓竹打印机的 G-code，见 §6.2）；装牢防振。要到约 1 m 得另做更高的杆，等仿真对比再定 | §6.2 |
| 底座配置 | D455 接在换下的 `camera2` 的 USB 口；关掉 `camera2` 和 `side_panorama`，加 `nav_camera`。自带前视和左侧环视只做记录 | §6.2 |
| Orin 环境 | ROS 2 Foxy 的系统 Python 3.8，加 numpy、Pillow、vla_rpc（从 4090 拷） | §5 |
| 服务器 | 4090 上一张没有别人进程的 48 GB 卡（两个服务端同卡约 40 GiB；24 GB 的卡放不下一套） | §3、§8 |
| 网络 | Orin 能 ssh 到 4090 宿主机（地址、账号、跳板向负责人要） | §4 |

## 5. 时间预算

启动脚本默认 flash-attention 2。两个服务端同卡，4090 实测：

- 平均每 3.7 个动作规划一次（SDPA 全量 1839 集）。
- 规划时狗原地等（中位数）：用上历史帧和位姿的轨迹调用 **3.3 s**，预热期（前 20 帧）1.8 s，只跑慢系统的调用 1.2 s
  （flash-attention 2，`NavAgent` 跑 4 集仿真）。SDPA 全量是 5.3 / 3.2 / 2.5 s。这些数含仿真里的取图（0.1–0.2 s）；
  单台相机时下视不用另取图。
- 底座执行一个动作约 **2.2–2.4 s**（底座 2026-10-08 实测，没装支架，含停稳判断），动作后连取两帧新图约 0.4 s（按 5 fps 推算，没实测）。
- 另加网络往返：每步一次，每次规划再加两次。粗算一集 100 个动作大约 5–6 分钟。

细分与带宽、显存见 §8。

## 6. 验证到哪一步了

**验证过的**：

- `NavAgent` 发出的请求与评测客户端（仿真里跑成绩用的 Habitat 客户端）逐字节相同（`tests/test_nav_agent_habitat.py`）。
- 接 4090 上的真服务端跑 4 集仿真，与部署的评测客户端逐调用相同，每集的结局和评测成绩都完全相同：SDPA 下 99/99 次调用（09-30），
  默认的 flash-attention 2 下 82/82 次调用（10-15）。
- `tests/test_nav_agent.py` 在 Python 3.8 上 23 个全部通过（macOS arm64，附录 C）。
- §6.5 的接底座代码在一个按底座客户端签名写的模拟客户端上跑通一集：只读 `nav_camera`、下视和水平重拍都是同一帧、动作映射、
  控制权释放都对（附录 C）。
- 底座自己（不是我们）：取图、相对动作 0.25 m / ±15° 的实测，PanaV 在它上面走完过一条完整路线。
- InternNav 自己（不是我们）：论文里 Turtlebot4、Go2、G1 都用 D455、下倾 15°，高度各不相同（论文 §5.2）；Go2 上官方支架把相机抬到约 0.7 m（#194）。
  下视用同一帧是他们实机 demo 的做法；早期 Go2 上装过一台水平、一台下倾两台相机，维护者说改成一台对成功率没有明显影响，但没给数字（#295）。

**没验证过的**（附录 A），风险从大到小：

1. **相机视点和仿真不同**：仿真是 1.25 m 水平前视加 30° 下视；实机是约 0.7–1 m、固定下倾 15°、下视用同一帧。
   冻结的慢系统和快系统在训练里见过 0.6 m、下倾 15° 的视角（0.7–1 m 没有），我们自己训练的历史认知和桥没见过。仿真里不同高度、俯角的对比在另一个会话里做。
2. 支架和 D455 还没装；D455 的彩色节点经底座采集没试过；装支架后底座的相对动作要重新核对。
3. 真实图像、光照、运动模糊；动作执行误差；碰撞（模型完全不避障）。
4. 里程计在真实相机上的尺度，以及起步原地转圈时是否稳定。
5. Orin 到 4090 的网络延迟和抖动；从容器外经 ssh 隧道连服务端这条路没有端到端测过（4090 上的核验都在容器里连）；Orin 上的 Python 3.8 环境还没装过。
6. LOOK_DOWN 相关的路径只在假服务端闭环里测过（09-28 那 4 集里没有出现 LOOK_DOWN）。

## 7. 其他文档

| 文件 | 讲什么 |
|---|---|
| [`navigation_interface.md`](navigation_interface.md) | 完整说明：输入输出、4090 侧、网络、Orin 环境、相机方案和接底座、延迟、验证记录 |
| 底座 `robot_platform_ws` 的 `README.md`、`docs/project_integration.md`（实验室工作区，不在本仓库） | 底座的相机、控制权、相对动作和急停 |
| InternNav 仓库 <https://github.com/InternRobotics/InternNav> 的 `assets/3d_printing_files/`、`scripts/realworld/` | Go2 支架、15° 楔块、实机服务端（相机方案的依据，附录 D） |
| [`../ops/deploy_rtx4090.md`](../ops/deploy_rtx4090.md) | 4090 部署机的布局、评测启动脚本与坑 |
| [`../experiments/exp20-latency-report.md`](../experiments/exp20-latency-report.md) | 延迟实测的完整分解 |
| [`../experiments/exp20-ablation-report.md`](../experiments/exp20-ablation-report.md) | 完整模型在 R2R-CE val_unseen 上的成绩（SR 62.70，1839 集） |
| [`../ops/deploy_ascend_910b.md`](../ops/deploy_ascend_910b.md) | 昇腾 910B 的移植（那台机器的延迟数还不能报） |
