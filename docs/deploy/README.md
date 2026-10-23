# Go2 导航模型部署说明

模型服务在 4090 服务器上。Go2 的 Orin 上跑底座 `robot_platform_ws` 和本仓库的脚本 `scripts/deploy/run_go2.py`。

```
4090 服务器（docker 容器 fjl-habitat）
    模型服务 :52400    里程计服务 :52500
          ▲  gRPC，经 ssh 隧道
          │
Go2 上的 Orin
    run_go2.py ── ROS 2 ── 底座 robot_platform_ws ── Go2（front_camera、相对动作）
```

## 1. 接口

### 1.1 输入

| 什么 | 格式 | 从哪来 |
|---|---|---|
| 导航指令 | 一句英文，例如 `"Walk past the sofa, turn left, and stop at the kitchen door."` | 命令行 `-i` |
| 图像 | 每一步一帧：底座 `front_camera` 的 JPEG，原样传给脚本 | `robot.get_frame("front_camera", wait_new=True)` |

- 每一帧都要在上一个动作做完、狗停稳以后再取。
- 脚本把这一帧居中裁剪、缩放成 640×480 的 RGB，再发给模型。
- 不需要底座的深度、里程计（`/platform/odom`）、定位和地图。

### 1.2 输出：动作格式

模型每一步输出**一个离散动作**。不是航点（waypoint），不是速度，也不是一串 dx / dy 轨迹。每个动作就是机体坐标系 `base_link` 下的一次固定的相对运动：

| 动作 | 代码 | dx | dy | dyaw | 底座调用 |
|---|---|---|---|---|---|
| `FORWARD` | 1 | +0.25 m | 0 | 0 | `move_relative(distance=0.25)` |
| `TURN_LEFT` | 2 | 0 | 0 | +15°（逆时针） | `move_relative(angle=+0.2618)` |
| `TURN_RIGHT` | 3 | 0 | 0 | −15°（顺时针） | `move_relative(angle=-0.2618)` |
| `LOOK_DOWN` | 5 | 0 | 0 | 0 | 不调用，原地不动，直接取下一帧 |
| `STOP` | 0 | — | — | — | 不调用，这条指令结束 |

- dx 向前为正，dyaw 左转为正，单位是 m 和 rad，和底座的 `relative` 模式一致。
- 一个动作执行完、停稳，再取下一帧、要下一个动作。动作之间狗是停着的。
- 一条指令通常是几十到几百个这样的动作，最多 500 个。
- `LOOK_DOWN` 时相机固定不动，所以什么都不做。连续出现 3 次，脚本就结束这条指令。
- 在 Python 里，动作是 `src.deploy.Action` 枚举，值就是上表的代码；`steps.jsonl` 里记的是名字，例如 `"FORWARD"`。

### 1.3 和底座的接口

脚本通过底座的 `ROSPlatformClient` 调用，控制模式 `relative`，owner 名 `heatmapvln`：

| 调用 | 什么时候 |
|---|---|
| `wait_for_motion_ready(timeout=40)` | 开始时（`--dry-run` 不调） |
| `acquire(owner="heatmapvln", mode="relative")` | 开始时拿控制权（`--dry-run` 不调） |
| `get_frame("front_camera", timeout=10, wait_new=True)` | 每步取图：动作做完后连取两帧，用第二帧；元数据标为不可用的帧跳过 |
| `move_relative(distance=...)` / `move_relative(angle=...)` | 按 1.2 的表执行；返回的 `state` 不是 `succeeded` 就结束这条指令 |
| `release()` | 结束时一定调用，包括出错和 Ctrl-C |

### 1.4 和 4090 的接口

两个 gRPC 服务，经 ssh 隧道映射到 Orin 本地：

| 服务 | 4090 容器内 | Orin 上访问 |
|---|---|---|
| 模型服务 | `<容器IP>:52400` | `127.0.0.1:52400` |
| 里程计服务 | `<容器IP>:52500` | `127.0.0.1:52500` |

每一步都会往 4090 发一张图，每隔几步等模型返回一次，每次约 2–4 秒，这段时间狗原地不动。

### 1.5 在自己的代码里调用

用整条指令：

```python
from robot_platform.ros_client import ROSPlatformClient
from src.deploy import NavAgent
from scripts.deploy.run_go2 import run_episode

with ROSPlatformClient(timeout=3.0) as robot, NavAgent("127.0.0.1:52400", "127.0.0.1:52500") as agent:
    summary = run_episode(robot, agent, instruction="Walk to the door and stop.", run_id=1)
```

自己控制每一步（`robot`、`agent` 同上）：

```python
from src.deploy import Action
from scripts.deploy.run_go2 import next_jpeg, to_model_frame

agent.reset("Walk to the door and stop.", scene_id="go2lab", episode_id=run_id)   # 每条指令一次，run_id 每次不同
while not agent.done:
    frame = to_model_frame(next_jpeg(robot, "front_camera", 10.0), 100.0)        # 640×480 RGB uint8
    action = agent.act(frame, lambda: frame, lambda: frame)                        # 返回一个 Action，三个参数传同一帧
    # 按 1.2 的表执行 action，做完再进下一轮
```

`act()` 抛出异常后这条指令作废，要重新 `reset()`。

## 2. 安装（只做一次）

### 2.1 4090：起模型服务

在 4090 宿主机的 tmux 里执行，`NAV_GPU` 换成一张空卡：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/nav_src_ded4bd3 && NAV_REPO=$PWD NAV_GPU=4 NAV_HOST=0.0.0.0 bash scripts/deploy/start_nav_servers_cuda.sh'
```

出现 `[nav-servers] READY ...` 就可以连了。Ctrl-C 停止；端口没有认证，不用时就停掉。

### 2.2 Orin：环境和代码

在底座的 shell 里执行：

```bash
cd /home/irmv/robot_platform_ws && ./platform shell
python3 -m venv --system-site-packages ~/heatmapvln_venv && source ~/heatmapvln_venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade "numpy==1.24.4" "pillow==10.4.0"
scp -r 6024_fjl:/home/fangjialei/rpc ~/vla_rpc_src && pip install -e ~/vla_rpc_src
git clone -b claude/hardware-docs-optimization-d11eb1 https://github.com/HarryFang30/HeatmapVLN.git ~/HeatmapVLN
cd ~/HeatmapVLN && python scripts/deploy/run_go2.py --offline     # 打印 offline check OK 即装好
```

`6024_fjl` 是 4090 的 ssh 别名，账号和地址找负责人要，配进 Orin 的 `~/.ssh/config`。

### 2.3 Orin：开隧道

在 4090 宿主机上查容器 IP：`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fjl-habitat`。
然后在 Orin 上开隧道，运行期间一直开着：

```bash
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L 52400:<容器IP>:52400 -L 52500:<容器IP>:52500 6024_fjl
```

## 3. 运行

在 `./platform shell` 里激活 venv，`cd ~/HeatmapVLN`：

```bash
./platform run --allow-motion                                               # 起底座（只看不走用 ./platform run）
python scripts/deploy/run_go2.py --dry-run -i "Walk to the door and stop."  # 不动，只打印动作
python scripts/deploy/run_go2.py -i "Walk to the door and stop."            # 正式跑
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i`, `--instruction` | 必填 | 英文导航指令 |
| `--dry-run` | 关 | 取图、调模型、打印动作，不拿控制权、不动 |
| `--offline` | 关 | 不连狗和 4090，检查安装 |
| `--model` / `--vo` | `127.0.0.1:52400` / `127.0.0.1:52500` | 两个服务的地址 |
| `--camera` | `front_camera` | 底座里的相机名 |
| `--hfov` | `100` | 相机水平视场（度），用于裁剪 |
| `--max-steps` | `500` | 最多动作数 |
| `--run-id` | 当前时间 | 每次运行的编号 |
| `--out` | `runs/go2/<时间>_<编号>/` | 记录目录 |

脚本结束的情况：模型输出 `STOP`、走满 500 步、底座动作失败、连不上服务，或者按 Ctrl-C。无论哪种都会先 `release()`。

## 4. 输出文件

| 文件 | 内容 |
|---|---|
| `frames/step_0000.jpg` … | 每一步从底座拿到的原始画面 |
| `steps.jsonl` | 每步一行：`step`、`action`（动作名）、`motion`（底座返回的 `state` / `reason`）、`seconds`（取图 / 模型 / 动作耗时） |
| `summary.json` | 指令、步数、用时、`outcome` |

`outcome` 的取值：
- `model_stop`：模型输出 STOP；
- `step_cap`：走满步数；
- `fallback_stop` / `empty_chunk`：模型没给出可执行动作；
- `failed`：出错或中断，原因在 `error`。

## 5. 停止和常见问题

- 急停：另开一个 `./platform shell` 执行 `ros2 service call /platform/stop std_srvs/srv/Trigger {}`，或用遥控器接管。模型不避障。
- 同一时间只能有一个程序拿控制权，不要同时开 PanaV。

| 现象 | 处理 |
|---|---|
| 连不上 `127.0.0.1:52400` | 隧道没开，或 4090 上的服务没起 |
| `fresh frame timed out: front_camera` | 底座没起，或相机还在预热（约 5 s） |
| `acquire` 报错，提示 `--allow-motion` | 底座是只读启动的；要动就用 `./platform run --allow-motion` |
| `relative motion failed` | 控制权丢了、急停了或被占用；重跑这条指令 |
| `LOOK_DOWN 3 times in a row` | 换个起点或换句指令再跑 |
