# 在 Go2 上跑导航模型

导航模型的服务跑在 4090 服务器上。狗上的 Orin 跑实验室底座 `robot_platform_ws`，再跑本仓库的脚本 `scripts/deploy/run_go2.py`。
脚本做的事和 PanaV 接底座的方式一样：

- 从底座取狗自带前视 `front_camera` 的画面，经 ssh 隧道发给 4090 上的模型；
- 收到动作后交给底座的相对动作 `/platform/relative_motion` 去执行。

不需要额外的硬件。

```
4090 服务器（docker 容器 fjl-habitat）
    模型服务 :52400    里程计服务 :52500
          ▲  gRPC，经 ssh 隧道
          │
Go2 上的 Orin
    run_go2.py ── ROS 2 ── 底座 robot_platform_ws ── Go2（front_camera、运动）
```

## 1. 要准备的

- Go2 EDU，Orin 上装好底座 `robot_platform_ws`，能 `./platform run --allow-motion`。
- 4090 服务器（ssh 别名 `6024_fjl`）上一张空的 48 GB 卡。
- Orin 能 ssh 到 4090：账号、地址、跳板找负责人要。
- 一块空地，跑的时候有人拿遥控器跟着。模型不避障。

## 2. 一次性安装

### 2.1 4090：起模型服务

在 4090 宿主机的 tmux 里执行，`NAV_GPU` 换成一张空卡（先 `nvidia-smi` 看）：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/nav_src_ded4bd3 && NAV_REPO=$PWD NAV_GPU=4 NAV_HOST=0.0.0.0 bash scripts/deploy/start_nav_servers_cuda.sh'
```

- 出现 `[nav-servers] READY model=0.0.0.0:52400 vo=0.0.0.0:52500 ...` 就可以连了。
- Ctrl-C 停止。这两个端口没有认证，不用时就停掉。

### 2.2 Orin：装环境、拿代码

在底座的 shell 里做，这样 ROS 和底座的 Python 包都已加载：

```bash
cd /home/irmv/robot_platform_ws && ./platform shell
python3 -m venv --system-site-packages ~/heatmapvln_venv && source ~/heatmapvln_venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade "numpy==1.24.4" "pillow==10.4.0" pytest
scp -r 6024_fjl:/home/fangjialei/rpc ~/vla_rpc_src && pip install -e ~/vla_rpc_src
git clone -b claude/hardware-docs-optimization-d11eb1 https://github.com/HarryFang30/HeatmapVLN.git ~/HeatmapVLN
cd ~/HeatmapVLN
python scripts/deploy/run_go2.py --offline        # 不连狗、不连 4090，最后打印 offline check OK 就是装好了
```

`python3 -m venv` 报 ensurepip 不可用时，找底座的维护者装 `python3.8-venv`。

### 2.3 Orin：开到 4090 的隧道

先在 4090 宿主机上查容器 IP：

```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fjl-habitat
```

在 Orin 上开隧道（Orin 的 `~/.ssh/config` 里先配好 `Host 6024_fjl`），跑的时候一直开着：

```bash
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L 52400:<容器IP>:52400 -L 52500:<容器IP>:52500 6024_fjl
```

检查能不能连上：

```bash
python -c "from src.deploy import NavAgent; NavAgent('127.0.0.1:52400', '127.0.0.1:52500').close(); print('ok')"
```

## 3. 每次运行

以下命令都在 `./platform shell` 里、激活 venv、`cd ~/HeatmapVLN` 之后执行。

1. 起底座：

   ```bash
   ./platform run --allow-motion        # 只看不走时用 ./platform run
   ```

2. 另开一个 `./platform shell`，把急停命令准备好：

   ```bash
   ros2 service call /platform/stop std_srvs/srv/Trigger {}
   ```

3. 先只看不走，确认取图和模型都通：

   ```bash
   python scripts/deploy/run_go2.py --dry-run -i "Walk to the door and stop."
   ```

4. 正式跑：

   ```bash
   python scripts/deploy/run_go2.py -i "Walk to the door and stop."
   ```

- 指令用英文，说清楚要经过哪里、停在哪里，例如 "Walk past the sofa, turn left, and stop at the kitchen door."
- 一次运行跑一条指令。脚本结束的情况：
  - 模型说「到了」；
  - 走满 500 步；
  - 底座动作失败或连不上服务；
  - 按 Ctrl-C。
- 无论怎么结束，脚本都会把控制权还给底座。
- 狗每走几步会原地停 2–4 秒等模型，这是正常的。
- 第一次接真狗前，可以先用底座的模拟模式检查接线：
  先 `./platform run --backend sim --allow-motion`，再 `python scripts/deploy/run_go2.py --fake-servers -i "test"`。

## 4. 接口

### 4.1 命令行

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i`, `--instruction` | 必填 | 英文导航指令 |
| `--dry-run` | 关 | 取图、调模型、打印动作，但不拿控制权、不让狗动 |
| `--offline` | 关 | 不连狗、不连 4090，用测试图和假模型跑一遍，检查安装 |
| `--fake-servers` | 关 | 连狗或底座模拟模式，模型换成假的，只检查接线 |
| `--model` | `127.0.0.1:52400` | 模型服务地址（隧道的本地端） |
| `--vo` | `127.0.0.1:52500` | 里程计服务地址（隧道的本地端） |
| `--camera` | `front_camera` | 底座里的相机名 |
| `--hfov` | `100` | 这台相机的水平视场（度），用来把画面裁成模型要的大小 |
| `--max-steps` | `500` | 最多走多少步 |
| `--run-id` | 当前时间 | 每次运行的编号，不同运行要不同，一般不用设 |
| `--out` | `runs/go2/<时间>_<编号>/` | 记录保存位置 |

### 4.2 用到的底座接口

都通过底座的 `ROSPlatformClient`，控制模式是 `relative`，控制权的 owner 名是 `heatmapvln`。

| 底座接口 | 什么时候用 |
|---|---|
| `wait_for_motion_ready(timeout=40)` | 开始时，等底座运动预检就绪（`--dry-run` 不调） |
| `acquire(owner="heatmapvln", mode="relative")` | 开始时拿控制权（`--dry-run` 不调） |
| `get_frame("front_camera", wait_new=True)` | 每一步取图：动作做完后连取两帧，用第二帧；底座标成不可用的帧跳过 |
| `move_relative(distance=0.25)` | 模型给「前进」 |
| `move_relative(angle=±0.2618)` | 模型给「左转 / 右转 15°」，左转为正 |
| `release()` | 结束时，任何情况都会调 |

### 4.3 模型给的动作和对应的执行

| 动作 | 狗做什么 |
|---|---|
| `FORWARD` | 前进 0.25 m |
| `TURN_LEFT` | 原地左转 15° |
| `TURN_RIGHT` | 原地右转 15° |
| `LOOK_DOWN` | 不动，直接取下一帧。连续 3 次就结束这次运行 |
| `STOP` | 停下，结束 |

### 4.4 在自己的程序里调用

不想用命令行时，可以在自己的代码里直接调用（同样要在 `./platform shell` 里、`~/HeatmapVLN` 下运行）：

```python
from robot_platform.ros_client import ROSPlatformClient
from src.deploy import NavAgent
from scripts.deploy.run_go2 import run_episode

with ROSPlatformClient(timeout=3.0) as robot, NavAgent("127.0.0.1:52400", "127.0.0.1:52500") as agent:
    summary = run_episode(robot, agent, instruction="Walk to the door and stop.", run_id=1)
    print(summary["outcome"])
```

自己控制每一步时，模型的接口只有两个（`robot`、`agent` 同上）：

```python
from src.deploy import Action
from scripts.deploy.run_go2 import next_jpeg, to_model_frame

agent.reset("Walk to the door and stop.", scene_id="go2lab", episode_id=run_id)   # 每条指令一次
while not agent.done:
    frame = to_model_frame(next_jpeg(robot, "front_camera", 10.0), 100.0)        # 640×480 RGB uint8
    action = agent.act(frame, lambda: frame, lambda: frame)                        # 返回一个 Action
    ...                                                                            # 按 4.3 执行
```

- `act()` 的三个参数都传同一帧。
- 动作执行完、狗停稳以后再取下一帧。
- `act()` 抛异常后这条指令作废，要重新 `reset()`。

## 5. 输出

每次运行在 `--out` 目录下写三样东西，跑完把整个目录拷走即可：

| 文件 | 内容 |
|---|---|
| `frames/step_0000.jpg` … | 每一步从底座拿到的原始画面 |
| `steps.jsonl` | 每步一行：动作、模型这一步的输出、底座返回的结果、取图 / 模型 / 动作各用了几秒 |
| `summary.json` | 指令、走了几步、用时、怎么结束的（`outcome`） |

`outcome` 的含义：

| 值 | 含义 |
|---|---|
| `model_stop` | 模型判断到了 |
| `fallback_stop` 或 `empty_chunk` | 模型没给出能执行的动作 |
| `step_cap` | 走满步数 |
| `failed` | 出错或被中断，原因在 `error` |

## 6. 停止和安全

- 随时 Ctrl-C：脚本会释放控制权，底座清零输出。
- 急停：在另一个 `./platform shell` 里执行 `ros2 service call /platform/stop std_srvs/srv/Trigger {}`，或者用遥控器接管。
- 脚本掉线时，底座在控制租约（1 s）到期后自动停。
- 不要同时开 PanaV 或别的控制程序，同一时间只能有一个程序拿到控制权。
- 前几次在空旷的地方、用短指令跑。

## 7. 常见问题

| 现象 | 怎么办 |
|---|---|
| 连不上 `127.0.0.1:52400` | 隧道没开，或 4090 上的服务没起 / 还没出现 `READY` |
| `fresh frame timed out: front_camera` | 底座没起，或相机还在预热（开机后约 5 s）。`./platform cameras --cameras front_camera` 看一下 |
| `wait_for_motion_ready` / `acquire` 报错，提示加 `--allow-motion` | 底座是只读启动的：要走就用 `./platform run --allow-motion` 重启；只看就加 `--dry-run` |
| `relative motion failed` | 控制权丢了、被急停了，或者被别的程序占着。这次作废，重跑 |
| `LOOK_DOWN 3 times in a row` | 模型在当前画面里找不到方向。换个起点或者换一句更简单的指令 |
| 画面裁得不对 | 用 `--hfov` 改这台相机的水平视场，默认 100° |

更细的内容（服务端参数、网络、延迟）见 [`navigation_interface.md`](navigation_interface.md)。
