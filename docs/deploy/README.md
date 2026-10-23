# Go2 实机：用狗本体跑导航模型

模型跑在 4090 服务器上；狗上的 Orin 跑实验室的底座 `robot_platform_ws`，再跑我们的一个脚本 `scripts/deploy/run_go2.py`。
脚本用狗自带的前视摄像头取图，交给 4090 上的模型，再把模型给的动作（前进 0.25 m、左转或右转 15°、停）交给底座去走。
**不加任何硬件。**

> 到 2026-10-23 为止，模型没有在真机上跑过。脚本只在模拟的底座上跑通过。
> 细节都在 [`navigation_interface.md`](navigation_interface.md)，下文的 §n 指它。

## 要什么

- 一台 Go2 EDU，Orin 上装好了实验室的底座 `robot_platform_ws`（就是 PanaV 在用的那套）。
- 4090 服务器（ssh 别名 `6024_fjl`）上一张空卡：两个服务端放一张卡，约占 40 GB。
- Orin 能 ssh 到这台 4090：账号、地址和跳板找负责人要，不写在仓库里。
- 一块空地，跑的时候有人拿着遥控器跟着。模型完全不避障。

## 一次性准备

**1. 在 4090 上起服务端**（§3）。在宿主机的 tmux 里跑，`NAV_GPU` 填一张空卡，先用 `nvidia-smi` 看一下：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/nav_src_ded4bd3 && NAV_REPO=$PWD NAV_GPU=4 NAV_HOST=0.0.0.0 bash scripts/deploy/start_nav_servers_cuda.sh'
```

看到 `[nav-servers] READY ... attn=flash_attention_2` 就好了。Ctrl-C 停。用完就停，这两个端口没有认证。

**2. 在 Orin 上装环境**（§5）。所有命令都在底座的 shell 里跑：

```bash
cd /home/irmv/robot_platform_ws && ./platform shell
python3 -m venv --system-site-packages ~/heatmapvln_venv && source ~/heatmapvln_venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade "numpy==1.24.4" "pillow==10.4.0" pytest
scp -r 6024_fjl:/home/fangjialei/rpc ~/vla_rpc_src && pip install -e ~/vla_rpc_src
git clone -b claude/hardware-docs-optimization-d11eb1 https://github.com/HarryFang30/HeatmapVLN.git ~/HeatmapVLN
cd ~/HeatmapVLN
python -m pytest tests/test_nav_agent.py tests/test_run_go2.py --noconftest -q    # 应为 31 passed
python scripts/deploy/run_go2.py --offline                                        # 最后一行是 offline check OK
```

**3. 开 ssh 隧道**（§4）。先在 4090 宿主机上查容器的 IP：

```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fjl-habitat
```

再在 Orin 上开隧道，开着别关：

```bash
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L 52400:<容器IP>:52400 -L 52500:<容器IP>:52500 6024_fjl
```

在 Orin 上测一下通不通：

```bash
python -c "from src.deploy import NavAgent; NavAgent('127.0.0.1:52400', '127.0.0.1:52500').close(); print('ok')"
```

## 每次跑

都在 `./platform shell` 里、venv 激活、`cd ~/HeatmapVLN` 以后运行。指令用英文，像 "Walk past the sofa and stop at the kitchen door." 这样。

1. **起底座**：只看不动用 `./platform run`，要让狗走用 `./platform run --allow-motion`。
2. **另开一个 `./platform shell`，把急停命令敲好放着**：

   ```bash
   ros2 service call /platform/stop std_srvs/srv/Trigger {}
   ```

3. **第一次先只看不动**：狗不走，动作只打印，用来确认取图和模型都通。

   ```bash
   python scripts/deploy/run_go2.py --dry-run -i "Walk to the door and stop."
   ```

4. **正式跑**：

   ```bash
   python scripts/deploy/run_go2.py -i "Walk to the door and stop."
   ```

   狗每走一步停一下，每走三四步停下来等模型 2–3.5 秒。模型说停、出错或按了 Ctrl-C，脚本都会结束，并把控制权还给底座。

每次跑的记录在 `runs/go2/<时间>_<编号>/`：
- `frames/` 是每一步的原始画面；
- `steps.jsonl` 每步一行，写着动作和用时；
- `summary.json` 写着怎么结束的，`outcome` 是 `model_stop` 表示模型认为到了。

跑完把整个目录拷回来就行。

**第一次接真狗之前**，建议先用底座的模拟后端把接线跑一遍（§7 第 4 步）：

```bash
./platform run --backend sim --allow-motion
python scripts/deploy/run_go2.py --fake-servers -i "test"
```

## 出问题

| 现象 | 原因和办法 |
|---|---|
| 连不上 `127.0.0.1:52400` | 隧道没开，或 4090 上服务端没起 / 没到 `READY` |
| 取图超时（`RosPlatformError`） | 底座没起，或 `front_camera` 没出图：`./platform cameras --cameras front_camera` 看一下 |
| `wait_for_motion_ready` 或 `acquire` 报错，提示加 `--allow-motion` | 底座是只读启动的。要走就用 `--allow-motion` 重启底座；只看就加 `--dry-run` |
| `relative motion failed` | 底座没走成：控制权丢了、急停了，或被别的程序（比如 PanaV）占着。本集作废，重新跑 |
| `LOOK_DOWN 3 times in a row` | 模型在这个画面里找不到路。换个起点、换句更短的指令再试 |
| 狗要撞东西 | 遥控器接管，或另一个 shell 里执行急停命令 |

## 预期

- **速度**：底座走一步约 2.3 s，模型规划一次 2–3.5 s，一条要走 100 步左右的指令大约 5–6 分钟。
- **成绩**：仿真里换成狗自带前视这个位置（0.4 m、水平），成功率比模型训练时的视角低约 10 个点（§6.2）。
  真实环境里会差多少，只能跑了看。
- 想多拿回几个点，可以以后在狗背上加一台下倾 15°、离地 0.6–0.7 m 的相机，仿真里能好约 5 个点（§6.2 最后一段）。现在不需要。
