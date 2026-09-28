# 部署在 RTX 4090 服务器（ssh 别名 `6024_fjl`）

2026-09-28 部署并跑通。机器是多人共用的：6 张 RTX 4090（48 GB），Ubuntu 24.04，CUDA 12.8 驱动，没有 conda；环境在 docker 容器 `fjl-habitat` 里。**开跑前先看 `nvidia-smi`，只用空着的卡**（部署时 1–3 号卡有别人的进程）。

## 1. 布局

容器把 `/home/fangjialei` 挂在 `/workspace`，把 `/data0/dataset` 挂在 `/dataset`。下表都是容器内路径。

| 物 | 路径 | 来源 |
|---|---|---|
| 代码 | `/workspace/HeatmapVLN` | 本仓库（部署时 `fc836bf` 起） |
| RPC 工具 `vla_rpc` | `/workspace/rpc` | 迁移包 `rpc_repo_with_git.tgz`：`21cc2fc` + 未提交改动，内容与 `b35e060` 相同 |
| AMB3R | `/workspace/amb3r` | 上游 `92c4081`（codeload 源码包）+ `amb3r_local.patch` + `amb3r_untracked.tgz`，各一个本地提交 |
| DA3 权重 | `/workspace/amb3r/checkpoints/DA3NESTED-GIANT-LARGE` | hf-mirror `depth-anything/DA3NESTED-GIANT-LARGE@8615eef`，`model.safetensors` sha256 `8899faf9…2ddd` |
| 慢系统 / 快系统底座 | `/workspace/InternNav_Model` | InternVLA-N1 发布权重（完整版） |
| 部署权重 | `/workspace/weights/ppa_refine_v2_best.pth` | sha256 `0b5a0644…6d69`（v2 信赖域桥） |
| 锁定评测计划 | `/workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802` | 迁移包 `evaluation_plans_all.tgz` |
| R2R val_unseen | `/workspace/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz` | sha256 `1767a407…d167c3`，与锁定计划一致 |
| MP3D 场景 | `/dataset/mp3d` | 11 个 val_unseen 场景齐全 |

**Python**：容器里只有一个 `/opt/conda/bin/python`（3.11）。它同时装了 torch 2.11+cu128、transformers 4.51.0（`runtime_compat` 要求必须是这个版本）、habitat-sim 0.1.7 和可编辑安装的 habitat-lab 0.1.7（源码就是 `/workspace/habitat-lab`，**不能删**），所以服务端和客户端共用这一个解释器。

部署时补装了两个包：`utils3d 0.0.2`（AMB3R 自带 wheel）和 `addict 2.4.0`（阿里云镜像）。

## 2. 跑评测

在宿主机上：

```bash
docker start fjl-habitat
```

然后进入容器跑启动脚本（下面是全量、种子 42、用 4、5 号卡）：

```bash
docker exec -it fjl-habitat bash -c 'cd /workspace/HeatmapVLN && PPA_EVAL_GPU_DEVICES=4,5 PPA_EVAL_PROTOCOL_SEED=42 bash scripts/run_ppa_r2r_val_unseen_cuda.sh'
```

- 协议和客户端参数与 C500 上认证过的 8 卡评测相同：同一份 8 分片锁定队列、确定性采样、在线 AMB3R、llvmpipe CPU 渲染。
- 可以用 1–8 张卡。8 个分片轮流分给各张卡，每张卡依次跑自己分到的分片。
- 模型服务端和里程计服务端默认放在同一张卡上，实测空载约 36 GB、运行中见到最高约 37 GB（48 GB 的卡）。显存不够时用 `PPA_EVAL_VO_GPU_DEVICES` 把里程计放到别的卡。
- 冒烟 / 金丝雀：加 `PPA_EVAL_SHARDS=0,1 PPA_EVAL_MAX_EPISODES_PER_SHARD=2`，只打印汇总，不合并。
- 输出默认在 `/workspace/eval_runs/ppa_refine_v2_seed<种子>/`（金丝雀在 `canary_seed<种子>/`）。**换种子一定换输出目录**，否则 `--resume` 会跳过另一个种子已跑的集。
- 跑满 8 个分片且不设上限时，会用锁定计划的 `merge_shards.py` 合并并自检，最后打印 `"status": "passed"`。
- 长任务在宿主机上用 tmux 或 `docker exec -d … > log 2>&1` 挂起，ssh 断开不会影响。

**速度**：每张卡约 1–1.3 秒一步（CPU 渲染 + 在线建图）。全量 1839 集在 2 张卡上大约一天。

## 3. 部署验证（2026-09-28）

- 模型服务端单独启动约 40 秒。日志里有 `Formal PPA online AMB3R runtime enabled … tensors={'heatmap': 79, 'future': 11, 'bridge': 10}`。
- 金丝雀：4、5 号卡两个槽位，分片 0、1 各 2 集。结果：4 集全部成功，SPL 87.0%，NE 0.91 m，注入生效 52 次，位姿全部来自 `amb3r_vo_da3`。

| 分片 | 集 | 成功 | SPL | NE (m) | 步数 | 注入生效 |
|---|---|---|---|---|---|---|
| 0 | zsNo4HB9uLZ / 1 | 1 | 0.995 | 0.27 | 49 | 7 |
| 0 | zsNo4HB9uLZ / 25 | 1 | 0.820 | 1.94 | 105 | 17 |
| 1 | zsNo4HB9uLZ / 2 | 1 | 0.741 | 0.93 | 89 | 12 |
| 1 | zsNo4HB9uLZ / 26 | 1 | 0.924 | 0.51 | 94 | 16 |

- **与 C500 的数字不会逐集相同**。快系统噪声由 CUDA 生成器按调用播种，bf16 算子也因平台而异。要判断部署是否等价，就跑全量两种子，与台账 §4 的 62.81% / 61.17% 比。

## 4. 这台机器的坑

- **仓库里大量文件属主是 root**（以前在容器里以 root 跑出来的），包括 `.git/index`，所以 git 操作要在容器里做：`git -c safe.directory='*' …`。
- **容器连不上 GitHub**，宿主机能连但很慢。更新代码的办法：在本地 `git bundle create x.bundle <旧提交>..<新分支>`，scp 上去，在容器里 `git fetch <bundle> <ref>:<ref>` 再快进。
- **HuggingFace 不通**，下载权重用 `https://hf-mirror.com/<repo>/resolve/<commit>/<file>`；pip 用阿里云镜像。
- 容器的 `.bashrc` 设了 `127.0.0.1:7890` 的代理，但代理并不存在。登录 shell 里装包或下载前先 `unset http_proxy https_proxy`（启动脚本已经处理）。
- 容器是 Ubuntu 20.04，bash 5.0，**没有 `wait -n -p`**，所以不能直接用 C500 的启动脚本。
- habitat-sim 是 GLX 版，必须有 X 服务。启动脚本为每张卡起一个 Xvfb，并按 TCP 探测就绪。NVIDIA GLX 渲染在这台机器上与 numba 冲突（`troubleshooting-guide.md` §12），所以默认走 llvmpipe。
- 在容器里用 `pkill -f <模式>` 时，模式会匹配到 `bash -c` 自己那一行，把自己的 shell 杀掉。停进程请用 PID。
