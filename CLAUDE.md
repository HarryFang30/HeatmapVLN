# CLAUDE.md

给在本仓库工作的 Claude Code 的指引。

## 开发模式：本地编辑，远端执行

代码在**本地 Mac** 上编辑，在**沐曦 C500 集群节点**上执行（ssh 别名 `finn_cci_c500`）。

```
本地 Mac (编辑)  ──push──▶  GitHub  ──pull──▶  服务器 (跑训练/测试)
```

一轮迭代：本地改 → commit（钩子自动推送）→ 服务器 `git pull` → 跑。

**为什么不在服务器上直接跑 Claude Code**：该节点出口在上海，`api.anthropic.com` 返回
`forbidden / Request not allowed`（地域限制）。不要尝试绕过它。同理，VS Code Remote-SSH
里的 Claude 扩展也跑在远端，一样不可用。

**服务器只拉不推。** 它是执行器，不是编辑器。所有提交都从本地发起，这样永远不会有双向冲突。

**不要把大量文件读写操作放到 ssh 上。** 即使有连接复用，单次远程命令仍需约 0.5 秒，而
本地文件操作接近零延迟。把编辑留在本地，只把「跑」这一步放到远端。

## 服务器关键信息

| 项 | 值 |
|---|---|
| 工作区根目录 | `/mnt/afs/liwenhao/agent/370910109` |
| 本仓库 | `<根目录>/HeatmapVLN` |
| Python | `<根目录>/envs/qwen25/bin/python`（3.12.13 + pytest 9.0.3）|
| 加速卡 | **沐曦 C500（MACA）**，不是 NVIDIA |

显卡是沐曦不是 N 卡，所以**没有 `nvidia-smi`**，torch 走的是 triton 的 metax 后端。
脚本命名里的 `mxc500` 指的就是这个。

## 两个会浪费你时间的坑

**1. 远程命令必须用 login shell。**
非交互式 ssh 不加载 `.bashrc`（Ubuntu 默认在开头就 `return`），导致 `MACA_PATH` 未设置，
triton 的 metax 后端会抛 `TypeError: expected str, bytes or os.PathLike object, not NoneType`，
表现为**几十个测试收集失败**。看着像代码坏了，其实只是环境没加载。

```bash
# 对：
ssh finn_cci_c500 'bash -lc "cd ... && python -m pytest"'
# 错（MACA_PATH 会是空的）：
ssh finn_cci_c500 'cd ... && python -m pytest'
```

**2. git 需要 safe.directory。**
仓库目录属主是 uid 1024，而你以 root 身份操作，git 会拒绝并报 `dubious ownership`。
远端所有 git 命令都要带 `-c safe.directory=<仓库路径>`。

## 跑测试

```bash
ssh finn_cci_c500 'bash -lc "cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN && /mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python -m pytest tests/ -q --continue-on-collection-errors"'
```

长任务（训练、评测）**务必套 tmux**。这条链路经过中转，掉线会连带杀掉裸跑的进程。

## 已知失败的测试（与你的改动无关）

以下 7 个失败 + 1 个收集错误在干净的 `main` 上就存在，基线是 **980 passed**。
看到它们不用排查，也不要试图「顺手修好」——先确认不是你引入的即可：

- `test_distributed_sync_multiprocess.py::test_rank_local_missing_gradients_keep_collectives_and_updates_aligned`
- `test_internnav_heatmap_control_collator.py`（3 个）
- `test_target_grounded_identity.py`（3 个）
- `test_stage3_dataloader_order.py` — 收集错误：从 `scripts.train` 导入
  `_dataloader_in_order_kwargs`，但该函数已不存在

判断是否为回归的办法：把干净的 HEAD 导出到临时目录单独跑，不要动真实仓库。

```bash
git -c safe.directory=$R -C $R archive HEAD | tar -x -C /tmp/hv-head
```

## 路径迁移（2026-08）

工作区从旧路径搬到了新路径：

```
/mnt/afs/lixiaoou/intern/fjl   →   /mnt/afs/liwenhao/agent/370910109
```

**在任何地方看到旧路径 `/mnt/afs/lixiaoou/intern/fjl`，都是迁移遗留的过期值**，
按上面的对应关系替换即可。仓库内被 git 跟踪的文件已全部更新（233 处，75 个文件）；
服务器上仍有残留的，都是 `.pyc`、`.log` 这类 gitignore 的产物，可忽略。

两个**不能**套用上述规则的例外：

- **conda 不在工作区下。** 旧脚本里的 `<旧根目录>/miniconda3/etc/profile.d/conda.sh`
  应指向 `/opt/conda/etc/profile.d/conda.sh`。
- **`r2r_paronamic_data` 已被删除。** 指向它的引用现在解析不到任何东西。不要凭空编一个
  替代路径——需要这份数据时先问用户。

## 提交约定

- commit message 里**不要**加 AI co-author 署名。
- `.git/hooks/post-commit` 会**自动 push**，所以 `git commit` 一执行就等于推到了远端。
  提交前先确认改动是你想公开的。
- 本仓库在 GitHub 上是 **public**，不要往里写服务器 IP、端口、密钥或其他凭据。
