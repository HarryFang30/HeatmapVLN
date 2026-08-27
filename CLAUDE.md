# CLAUDE.md

给在本仓库工作的 Claude Code 的指引。这里只写**从代码里看不出来、但会实实在在浪费时间**的事。

---

## 1. 开发模式：本地编辑，开发机验证，集群任务走网站提交

代码在**本地 Mac** 上编辑；ssh 别名 `finn_cci_c500` 连的是**开发机**，不是集群
计算节点。正式的训练/评测/采集任务在**网站上提交**，落到一个挂载 `/mnt/afs`
共享存储的**空白容器**里执行。

```
本地 Mac (编辑) ──push──▶ GitHub ──pull──▶ 开发机 (测试/冒烟/看日志)
                                              │
                          网站提交 ──▶ 集群空白容器 (正式训练/评测/采集)
```

一轮迭代：本地改 → `git commit`（钩子自动推送）→ 开发机 `git pull` 验证 →
把正式任务写成提交物交网站。

### 1.1 网站提交物的形态

一段 shell：`cd 工作区` + 一串 `export` + 一条 `bash scripts/xxx.sh`（或直接的
python 命令）。形如：

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export PPA_DATA_ROOT=...
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_xxx_mxc500.sh
```

由此对脚本的硬约束（写新启动脚本时必须满足）：

- **参数用环境变量传**，不要依赖交互输入；位置参数只留给开发机冒烟。
- **容器是空白的**：只保证挂载 `/mnt/afs` 和 `/opt/maca-*`。不要假设
  `/opt/conda`、tmux、已起的 Xvfb、任何交互态存在。conda 环境一律用绝对路径
  的 `bin/python` 直接调，不要 `conda activate`。
- **不要教用户对集群任务套 tmux** —— 网站任务本身就是常驻进程。tmux 只用于
  确需在开发机上直跑的较长调试任务（这条 ssh 链路经中转，掉线杀裸进程）。
- 所有路径写 `/mnt/afs` 绝对路径（脚本里已如此，见 §6）。

**为什么不在服务器上跑 Claude Code。** 该节点出口在上海，`api.anthropic.com` 返回
`forbidden / Request not allowed`（地域限制）。**不要尝试绕过它。** 同理，VS Code
Remote-SSH 里的 Claude 扩展跑在远端，一样不可用。

**服务器只拉不推。** 它是执行器，不是编辑器。所有提交从本地发起，这样永远没有双向冲突。

**不要把大量文件读写放到 ssh 上。** 即使有连接复用，单次远程命令仍需约 0.5 秒，本地
文件操作接近零延迟。编辑留本地，只把「跑」放远端。日志、数据集、checkpoint 这类只存在
于服务器的东西才值得走 ssh 看。

> 本地 `~/.ssh/config` 建议开 `ControlMaster auto` + `ControlPersist 10m`。
> 这条链路的 ssh 握手约 2.4 秒，复用后降到 0.5 秒。

---

## 2. 开发机与共享存储环境

下表对开发机和网站提交的集群容器都成立：工作区在共享存储 `/mnt/afs` 上，
两边看到的是同一份。

| 项 | 值 |
|---|---|
| 工作区根目录 | `/mnt/afs/liwenhao/agent/370910109` |
| 本仓库 | `<根目录>/HeatmapVLN` |
| Python | `<根目录>/envs/qwen25/bin/python`（3.12.13 + pytest 9.0.3）|
| conda | `/opt/conda`（**不在工作区下**）|
| 加速卡 | **沐曦 C500（MACA）**，不是 NVIDIA |
| 规格 | 128 核 / 2TB 内存 |

显卡是沐曦不是 N 卡，所以**没有 `nvidia-smi`**，torch 走 triton 的 metax 后端。
脚本名里的 `mxc500` 指的就是这个。

---

## 3. 远程操作的四个坑

### 3.1 远程命令必须用 login shell

非交互式 ssh 不加载 `.bashrc`（Ubuntu 默认在开头就 `return`），于是 `MACA_PATH` 为空，
triton 的 metax 后端抛：

```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

表现为**几十个测试收集失败**，看着像代码坏了，其实只是环境没加载。

```bash
# 对
ssh finn_cci_c500 'bash -lc "cd ... && python -m pytest"'
# 错（MACA_PATH 为空）
ssh finn_cci_c500 'cd ... && python -m pytest'
```

多行脚本用 `ssh finn_cci_c500 'bash -l -s' <<'EOF'`，别嵌套引号自找麻烦。

### 3.2 git 需要 safe.directory

仓库属主是 uid 1024，而你以 root 操作，git 报 `dubious ownership` 拒绝执行。
远端每条 git 命令都要带：

```bash
git -c safe.directory=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN -C <repo> <cmd>
```

### 3.3 过期 `.pyc` 会让堆栈显示旧路径

`__pycache__` 里缓存的字节码带着编译时的 `co_filename`，导致 traceback 里出现
**早已废弃的旧路径**，误导排查方向。看到堆栈路径不对劲就先清：

```bash
find <repo> -name __pycache__ -type d -not -path "*/.git/*" -exec rm -rf {} +
```

### 3.4 在服务器上验证改动的正确姿势

**不要在服务器上 commit，也不要直接编辑服务器上的文件。** 要在远端验证本地改动，
用 patch 走一遭，验完还原：

```bash
git diff > /tmp/x.patch && scp /tmp/x.patch finn_cci_c500:/tmp/
# 远端：apply → 跑测试 → git checkout -- .
```

**如果服务器上已经有未提交的改动**（别人或早先的会话留下的），不要在服务器上提交，
把它捞回本地：

```bash
# 远端：已跟踪的改动
git -c safe.directory=$R -C $R diff HEAD --binary > /tmp/hv.patch
# 远端：未跟踪的新文件
git -c safe.directory=$R -C $R ls-files --others --exclude-standard -z \
  | tar -czf /tmp/hv-untracked.tgz -C $R --null -T -
# 本地：scp 下来，git apply + tar -xzf，然后用 sha256 逐文件校验一致
```

捞完再 `git stash push -u` 留个后路，然后 `git pull` 让服务器回到干净状态。

---

## 4. 跑测试

```bash
ssh finn_cci_c500 'bash -lc "cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN && /mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python -m pytest tests/ -q --continue-on-collection-errors"'
```

全量约 2.5 分钟，在开发机上直接跑。正式长任务（训练、评测、采集）**不在开发机
上跑**，写成提交物走网站（见 §1.1）；只有确需在开发机上直跑的较长调试任务才套
tmux —— 这条链路经过中转，掉线会连带杀掉裸跑的进程。

### 基线

**1004 passed / 1 skipped / 1 collection error。** 只有这一个已知错误：

- `test_stage3_dataloader_order.py` — 从 `scripts.train` 导入 `_dataloader_in_order_kwargs`，
  但这个 helper 从未存在。`in_order` 功能本身是好的，逻辑内联在 `scripts/train.py:1234`。
  测试期待的是一个没被抽出来的函数，**修它等于做重构，不要顺手改**。

已知的环境性抖动（不算基线破坏，先看机器状态再怀疑代码）：

- `test_trajectory_dagger.py::test_same_host_multiprocess_commits_share_one_locked_ledger`
  在开发机高负载/冷 AFS 时偶发失败，签名是 `assert None == 0`（spawn 出的 8 个
  子进程要从 AFS 重新 import 整条依赖链，30 秒 join 超时时子进程还活着）。冷缓存
  时全量套件本身也会从 2.5 分钟膨胀到小时级。同理，**空白容器里的任务启动要几分
  钟才见第一行日志，不要急着杀**。

### 判断一个失败是不是你引入的

把干净的 HEAD 导出到临时目录单独跑，**不要动真实仓库**：

```bash
git -c safe.directory=$R -C $R archive HEAD | tar -x -C /tmp/hv-head
cd /tmp/hv-head && <python> -m pytest <那几个测试>
```

同样失败 → 既有问题，不是你的锅。

---

## 5. 改测试之前先想想

这套测试有两个历史包袱，遇到失败先排除它们，别急着改生产代码。

### 5.1 测试桩绕过 `__init__`

有些测试用 `VLNPipeline.__new__(...)` + `nn.Module.__init__` 造对象，再手工设置
少数几个属性（见 `tests/test_target_grounded_identity.py::_pipeline_stub`）。

所以**你在 `forward()` 里新读一个 `self.xxx`，桩不会自动有它**，会抛
`AttributeError: 'VLNPipeline' object has no attribute 'xxx'`。这看着像生产代码的
bug，其实是桩没跟上 —— 加属性时记得同步更新桩。

### 5.2 测试是在 macOS 上写的

部分测试出自 macOS 的 Codex runner，把平台假设写死了。典型例子：
`GLOO_SOCKET_IFNAME` 曾被固定成 `lo0`（macOS 回环名，Linux 上叫 `lo`），
导致该测试在这台 Linux 节点上从未通过过。

**只在服务器上失败的测试，先怀疑平台假设，再怀疑代码。**

### 5.3 断言匹配的是报错文案

多处 `pytest.raises(..., match="...")` 匹配的是异常消息原文。改报错文案会连带
打挂测试，而这类失败的**行为其实是对的**，只是措辞变了。看到
`Regex pattern did not match` 先对比一下实际消息。

---

## 6. 路径

**所有 configs 和 `run_*_mxc500.sh` 都写死绝对路径**，工作区一旦搬家就整片失效。

### 迁移对照（2026-08）

```
/mnt/afs/lixiaoou/intern/fjl   →   /mnt/afs/liwenhao/agent/370910109
```

**任何地方看到旧路径 `/mnt/afs/lixiaoou/intern/fjl` 都是迁移遗留的过期值。**
仓库内被 git 跟踪的文件已全部更新（233 处 / 75 个文件）；服务器上的残留都是
`.pyc`、`.log` 这类 gitignore 产物，可忽略。

### 两个不能套用上述规则的例外

- **conda 不在工作区下。** 旧脚本里的 `<旧根>/miniconda3/etc/profile.d/conda.sh`
  应指向 `/opt/conda/etc/profile.d/conda.sh`。
- **`r2r_paronamic_data` 已被删除。** 指向它的引用现在解析不到任何东西。
  **不要凭空编一个替代路径** —— 需要这份数据时先问用户。

---

## 7. 提交约定

- commit message 里**不要**加 AI co-author 署名。
- `.git/hooks/post-commit` 会**自动 push**。`git commit` 一执行就等于推到了远端，
  没有反悔窗口 —— 提交前先确认改动是你想公开的。
- **本仓库在 GitHub 上是 public。** 不要往里写服务器 IP、端口、密钥或任何凭据。
  仓库里已有大量集群内部绝对路径，别再增加新的敏感信息。
- 推之前先在服务器上跑一遍全量测试，对照上面的基线确认没引入回归。
