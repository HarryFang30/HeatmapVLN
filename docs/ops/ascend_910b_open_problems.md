# 昇腾 910B 部署：现在还有什么问题

写于 2026-10-10 04:40 北京 / 2026-10-09 16:40 美东。
部署怎么跑见 [`deploy_ascend_910b.md`](deploy_ascend_910b.md)；这份只写**还差什么、哪里会坑你**。

一句话：**服务端在昇腾上跑通了，但这台机器上一个数都还不能报**；而且在下一次开跑之前有
七处要先处理（§2 五处 + §7.1、§7.2），**其中四处会悄悄给你错的东西，而不是报错**——一个
空卡检查永远读到 0、一个证据检查永远不会失败、一个重启会偷偷多跑集、一个补丁不打也不报警。

每条都给证据。标了 ✅ 的是我在真机上跑出来确认过的；标了 ⚠️ 的是读代码推出来、还没在真机
上触发过的；标了 ❓ 的是没有定论的猜测。

---

## 0. 先清场：机器上现在还挂着东西

| 位置 | 残留 | 影响 |
|---|---|---|
| 910B | 两套服务端（02:54 和 03:24 起的，诊断用），占住卡 0、卡 1 | 卡 0 的 HBM 停在 **65521 / 65536 MiB**，这张卡现在塞不进任何东西 |
| 910B | `zhr_1/tmp/{m0,v0}` 下 12 个泄漏的 `pymp-*` 和 3 个 `amb3r_vo_cfg_*` | 见 §2.5 |
| 4090 容器 | 两条隧道进程（slot 0、slot 1） | 端口占着，下次起隧道会撞 |

清场（**按 PID 停，不要 `pkill -f`**）：

```bash
ssh modelarts_nb 'ss -ltnp | grep -E "5240|5250"'
```

拿到 PID 逐个 `kill`。每个服务端有一个 CANN 派生的子进程，父进程退出时跟着走。4090 那边用
`ps -eo pid,cmd | grep start_tunnel` 找 PID 再停。

---

## 1. 阻塞级：这台机器上还不能报任何数

### 1.1 金丝雀没跑完（1 / 4 集）

唯一的产出是一集（`npu_canary_seed42_run1/workers/shard_00/progress.json`，就一行）：

```
zsNo4HB9uLZ / ep 1 — success 1.0, SPL 1.0000, NE 0.5101, steps 50,
vlm_calls 14, trajectory_calls 11, ppa_applied 7, warmup 4
```

第二集死在里程计的一次 RPC 上（根因 §3）。**所以 910B 上没有 SR/SPL/NE，一个都没有。**

### 1.2 同机两遍逐调用一致，没验证

预注册判据里最严格的一条：同机、同 4 集、跑两遍，用
`scripts/exp19/select_cases.parse_client_log` + `scripts/exp19/build_records.compare_calls`
比到每一次调用。`--rng-seed` 是为它准备的，**但"准备好了"不等于"过了"**。

这条在昇腾上尤其要做，因为有一个 CUDA 上不存在的隐患：`torch.npu.initial_seed()`
**在不同进程里不一样**（实测 4278829994501441 vs 547798375190254），而 CUDA 的默认生成器
种子是常量。DA3 的前向有 3 处没播种的设备随机（`utils/alignment.py` 的 `randperm`、
`model/da3.py` 两处 `randint`），全靠显式的逐集重播种兜住；那段代码写了，没被验证过。

做这条之前要知道两件事：

- ⚠️ **没有任何地方要求确定性算子。** 服务端没设 `torch.use_deterministic_algorithms`，
  昇腾上也没设对应的 torch_npu 开关。播种只决定"抽到哪些随机数"，不保证归约顺序一致。
  要是两遍不一致，先查这个，别先怀疑播种。
- ✅ **做这一对的时候别开重启保护。** 客户端日志现在是 `>>` 追加的（`run_shard_once`
  第 399 行），重启会把第二段跑进同一个文件；`parse_client_log` 本身能处理（后一个完整
  块覆盖前一个，残块计入 `incomplete`），但 `select_cases.self_check` 第 554 行
  **断言 `duplicate blocks == 0`**。一致性那两遍要干净跑。
- ⚠️ **里程计服务端必须由启动脚本起。** `--rng-seed` 是启动脚本加的（默认 0）；手工
  起一个服务端调试时这个 flag 就没了，DA3 的 `randperm`/`randint` 回到"每进程不同"的
  默认生成器，而客户端侧所有检查照样通过。

### 1.3 延迟数字一个都不许进台账

计时中立性（打开计时不改变动作）只在 CUDA 上验证过，昇腾上要**重做一次**。
`deploy_ascend_910b.md` §8 那张分阶段表是诊断用的（带计时，比真实部署慢），不可引用。

---

## 2. 下一次开跑前必须处理的五处（按危害排序）

### 2.1 ✅ 空卡检查是死的：每张卡都读成 0 MiB

`run_ppa_servers_npu.sh` 的 `npu_used_mib()`（第 174 行）在**真机上对所有卡都返回 0**，
包括现在实际占了 65.5 GB 的卡 0 和卡 1：

```
card 0 -> 0 MiB (启动脚本读到的)      而真实是 65521 / 65536
card 1 -> 0 MiB                       而真实是 65329 / 65536
card 2 -> 0 MiB                       而真实是 3401  / 65536
```

原因：npu-smi 的芯片行里，**最后一个 `|` 段里有两对 `used / total`**——
`Memory-Usage` 的 `0 / 0` 在前，HBM 的 `65521/ 65536` 在后：

```
| 0                         | 0000:C1:00.0  | 0           0    / 0          65521/ 65536         |
```

`split(field[i], pair, "/")` 按**第一个**斜杠切，于是 `pair[1]` 是 `" 0   0  "`，去掉非数字
变成 `00` → 打印 0。6d6f880 那次修的是"把整行数字连起来"，没修这个。

于是 `(( used <= MAX_USED_MIB ))` 永远通过，**"只拿空卡"这个保护等于不存在**：它会把一槽
起到一张满卡上（结果是跑到一半 OOM，或者更糟——和别人的任务挤同一张卡）。

修法：在那个 `|` 段里取**最后**一对，而不是第一对（`while (match(...))` 推到末尾再 split）。

### 2.2 ✅ 重启 + 每片集数上限 = 悄悄多跑集

上次金丝雀是这么跑的：

```bash
export PPA_EVAL_SHARDS=0,1
export PPA_EVAL_MAX_EPISODES_PER_SHARD=2    # 2 片 × 2 集 = 预注册的 4 集
```

集数上限是**对"新跑的集"计数**的：

```python
# scripts/evaluation/r2r_val_unseen.py:1322
pending = sum(1 for key in target_list if key not in done)
return min(pending, max(args.max_episodes, 0))
```

而 `--max_episodes` 每次重启都原样再传一遍（`run_shard_once` 第 373 行）。所以**客户端死一次、
重启一次，这片就再跑 2 集新的**：开了 `PPA_EVAL_SHARD_RETRIES=2`，"4 集金丝雀"最坏变成
12 集，`progress.json` 里还看不出哪几集是补的。跟 4090 的 4 集金丝雀就不是同一个样本，
**判据作废**。

怎么处理（选一个，推荐前者，不动认证过的脚本）：

- 把那 4 集写成一个**只含 4 集的 `--episode_list`**，集数上限留空——`pending` 自然不超过 4；
- 或改 `run_shard`：重启前把上限减去已完成的集数。

### 2.3 ⚠️ 外部模式分不清"远端昇腾"和"本机 CUDA"

外部模式只核对 `servers.json` 里的端口和 `bridge_off`
（`run_ppa_r2r_val_unseen_cuda.sh:239–276`），**不核对那个端口后面是谁**。而远端槽位用的是
`52400+k / 52500+k`，**跟 4090 本机认证路径的默认端口是同一组**。

后果：4090 上只要有一套本机 CUDA 服务端占着 52400/52500（金丝雀 `canary_cuda_seed42`
就是用这组端口跑的），"昇腾评测"会**悄悄跑在本机 4090 上**，产出被标成昇腾的结果。

好消息是**事后能分辨**：`progress.json` 里的 `rpc_model_version` 在昇腾上是
`ppa-stage2-online-amb3r:zhr_1`，在 4090 上是 `...:workspace`（上一次金丝雀的两份记录正好
能对上）。所以修法很便宜：外部模式启动时拉一次 `get_server_info`，把这个串和
`servers.json` 的记录比一下。

顺带，`servers.json` 的核对还有几个窟窿：

| 字段 | 现状 |
|---|---|
| `repo_commit` | 读不到 git 时写 `unknown`（`run_ppa_servers_npu.sh:356`），客户端不查 |
| `vo_rng_seed` | 写了，客户端不查 |
| `timing` | 写了，客户端不查 |

`timing` 这一项的后果值得单独说，因为 §1.3 要做的正是计时的事：`HEATMAPVLN_TIMING`
**是按进程生效的**，所以客户端开着计时、服务端没开，所有检查都过，而延迟汇总里
**只有客户端侧的阶段、服务端的阶段全部静悄悄缺失**，看上去却是一份完整的汇总。

### 2.4 ✅ "端口还开着"不等于"服务端还活着"

外部模式判活只有一句 TCP 连接：

```bash
# run_ppa_r2r_val_unseen_cuda.sh:123
tcp_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
```

重启等待循环（第 418–421 行）也只等这个。而**昇腾的设备错误不杀进程**——现在就有活证据，
就是 03:11:30 吃了 AICPU 超时的那个里程计服务端：

```
PID 201838   已运行 1:01:54   仍在 LISTEN 127.0.0.1:52500
日志最后一行写于 03:11:35，之后 35 分钟一个字没写
```

所以重启后的客户端会连上一个"端口开着、里面可能已经坏了"的服务端，在几分钟里把重试次数
烧光；NPU 启动脚本的"服务端死了就一起死"也不触发，因为进程没死。TCP 连通还有第二层问题：
**连上的其实是 ssh 转发器**，隧道活着而远端服务端死了，这个检查照样过。

修法：重启等待里加一次真探活（`get_server_info` 这类廉价 RPC），探不通就按服务端故障处理
（停掉重起），而不是只看端口。**在这之前**：重跑金丝雀时人盯着里程计日志的 mtime，它停止
增长就是这个情况。

> ❓ **一个一直没答的问题：吃过 AICPU 超时的服务端还能不能继续服务？** 昇腾上设备侧异常
> 之后流常常是脏的。没测过。如果不能，§2.4 的修法就必须是"重起服务端"而不是"重连客户端"。
> 上面那个进程还在，可以直接拿它测。

### 2.5 ✅ TMPDIR 按槽号命名，同槽号的两次运行会撞

```bash
# run_ppa_servers_npu.sh:264
model_tmp="$TMP_ROOT/m${slot}"
vo_tmp="$TMP_ROOT/v${slot}"
```

只看槽号，不看运行时目录。于是**两次"槽 0"的运行（哪怕在不同卡、不同端口）共用一个
TMPDIR**，而 CANN 的核函数仓和 `multiprocessing.Manager` 的 AF_UNIX 套接字就住在那里。
已经撞过，证据是第一轮计时诊断退出时的报错：

```
OSError: [Errno 39] Directory not empty: '/home/ma-user/work/zhr/zhr_1/tmp/m0/pymp-vitae_72'
```

另有两个泄漏：那 12 个 `pymp-*` 没人清；`_config_for_device` 每起一次里程计服务端就在
TMPDIR 里留一个 `amb3r_vo_cfg_*`（现在 3 个）。都在 NFS 上，而且因为 AF_UNIX 的 108 字节
限制这个路径不能变长——跑一轮全量会攒出成百个。

修法：目录名带上运行标识（`m0a`/`m0b` 或单字符运行序号），**并且**启动时清掉上一轮残留；
清理要保留那条长度断言。

---

## 3. 那次 AICPU 超时：时间线对得很准，但没定论

栈底在 DA3 的 RoPE 里，是一次设备到主机的同步：

```
depth_anything_3/model/dinov2/layers/rope.py:183
    max_position = int(positions.max()) + 1
RuntimeError: ... AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507017
AI_CPU_Timeout(E30008): AI CPU operator execution time out.
```

时间线（北京 / 美东）：

| 时刻 | 事件 |
|---|---|
| 02:53:47 / 14:53:47 | 金丝雀用的那套服务端在卡 0 起来 |
| 02:56 / 14:56 | 金丝雀接着跑 |
| **03:09:35 / 15:09:35** | **我在另一张卡上起了第二套服务端**（计时诊断），03:09:57–03:10:51 在装模型 |
| **03:11:30 / 15:11:30** | **卡 0 的里程计吃了 AICPU 超时** |
| 03:23:52 / 15:23:52 | 第二套停掉 |

在这之前别的服务端都已经停了（各运行时目录的 `STOPPED` 标记最晚 02:47:47），所以当时全机
只有这一套在服务，唯一的另一件事就是第二套的启动——从 NFS 装 16 GB 权重 + CANN 编核，
主机侧很重（空载 load average 就有 10.4）。

❓ **这是很紧的相关，不是因果。** 另一种可能是 DA3 这条路径上的偶发故障，与并发无关。两种
解释导向不同做法，所以**不要把"别同时跑别的"当成已经解决**。

一个安全的缓解（只延长等待，不改数值）：`ACL_STREAM_TIMEOUT` / `ACL_DEVICE_SYNC_TIMEOUT`
（两个环境变量都在 `libtorch_npu.so` 里，对应失败的那个
`AclrtSynchronizeStreamWithTimeout`）。没试过。

---

## 4. ✅ 跨平台：910B 的数和 4090 的数不能混（量级拿到了）

同权重、同种子、同一集，两台机器：

| | 4090（`canary_cuda_seed42`） | 910B（`npu_canary_seed42_run1`） |
|---|---|---|
| success / SPL | 1.0 / 0.9951 | 1.0 / 1.0000 |
| NE | **0.2665** | **0.5101** |
| steps | **49** | **50** |
| vlm_calls / trajectory_calls | 14 / 11 | 14 / 11 |
| ppa_applied / warmup | 7 / 4 | 7 / 4 |

**模型调用次数一模一样，轨迹却不一样**：多走一步，终点差 24 cm。说明前面大部分决策一致、
后段某处岔开——这正是 bf16 算子和求和顺序不同该有的样子，不是 bug。

由此两条硬约束：

1. **昇腾上的任何对比臂只能和昇腾上的 A0 比。** 台账 §4 已经预埋了这句话
   （`docs/experiments/README.md:2423`），但**昇腾的 A0 还不存在**——这是把 EXP-21 搬过来的
   前置成本，一次全量约 22 小时。
2. 不要把两台机器的集级数字放进同一张表、同一个均值、同一次配对检验。

---

## 5. 显存：比我之前说的更紧

`deploy_ascend_910b.md` §8 已经改成负载下的数（整卡 62.2 / 65.5 GB）。还要再紧一档：

✅ **跑完之后空载，卡 0 仍然是 65521 / 65536 MiB。** 缓存分配器不归还，所以"一张卡一槽"
不是建议而是上限——那张卡在服务端进程退出之前连 15 MiB 都不剩。

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`：这个旋钮**确实存在**
  （`libtorch_npu.so` 里有，`torch_npu/npu/memory.py:580` 也提到），**但没试过**。真实占用
  （两个服务端 `peak_allocated` 合计约 40 GB）比保留量小得多，碎片有机会收回。
- 兜底仍是 `PPA_NPU_VO_DEVICES` 把里程计分到另一张卡，8 卡变 4 槽。
- **8 槽从没压过。** 唯一一次两套并发就撞上了 §3。
- ⚠️ 另外，`deploy_ascend_910b.md` §4 还写着 64 GB 一张卡"很宽裕"，与 §8 的实测矛盾，
  而启动脚本也**不阻止**你往同一张卡上叠槽（§2.1 的保护是死的，更拦不住）。

---

## 6. 测试：20 过 1 挂，挂的是测试自己；另有一条会打挂 CUDA 机器的基线

✅ 本地跑的（Mac，`--noconftest`，因为 `tests/conftest.py` 要 torch）：

```
tests/test_ascend_npu_port.py  →  20 passed, 1 failed
```

挂的是 `test_model_server_npu_requires_a_real_npu`：

```
NameError: name '_disable_fused_mha_fastpath' is not defined
```

**不是产品代码的问题**：`_load_resolve_device()` 只把 `_resolve_device` 一个函数的源码
`exec` 到塞了假 torch 的命名空间里，而 36ac0aa 之后 `_resolve_device` 在 npu 分支里会调
`_disable_fused_mha_fastpath()` 和 `_warm_up_npu()`，命名空间里没这两个名字。**这条测试
现在等于不存在**，要补（把两个 helper 也放进命名空间，或者一起 exec，顺便测到它们）。

⚠️ 还有一条更麻烦：`tests/test_latency_timing.py::test_npu_is_an_accelerator_like_cuda`
**在没装 torch_npu 的机器上必挂**——它 `monkeypatch` 了 `torch.npu` 和 `sys.modules`，但
`torch.device("npu:1")` 本身在原版 torch 上就会抛（设备名要靠 torch_npu 注册）。也就是说
**这条是我给 4090 和 C500 的基线引入的新失败**，跑全量之前得先给它加 skip 条件。
（在装了 torch 的环境里实测两个文件合计 36 passed / 2 failed：就是上面这两条。）

这套测试还有几处"看着在守、其实没守住"：

| 断言 | 实际守住的范围 |
|---|---|
| 两个服务端的 flag "除 `--device`/`--rng-seed` 外一致" | **只比 flag 名，不比值**，也不看 `MODEL_EXTRA` 里的 flag |
| 外部模式 `servers.json` 匹配 | 只核端口和 `bridge_off`（见 §2.3） |
| "没有无条件的 `torch.cuda.*`" | 只扫那两个服务端文件，只认字面 `torch.cuda.<attr>` 形式 |
| AMB3R 补丁 | 只读 `.patch` 文件的内容，**不检查补丁有没有打到真的树上**（见 §7.2） |
| 启动脚本的"分块注意力"证据行 | **自己证自己，永远不会失败**（见 §7.1） |

另外：`test_latency_timing.py` 在 **protobuf 升到 6.33.6 之后一次都没跑过**；
**全量套件在任何机器上都没跑过**（C500 已停用，只能在 4090 或 910B 上跑）。

---

## 7. 第三方树和换新实例：几个"看着在守、其实没守"的地方

### 7.1 ✅ 那条"分块注意力"的证据检查永远不会失败

启动脚本的四条证据行里有三条是真的，第四条是**自己证自己**：

```bash
run_ppa_servers_npu.sh:155   export DA3_SDPA_QUERY_CHUNK_SIZE=256
rpc_amb3r_vo_server.py:447   chunk = os.environ.get("DA3_SDPA_QUERY_CHUNK_SIZE", "")   # 原样回显
run_ppa_servers_npu.sh:338   grep -F "DA3_SDPA_QUERY_CHUNK_SIZE=256" ... || die "VO slot $slot did not take the chunked-SDPA path"
```

启动脚本导出一个环境变量，服务端把它原样打回日志，启动脚本再 grep 自己导出的东西。
**不管 DA3 拿这个变量做了什么（甚至完全没读），这个 die 都不可能触发。** 而
`deploy_ascend_910b.md` §4 的表里写的是"走的是认证过的分块注意力路径"——那行日志**证明不了
这件事**。同一行里的 `bf16=True` 也一样：它是 `is_bf16_supported()`，说的是"这张卡支持
bf16"，不是"这次前向用的是 bf16"。

修法：让 DA3 自己把实际用的 chunk 大小和 autocast dtype 打出来，grep 那一行。

### 7.2 ⚠️ AMB3R 的两处补丁没有任何自动校验

`scripts/ascend/amb3r_npu.patch` 跟着仓库走，但 `bootstrap_instance.sh` 只检查文件**存在**
（第 65–69 行，`slam_config.yaml`、`model.safetensors` 之类），**一份干净的 clone 和打过
补丁的树在所有检查下表现完全一样**；启动脚本的证据行也不覆盖它（§7.1 那条尤其不覆盖）。

偏偏这两处是**不打也不报错**的类型：

- `autocast(device_type='cuda')` 在没有 CUDA 的机器上只警告一句然后 `enabled=False`，
  于是整个 mapping 前向**跑成 fp32**——位姿与认证参照不同，激活显存还要翻一倍，
  而卡本来就只剩 3 GB；
- `torch.cuda.is_bf16_supported()` 返回 False，DA3 就**悄悄降到 fp16**。

两种都不会在日志里留下任何痕迹。**这是整个移植里最该加一条校验的地方**，而且很便宜：
比对 AMB3R 树的 tree hash（`09f1b2faff3b3be71cf2dc775afa26041356d296`），或者
`git apply --reverse --check scripts/ascend/amb3r_npu.patch`。

### 7.3 其余
- ⚠️ **隧道那一半只存在于 4090 容器里。** `start_tunnel.sh` / `start_tunnel_slots.sh` /
  `known_hosts` 都在 `/workspace/ppa_tunnel/`，**仓库里一个字都没有**，而 §10 写的是"只有
  `~/.ssh/authorized_keys` 跟着旧实例消失"。容器一换、机器一换，这半边要重写。
  私钥当然不能进仓库，**脚本应该进**。
- ⚠️ `bootstrap_instance.sh` 重写 `authorized_keys` 的方式是按**注释子串** `grep -vF`
  删行再追加（第 124–147 行）：注释串要是撞上别的 key 的任何部分，会把别人的 key 一起删掉；
  而且 `mv` 和追加之间有一个窗口，那期间这把钥匙是不被授权的。
- **`bootstrap_instance.sh` 没在真正的新实例上跑过。** 现在这台是手工装起来的，脚本按
  "应该这样"写的，不是按"这样试过"写的。第一次换实例要留返工时间。
- 两个依赖冲突（`plyfile` 要 numpy≥2、`mindstudio-probe` 要 protobuf≤3.20.2）都不在服务
  路径上，但**已经被坑过一次**：一次孤儿 pip 把 numpy 升到 2.4.6，cv2 的 ABI 就断了，症状
  不是 import 失败，而是 CANN 的核函数仓在第一个真请求上死。启动预检现在会查，但手动装
  任何东西之后都要再看一眼 `numpy.__version__`。
- CANN 来自镜像、环境在共享盘上。换镜像版本时 bootstrap 只 **WARN** 不拦。

---

## 8. 速度：剩下的 4 倍，和两笔"能换速度但要重新认证"的账

不打开计时的真实速度约 **11.2 秒一次调用**，对 4090 的 5.26 秒是 2.1 倍；按 8 槽算一次全量
约 22 小时（4090 上拿 2 张空卡约 41 小时）。已拿到的优化是关掉融合注意力 fastpath（那条
路径没有 NPU 实现、会悄悄回落 CPU：单次前向 1.35 s vs 0.0037 s，**365 倍**）。

剩下的 4 倍在慢系统两轮解码（7.4 s）、`system1_condition_latents`（4.0 s）、历史头（2.9 s）。
910B3 的 bf16 稠密算力纸面上不低于 4090，所以**这是软件不是硅片**。但要拿到就得动昇腾专用
算子（`npu_fusion_attention`、静态 KV cache），**会改变数值，要单独重新认证**。

| 账 | 现状 | 代价 |
|---|---|---|
| 里程计注意力分块 | 为与认证路径一致保留 `DA3_SDPA_QUERY_CHUNK_SIZE=256` | 昇腾上不分块快 **7 倍**（0.034 s vs 0.258 s）且更省显存，但要重新验证 |
| `aten::linalg_inv_ex.inverse` | 仍回落 CPU（位姿对齐，量小） | 没量过，估计可忽略 |

---

## 9. 文档自相矛盾的地方，和台账

- `deploy_ascend_910b.md` §5 还写着"客户端对 RPC 失败零重试，隧道断一次就会杀掉整个分片"。
  b2ae710 之后有了 `PPA_EVAL_SHARD_RETRIES`（默认 0），**文档没更新**——那个能救一次 22 小时
  长跑的开关，现在哪儿都没写。
- §4 的"64 GB 很宽裕"与 §8 的实测矛盾（见 §5）。
- §4 说"启动脚本自己 grep 这几行"，四行里有一行并非如此。
- **`docs/experiments/README.md` 里没有这次移植的条目。** 昇腾上要出数就得按 CLAUDE.md §8
  先写"问题 / 假设 / 判据 / 设置"四段并提交——**判据必须在看到任何结果之前写死**；昇腾的
  A0 参照（§4）属于"设置"的一部分。
- 两条已推的 commit message 混进了中文字（`645369a` 的 `was验证`、`b2ae710` 的 `the死`）。
  内容没问题，是手误；已推的历史我没有改写。

---

## 10. 建议的顺序

1. 清场（§0），把卡 0、卡 1 放出来。
2. 修 §2.1（空卡检查）和 §2.2（集数上限），这两条**在下一次跑之前必须修**——一个让你起到
   满卡上，一个让你的金丝雀样本变脏。§2.3–2.5 和 §7.1、§7.2 可以同一批做掉；§7.2 那条
   校验（tree hash 或 `git apply --reverse --check`）是这批里最便宜、保护面最大的一条。
3. 修 §6 那两条测试（一条过时、一条会打挂 CUDA 机器），然后在 910B 上跑一次 `tests/` 全量，
   拿到这台机器的基线。
4. 重跑 4 集金丝雀：只含 4 集的 `episode_list`、不设集数上限、开重启保护、**机器上不跑别的**。
   拿到 4/4 再谈数。
5. 紧接着原样再跑一遍（这一对**不开**重启保护，见 §1.2），做逐调用一致性比对。这步过了，
   昇腾才算"可用"。
6. 再决定要不要把 EXP-21 搬过来——前提是先在昇腾上跑出自己的 A0（约 22 小时）。
