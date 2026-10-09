# 昇腾 910B 部署：现在还有什么问题

初稿 2026-10-10 04:40 北京；2026-10-10 05:4x 这一轮把能在代码里关掉的都关掉了，重写了一遍。
部署怎么跑见 [`deploy_ascend_910b.md`](deploy_ascend_910b.md)；这份只写**还差什么、哪里会坑你**。

一句话：**七处"不报错但给你错东西"的缺陷已经在代码里关掉了，机器也清空了；但这台机器上
一个数还是不能报**——剩下的全是必须在机器上跑一遍才能消掉的（金丝雀 4 集、同机两遍逐调用
一致、计时中立、多槽并发），加上一个仍然没有定论的根因（AICPU 超时）。

标记沿用上一稿：✅ 真机上跑出来确认过；⚠️ 读代码推出来、还没在真机上触发过；❓ 没有定论。

---

## 0. 这一轮做掉了什么

下表"现在"一栏里的 ✅ 只表示**代码里关掉了**。哪几条在**真机上**跑过，单独列在表后面——两者
不是一回事，这份文档上一稿就是在这里把话说大了。

| 原条目 | 原来的毛病 | 现在 |
|---|---|---|
| §2.1 空卡检查 | 每张卡都读成 0 MiB，"只拿空卡"等于不存在 | ✅ 解析器挪进 [`npu_hbm_used_mib.awk`](../../scripts/ascend/npu_hbm_used_mib.awk)，取 HBM 那一对；**认不出的表就不打印数**，启动脚本把表打出来后退出 |
| §2.2 重启 + 集数上限 | 悄悄多跑集，金丝雀样本变脏 | ✅ 带上限的运行拒绝重启、也拒绝续跑进已有结果的目录（`PPA_EVAL_ALLOW_CAPPED_RESUME=1` 才放行）；集表用 [`make_episode_lists_from_run.py`](../../scripts/tools/make_episode_lists_from_run.py) 钉死 |
| §2.3 分不清远端昇腾和本机 CUDA | 只核端口和 `bridge_off`，可能悄悄跑在 4090 上 | ✅ 每次运行发一个实例令牌，两个服务端在 `GetServerInfo` 里报，客户端**每次探活都要求它**；`servers.json` 升到 v2，commit / dirty / timing / 采样参数全部核对 |
| §2.4 端口开着≠服务端活着 | 重启等待只看 TCP，白烧重试次数 | ✅ 服务端自己判定设备不可用并转 `NOT_SERVING`；启动脚本在日志里发现标记就**只退掉那个槽**；客户端重启等待改成真 RPC 探活，探到"不是这套服务端"就放弃分片。⚠️ "坏掉的服务端能过所有判活"这半边是真机测出来的，但**自检那条路还没被真机上真正的 507017 触发过**：万一 `torch.npu.synchronize()` 在那种状态下不抛，就不会转 `NOT_SERVING`，所以金丝雀期间仍然要盯里程计日志的 mtime |
| §2.5 TMPDIR 按槽号命名 | 同槽号两次运行撞一个目录，还漏一堆残留 | ✅ 改成本地 `/tmp` 下 `mktemp -d` 的按运行目录，退出时由 trap 删掉（真机上验过：SIGTERM 停掉之后目录没了）；旧的 `$ROOT/tmp/{m0,v0}` 已手工清掉。⚠️ 被 SIGKILL（或 `timeout` 的二段杀）打断在 cleanup 中途时还是会留下目录——这次测的时候就留了 4 个，手工删了。有界且无害：`/tmp` 是本地盘，换实例就没了 |
| §7.1 分块注意力的证据自证自己 | 启动脚本 grep 自己导出的环境变量 | ✅ 换成 DA3 自己报的那行（见下面"还剩什么没证明"） |
| §7.2 AMB3R 补丁没有任何校验 | 干净 clone 和打过补丁的树所有检查下一模一样 | ✅ [`check_amb3r_patch.sh`](../../scripts/ascend/check_amb3r_patch.sh)，启动脚本和 bootstrap 都调；真机上已验证通过 |
| §7.3 隧道只存在于 4090 容器里 | 仓库里一个字都没有，换容器就得重写 | ✅ [`start_tunnel.sh`](../../scripts/ascend/start_tunnel.sh) 进仓库（私钥当然没进）；`authorized_keys` 不再按注释子串删行 |
| §9 文档自相矛盾 | 显存、证据、重试开关三处对不上 | ✅ 见 `deploy_ascend_910b.md` |
| §6 两条测试 | 一条等于不存在，一条会打挂 CUDA 机器的基线 | ✅ 都修了；`tests/test_ascend_npu_port.py` 21 passed |

机器也清场了：两套诊断服务端按 PID 停掉，**8 张卡全部回到约 3.4 GB 空载**（卡 0 之前钉在
65521/65536），`$ROOT/tmp` 下 12 个 `pymp-*` 和 3 个 `amb3r_vo_cfg_*` 一并删掉。4090 容器里
那两条隧道进程没动。

### 这些改动里，哪些在真机上真跑过

把工作树打到 910B 的 `/tmp/hv-test`（不碰共享盘上的部署 checkout），起了一次单槽服务端又停掉。
下面这些是**那次启动的真实输出**，不是推断：

| 跑过的 | 证据 |
|---|---|
| 空卡检查读到真值 | `[ppa-npu] npu=7 used=3403MiB free enough`（旧解析器在同一张卡上读 0） |
| AMB3R 补丁校验 | `[amb3r-patch] ok tree matches the certified base 09f1b2f outside the two patched files` + `carries the NPU patch` |
| 提交号在起服务端**之前**读 | `[ppa-npu] repo=/tmp/hv-test commit=80432fd83c62 dirty=0`；把 `.git` 删掉再跑，**同一秒**就拒绝了（之前这个检查在两个 16 GB 权重装完之后才做） |
| 实例令牌 + 在线探活 | 真服务端起来后 `slot=0 ... ready`——探活要求 `heatmapvln-instance:<令牌>/slot0` 出现在 `supported_formats` 里，它过了 |
| `servers.json` v2 | 写出来了，`device: npu`、`model_version: ppa-stage2-online-amb3r:zhr_1`（**从活着的服务端取的**，不是算出来的）、40 位 `repo_commit`、`repo_dirty: 0`、`server_instance` |
| DA3 自报的分块证据 | 真里程计服务端日志里：`DA3 attention: query_chunk=256 (parsed by DA3), memory_bounded=True, xformers_disabled=True` |
| TMPDIR 按运行建、trap 删 | `/tmp/ppa-m7.XXXXXX`、`/tmp/ppa-v7.XXXXXX` 建了；SIGTERM 之后没了；`STOPPED` 标记写了；卡 7 回到 3404 MiB |
| id / 阈值 / 排布的拒绝 | `invalid NPU id: '00'`、`PPA_NPU_MAX_USED_MIB must be ... without leading zeros`，都在起任何东西之前 |
| 集表工具 | 在 4090 上对真参照 `canary_cuda_seed42` 跑过：钉出 4 集（见 §9 第 3 步），`--expect-per-shard 3` 被拒绝且一个文件都不写 |

**没在真机上跑过的**：服务端自检转 `NOT_SERVING`（要一次真的 507017 才触发）、槽位退休、客户端
外部模式的那些拒绝（要起隧道 + 4090 客户端）、带上限运行的拒绝（只在 Mac 上用假树跑过）。

---

## 0.5 2026-10-10 06:1x–06:48 的金丝雀重跑：两个根因分开了

用钉死的集表、`SHARD_RETRIES=0`、机器上没有别的任务，重跑了两次（`npu_canary_seed42_run2`
在 `d0bd852`、`run3` 在 `ffbe22e`）。结果把**两个一直被混在一起的故障**分开了。

### ✅ 之前两次金丝雀死的根因不是 AICPU 超时，是 gRPC 保活策略

客户端日志里写得很清楚：

```
Received a GOAWAY with error code ENHANCE_YOUR_CALM and debug data equal to "too_many_pings"
Current keepalive time (before throttling): 30000ms
RuntimeError: AMB3R VO RPC returned no response for ingest_frame
```

而且**上一次会话那次失败（10-09 19:05:35）的日志里是同一句**。所以本文件上一稿 §1.1 把它
归给"根因 §3（AICPU 超时）"是**归错了**：那次 AICPU 超时（03:11:30）是另一套服务端上另一件事。

机制：`vla_rpc` 的客户端设 `grpc.keepalive_time_ms=30000`（每 30 秒一个 ping），而 gRPC 服务端
默认只容忍**一条没有数据流动的通道上、5 分钟内 2 个 ping**，超了就 GOAWAY 断连。昇腾上一次
规划调用约 11 秒，模型服务端在算的时候里程计那条通道是空的，客户端的保活 ping 正好撞上。
客户端对 RPC 失败零重试，一次 GOAWAY 带走整个分片。4090 上从没触发过（全量 1839 集跑完了），
所以它一路活到了昇腾起步阶段。

修在 `ffbe22e`：两个服务端允许空闲通道上的保活 ping、接受每 20 秒一个、不限 ping 数。纯传输层，
不改任何请求/响应/数值，所以没有用 flag 门控。重跑后 `too_many_pings` 出现 **0 次**。

**顺带得到一条数值中立的旁证**：`run2`（`d0bd852`）和 `run3`（`ffbe22e`）的第 1 集
**逐位相同**——success 1.0、SPL 0.9649、NE 0.1478、48 步、注入 7 次。两个 commit 只差这个
传输层选项，这正是它该有的样子。

### ✅ 修掉保活之后，撞上的才是真的 AICPU 超时——而且这次条件很干净

`run3` 的第 2 集死于：

```
StatusCode.INTERNAL ... AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507017
The aicpu execution times out ... stream_id:2, task_id:34378
```

条件：**真实 Habitat 帧、单进程、整台机器上没有别的任务**。所以 §2.2 原来那条"并发相关"的
假设**不再需要**了——不开并发、不喂噪声，它照样发生。

两条新线索，留给下一次排查：

- **它卡在同一个地方**：两次都是第 2 集的第一次建图前向（第 20 帧，`map_init_window=20`），
  第 1 集整集没事。这像是跨集残留的状态，不像随机抖动。
- **`task_id:34378` 和今天那次合成噪声的对照组（04:44:23）是同一个**，03:11:30 那次是 35086。
  同一个 task id 出现两次，指向建图图里某个具体算子（AI CPU 算子跑在设备的 CPU 核上，
  DA3 这条路上的候选是动态形状的 `nonzero`/`unique`/`randperm`/`randint` 之类）。

### ✅ §2.4 的修法在一次**真的** 507017 上验证通过了

这是上一稿列为"还没被真机触发过"的那条，现在触发了，整条链路按设计走完：

| 时刻 | 发生了什么 |
|---|---|
| 06:48:17 | 里程计服务端自检：`NPU device unusable after a failed request; HealthCheck now reports NOT_SERVING (synchronize: RuntimeError: ... AclrtSynchronizeDeviceWithTimeout, error code is 507017)` |
| 06:48:57 | 启动脚本发现标记：`slot 0: the vo server reports its NPU unusable; retiring the slot`，写 `RETIRED`（`slot=0 role=vo 2026-10-10T06:48:57+0800`）|
| 之后 | 只有这一个槽，于是 `every slot has been retired` → 整体退出、`STOPPED` 写好、进程全清、卡 0 回到 3401 MiB |

**顺带答了上一稿自己提的问题**：`torch.npu.synchronize()` 在那个状态下**会抛**，所以
`NOT_SERVING` 这条路不是空的。修之前的同样情形会是：客户端死在那次 RPC 上，服务端继续
LISTEN、设备已坏，开了重启保护的话客户端再把重试次数烧在一台永远答不上来的服务端上。

### 所以金丝雀现在卡在哪

**1 / 4 集**，卡在 AICPU 超时上。保活那条已经不是障碍了；在 AICPU 这条解决之前，
H1（同机两遍逐调用一致）**没法测**——第 2 集过不去。

---

## 1. 阻塞级：这台机器上还不能报任何数

这一节和上一稿一样，一条都没少——它们**只能靠在机器上跑**来消掉。

### 1.1 金丝雀没跑完（1 / 4 集）

唯一的产出是一集（`npu_canary_seed42_run1/workers/shard_00/progress.json`，就一行）：

```
zsNo4HB9uLZ / ep 1 — success 1.0, SPL 1.0000, NE 0.5101, steps 50,
vlm_calls 14, trajectory_calls 11, ppa_applied 7, warmup 4
```

第二集死在里程计的一次 RPC 上——**那次的根因是 gRPC 保活策略，不是 AICPU 超时，见 §0.5**，本文件上一稿在这里归错了。**910B 上没有 SR/SPL/NE，一个都没有。**
这一集是开机观测，不是测量结果；台账里按这个口径记（EXP-22）。

重跑时的要求已经写进代码了：集表钉死（不设 `PPA_EVAL_MAX_EPISODES_PER_SHARD`）、
`PPA_EVAL_SHARD_RETRIES=0`、机器上不跑别的。前两条现在不照做就会被拒绝，不再只是嘱咐。

### 1.2 同机两遍逐调用一致，没验证

预注册判据里最严格的一条：同机、同 4 集、跑两遍，用
`scripts/exp19/select_cases.parse_client_log` + `scripts/exp19/build_records.compare_calls`
比到每一次调用。`--rng-seed` 是为它准备的，**"准备好了"不等于"过了"**。

为什么昇腾上非做不可：`torch.npu.initial_seed()` **在不同进程里不一样**（实测
4278829994501441 vs 547798375190254），而 CUDA 的默认生成器种子是常量。DA3 的前向有 3 处
没播种的设备随机（`utils/alignment.py` 的 `randperm`、`model/da3.py` 两处 `randint`），全靠
显式的逐集重播种兜住；那段代码写了，没被验证过。

做之前要知道的三件事（两条沿用上一稿，第三条已变成硬约束）：

- ⚠️ **没有任何地方要求确定性算子。** 服务端没设 `torch.use_deterministic_algorithms`，昇腾上
  也没设对应的 torch_npu 开关。播种只决定"抽到哪些随机数"，不保证归约顺序一致。两遍不一致
  时先查这个，别先怀疑播种。
- ⚠️ **这一对不要开重启保护，而且这条要靠人记住。** 客户端日志是 `>>` 追加的，重启会把第二段
  写进同一个文件；`select_cases.self_check` 断言 `duplicate blocks == 0`。启动脚本只在**设了集数
  上限**时拒绝 `SHARD_RETRIES>0`；用钉死的集表跑（这一对的做法）时它**不拦**，所以
  `PPA_EVAL_SHARD_RETRIES=0` 得自己写上。
- ⚠️ **里程计服务端必须由启动脚本起。** `--rng-seed` 是启动脚本加的（默认 0）；手工起一个
  服务端调试时这个 flag 就没了，DA3 回到"每进程不同"的默认生成器。现在服务端**每次启动都
  打印自己的种子**（`VO device RNG seed per reset_episode: ...`），而且手工起的服务端没有实例
  令牌、客户端外部模式会拒绝它，所以这个坑现在有两道提示。

### 1.3 延迟数字一个都不许进台账

计时中立性（打开计时不改变动作）只在 CUDA 上验证过，昇腾上要**重做一次**。
`deploy_ascend_910b.md` §8 那张分阶段表是诊断用的（带计时，比真实部署慢），不可引用。

这一轮加了一个相关的防线：计时是**按进程**生效的，以前客户端开着计时、服务端没开，所有检查
都过，而汇总里只有客户端侧的阶段。现在 `servers.json` 的 `timing` 会被核对，
`summarize_latency.py` 在服务端阶段缺失时打警告并返回 3，启动脚本把那条警告打出来。

### 1.4 多槽并发没压过

只做过单槽验证。唯一一次两套并发就撞上了 §3。placement 预算现在会拦住"一张卡塞一个模型
服务端 + 两个里程计服务端"这类排布，但**并发本身还是没压过**。

---

## 2. AICPU 超时：恢复性已有定论，触发条件仍然没有（§0.5 之后更新）

这一节有新证据，结论比上一稿更硬，也更让人不安。

### 2.1 ✅ "吃过 AICPU 超时的服务端还能不能继续服务？"——不能，而且它看起来好得很

上一稿把这个列成 ❓ 并建议"重启等待里加一次 `get_server_info` 探活"。**那个建议是错的**，因为
这次拿那个还活着的进程（PID 201838，03:11:30 吃的超时）直接测了：

| 调用 | 结果 |
|---|---|
| `connect` / `HealthCheck` / `GetServerInfo` | 全部正常 |
| `reset_episode` | 正常 |
| `ingest_frame` × 19 | 全部正常 |
| 第 20 帧（`map_init_window=20`，第一次真跑 DA3 建图前向） | **立刻失败**：`ACL stream synchronize failed, error code:507017` |

也就是说：**当时部署里所有的判活手段——TCP 连通、HealthCheck、GetServerInfo，甚至整整 19 次
ingest——在一台已经不能算数的机器上全部通过**。所以修法不能是"客户端探活"，必须是
"服务端自己说自己坏了，然后重起"。现在：

- 服务端在一次请求失败后用 `torch.npu.synchronize()` 探自己的设备，探不通就记下来、打标记
  `NPU device unusable after a failed request`，并从此 `HealthCheck` 返回 `NOT_SERVING`；
- 启动脚本的守护循环看见标记就**只退掉那个槽**（停掉那一对、放出那张卡、写进 `RETIRED`），
  其余槽继续服务自己的分片，全退完才整体退出；
- 客户端外部模式看见 `STOPPED` / `RETIRED` / 日志里的标记就拒绝开跑；重启等待改成真 RPC 探活。

### 2.2 ⚠️ 而且一台健康的服务端也会吃到同样的超时

做上面那条的对照组时（卡 1 上那套计时诊断服务端，从没报错过，同样 21 帧**合成随机噪声**图），
它在第 20 帧上也进了同一个坑：日志里 **04:44:23 一条 `AI_CPU_Timeout(E30008)` / 507017**，
而请求是 04:35 左右发出去的——**CANN 自己的超时等了约 9 分钟（约 540 s）才触发**。这和客户端
默认的 600 s RPC 期限是同一个量级，略早于它：所以正常配置下客户端大概会先收到服务端自己报的
错，而不是先超时（我诊断时把期限设成 120 秒，才先拿到 `DEADLINE_EXCEEDED`，那时服务端还在算）。
之后这台也一样，每个请求立刻 507017。

对 §3 原来那条"并发相关"假设的影响，要说得很清楚：

- 这次是**单进程、同机没有别的任务**，照样复现了 507017。所以"别同时跑别的"**不能**算作
  已经解决，上一稿的那条时间线相关性现在更弱了；
- 但我喂的是**合成噪声图，不是 Habitat 的真实帧**。噪声上 DA3 的几何可能进了退化分支，这和
  03:11:30 那次（真实帧）**不一定是同一个原因**。所以：既没证明是并发，也没证明不是。

❓ **仍然没有定论的是根因**：建图前向在这个平台上会挂死在某个 AI CPU 算子上，触发条件不明。
已知的可缓解方向（都没试）：`ACL_STREAM_TIMEOUT` / `ACL_DEVICE_SYNC_TIMEOUT`（只延长等待）、
`OpExecuteTimeOut`。下一次真机排查建议从"真实帧能不能复现"开始，而不是从并发开始。

实践含义：一次 22 小时的全量跑里，任何一个槽吃到这个，那个槽就报废到重起为止。现在这件事
至少是**可见**的（标记 + 槽位退休 + 客户端拒绝），而不是把重试次数烧光然后得到一份看着完整
的结果。

---

## 3. ✅ 跨平台：910B 的数和 4090 的数不能混（量级拿到了）

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

由此两条硬约束（已写进台账 EXP-22 的"边界"）：

1. **昇腾上的任何对比臂只能和昇腾上的 A0 比。** 而**昇腾的 A0 还不存在**——一次全量约 22 小时。
2. 不要把两台机器的集级数字放进同一张表、同一个均值、同一次配对检验。

---

## 4. 显存与排布

✅ **跑完之后空载，卡 0 仍然是 65521 / 65536 MiB**（这一轮停进程之后才回到 3401）。缓存分配器
不归还，所以"一张卡一槽"不是建议而是上限。

现在启动脚本按 §8 的实测值拦排布：一张卡上

- 两个模型服务端 → 拒绝（原来就有）；
- 一个模型服务端 + ≥2 个里程计服务端 → 拒绝（约 41.8 + 2×17.9 + 3.3 > 65.5 GB）；
- ≥4 个里程计服务端 → 拒绝；
- 2–3 个里程计服务端（这张卡上没有模型服务端）→ 放行，但打警告：**这个排布从没实测过**。

另外两条没变：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 这个旋钮**确实存在**，**没试过**。
- 兜底仍是 `PPA_NPU_VO_DEVICES` 把里程计分到另一张卡，8 卡变 4 槽。

---

## 5. 还剩什么没证明（证据的边界）

这一轮把"自证自己"的检查换成了真检查，但**不要把它当成比它更强的东西**：

| 现在有的证据 | 它证明了什么 | 它**没有**证明什么 |
|---|---|---|
| `DA3 attention: query_chunk=256 (parsed by DA3), memory_bounded=True, xformers_disabled=True` | DA3 自己的解析器看到的 chunk 是 256（真机实测：不设这个变量时它打 `query_chunk=0`，启动脚本的 grep 就会失败）；dinov2 的注意力层绑的确实是分块那个函数 | **某一次前向真的分了块**。`memory_bounded_scaled_dot_product_attention` 只在 `query_length > chunk_size` 时分块，否则直接走普通 SDPA |
| `check_amb3r_patch.sh` 通过 | **两处补丁的代码在树里**（缺了就退出 2）。**如果这棵树是 git checkout 且 `09f1b2f` 在里面**：两个被补文件之外的部分与它逐字节相同，不同就退出 2 | 前向用的 dtype。补丁让 fp16 这条路**报错**而不是静默，所以这条是"代码保证"，不是"日志证据"。另外**反向 apply 失败只是警告**（上下文漂移不拦），基线树不在或这棵树不是 checkout 时**也只是警告**——所以"通过"里可能只含第一行那一条 |
| `bf16=True`（设备行） | 这张卡支持 bf16 | 任何一次前向的 dtype |
| `NPU op toolchain ready (conv2d in bf16 ...)` | CANN 的算子工具链在启动时初始化过了 | 同上：那是手搓的热身算子 |

❗ 上一稿把 §7.2 的校验写成"比对 AMB3R 树的 tree hash `09f1b2f`"。**那条是反的**：
`09f1b2f` = `df74392^{tree}`，是**不含**昇腾补丁的那棵树（也就是 4090 部署的那棵）。打过补丁
的树是 `d89957c`。树哈希等于 `09f1b2f` 恰恰说明补丁**没**打上。所以 `check_amb3r_patch.sh` 用
的是另一种办法：补丁内容在不在 + 能不能反向 apply + **两个被补文件之外**是否仍等于 `09f1b2f`。

---

## 6. 测试

✅ `tests/test_ascend_npu_port.py` → **21 passed**（Mac，`--noconftest`）。两条坏的都修了：

- `test_model_server_npu_requires_a_real_npu` 以前抛 `NameError: _disable_fused_mha_fastpath`
  ——桩只 `exec` 了 `_resolve_device` 一个函数，而它现在会调两个 helper。现在桩带记录，顺便
  断言 CUDA 分支**不**调它们，并用 AST 钉住两个 helper 的签名。
- `test_latency_timing.py::test_npu_is_an_accelerator_like_cuda` 以前在没装 torch_npu 的机器上
  必挂（`torch.device("npu:1")` 本身就抛）。现在在测试里替掉 `torch.device` 工厂，生产代码
  `src/utils/latency.py` 一行没动。

新增的行为级测试（不是 grep 脚本字符串那种）：`tests/test_ascend_free_card_check.py`、
`tests/test_ascend_amb3r_patch_check.py`、`tests/test_external_server_guards.py`、
`tests/test_latency_summary_server_gap.py`、`tests/test_make_episode_lists_from_run.py`。

上一稿列的"看着在守、其实没守住"也收紧了两条：两个启动脚本现在比**flag 的值**而不只是名字
（还比 `MODEL_EXTRA` 的构造行），外部模式的 `servers.json` 核对从 2 个字段变成 9 个。

### ✅ 910B 上的全量基线（2026-10-10，这一轮实测）

把工作树打包到 `/tmp/hv-test`（不碰共享盘上的部署 checkout），用 `envs/ppa` 的 pytest 跑全量：

```
12 failed, 1760 passed, 35 skipped, 2 xfailed, 25 errors in 376s（6 分 16 秒）
```

**这 12 failed + 25 errors 一条都不是这一轮引入的**：把干净的 HEAD（3ebe58c）单独导到
`/tmp/hv-head` 只跑这几个文件，得到**完全一样的 12 failed / 25 errors**（CLAUDE.md §4 的判法）。
按根因分四类，全是平台/环境，不是代码：

| 类 | 数量 | 根因 |
|---|---|---|
| `test_trajectory_dagger.py` / `_dataset.py` | 25 errors | 测试里**写死了 C500 的工作区路径** `/mnt/afs/liwenhao/agent/370910109`，这台机器上不存在 → `PermissionError: /mnt/afs` |
| `test_heatmap_nextdit_control.py` | 4 failed | `NotImplementedError: Could not run 'npu::npu_rms_norm' with arguments from the 'CPU' backend`——装了 torch_npu 之后这条 CPU 路径就跑不了 |
| `test_exp19_figures_v2.py` | 7 failed | 中文文本宽度/布局断言（字体与 C500 不同） |
| `test_stage3_dataloader_order.py` | 1 error | 已知的 `_dataloader_in_order_kwargs` collection error，CLAUDE.md §4 写了不要顺手改 |

所以**这台机器自己的基线就是上面那一行**，不是 CLAUDE.md 里 C500 的 1004 / 1 / 1（套件这期间长大了
很多）。再跑的时候对照这个数；前三类要么是写死路径、要么是平台假设，修它们是独立的活。

---

## 7. 换新实例

- ⚠️ `bootstrap_instance.sh` **仍然没在真正的新实例上跑过**。现在它多做一件事（校验 AMB3R
  补丁），还是按"应该这样"写的。第一次换实例要留返工时间。
- ✅ `authorized_keys` 那段重写了：按**公钥本体**匹配而不是注释子串（旧写法会把注释里恰好
  含同一串的别人的 key 一起删掉，已在本地复现），并且先写临时文件再原子 `mv`，不再有"这把
  钥匙暂时没被授权"的窗口。
- ✅ 隧道脚本进了仓库：[`scripts/ascend/start_tunnel.sh`](../../scripts/ascend/start_tunnel.sh)，
  主机 / 端口 / 私钥路径全走环境变量。私钥仍然只在 4090 容器里。
- 两个依赖冲突（`plyfile` 要 numpy≥2、`mindstudio-probe` 要 protobuf≤3.20.2）都不在服务路径上，
  但已经被坑过一次：一次孤儿 pip 把 numpy 升到 2.4.6，cv2 的 ABI 就断了，症状不是 import 失败，
  而是 CANN 的核函数仓在第一个真请求上死。启动预检会查，手动装任何东西之后都要再看一眼
  `numpy.__version__`。
- CANN 来自镜像、环境在共享盘上。换镜像版本时 bootstrap 只 **WARN** 不拦。
- 这台实例上 `npu-smi info -m` 显示 **NPU ID == Chip Logic ID**（8 张卡都是恒等），所以启动脚本
  里的逻辑卡号和 `npu-smi` 的卡号是同一个东西。换了规格（比如只给 4 张卡的 flavour）要重新确认。

---

## 8. 速度：剩下的 4 倍，和两笔"能换速度但要重新认证"的账

不打开计时的真实速度约 **11.2 秒一次调用**，对 4090 的 5.26 秒是 2.1 倍；按 8 槽算一次全量
约 22 小时（4090 上拿 2 张空卡约 41 小时）。已拿到的优化是关掉融合注意力 fastpath（那条路径
没有 NPU 实现、会悄悄回落 CPU：单次前向 1.35 s vs 0.0037 s，**365 倍**）。

剩下的 4 倍在慢系统两轮解码（7.4 s）、`system1_condition_latents`（4.0 s）、历史头（2.9 s）。
910B3 的 bf16 稠密算力纸面上不低于 4090，所以**这是软件不是硅片**。但要拿到就得动昇腾专用
算子（`npu_fusion_attention`、静态 KV cache），**会改变数值，要单独重新认证**。

| 账 | 现状 | 代价 |
|---|---|---|
| 里程计注意力分块 | 为与认证路径一致保留 `DA3_SDPA_QUERY_CHUNK_SIZE=256` | 昇腾上不分块快 **7 倍**（0.034 s vs 0.258 s）且更省显存，但要重新验证 |
| `aten::linalg_inv_ex.inverse` | 仍回落 CPU（位姿对齐，量小） | 没量过，估计可忽略 |

---

## 9. 建议的顺序

1. ~~清场~~、~~修那七处~~、~~修两条测试~~ —— 这一轮做完了（§0）。
2. ~~在 910B 上跑一次 `tests/` 全量~~ —— 做了，基线在 §6（**1760 passed / 12 failed / 25 errors**，
   12+25 全是既有的平台问题，不是这一轮引入的）。再跑时对照那一行，不要对照 CLAUDE.md 里 C500 的
   1004 / 1 / 1。
3. 用 `make_episode_lists_from_run.py` 从 4090 的 `canary_cuda_seed42` 钉出 4 集的集表，重跑
   金丝雀：不设集数上限、`PPA_EVAL_SHARD_RETRIES=0`、机器上不跑别的。拿到 4/4 再谈数。

   ✅ 这一步已经在 4090 上实跑过一次（只生成集表，没跑评测）：参照运行的 runtime 戳是
   `20260928_212453_1550`，钉出来的就是这 4 集——

   | 片 | 集 |
   |---|---|
   | shard_00 | `zsNo4HB9uLZ` ep 1、ep 25 |
   | shard_01 | `zsNo4HB9uLZ` ep 2、ep 26 |

   （`--expect-per-shard 3` 这种对不上的参照会被拒绝，并且一个文件都不写。910B 上已跑完的那
   一集正是 shard_00 的 ep 1。）
4. 紧接着原样再跑一遍，做逐调用一致性比对（§1.2）。这步过了，昇腾才算"可用"。
5. 计时中立性重做一次（§1.3），之后延迟数字才能进台账。
6. 再决定要不要把 EXP-21 搬过来——前提是先在昇腾上跑出自己的 A0（约 22 小时）。
7. ~~AICPU 超时：从"真实帧能不能复现"开始查~~ —— 已复现（§0.5）：真实帧、单进程、机器上没别的任务，卡在**第 2 集的第一次建图前向**，`task_id:34378`。**第 3、4 步现在被它堵着**，下一步应当从这里开始：先看第 1 集到第 2 集之间 AMB3R 侧留下了什么状态，再看那个 AI CPU 算子是哪一个。
