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

2026-10-10 第二轮又加了三样（细节在 §0.5）：

| 加了什么 | 为什么 | 现在 |
|---|---|---|
| [`probe_plan_latency.py`](../../scripts/tools/probe_plan_latency.py) | 延迟只能在服务器本机量；之前每次都临时搭一个驱动，其中一个版本报出过一次比了 0 次调用的空 "identical" | ✅ 进仓库：`run` 按类别报延迟，`compare` 逐字段比两遍决定了什么、比到 0 次调用就拒绝给结论；`--fake` 用 `src/deploy/fake_servers.py` 空跑，不需要加速卡 |
| [`qwen2_5_vl_vision_count.py`](../../src/models/qwen2_5_vl_vision_count.py) | "视觉塔一次调用过几遍"之前全靠估，而所有优化优先级都挂在这个数上 | ✅ 服务端自己数，写进响应的 `vision_tower` 字段（不进 `timing_ms`，那是阶段→毫秒的表）；不读设备数据，所以自己不花同步 |
| `_window_mask`（在已有的 mask 补丁里） | `.tolist()` 之后剩下的是下发开销：每个窗口发一次切片赋值，32 层重复 | ✅ 改成比较窗口 id，四个 kernel 封顶；盖不满序列时退回原循环 |

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

### ✅ 加上 `TASK_QUEUE_ENABLE=0` 之后，金丝雀 4/4 跑完了（run4）

```
{"episodes": 4, "sr": 100.0, "spl": 83.48, "ne": 0.613, "os": 100.0, "ppa_applied_calls": 57, "status": "passed"}
ep 1  | SPL 0.9649  NE 0.1478  steps  48 | 注入 7
ep 25 | SPL 0.8916  NE 0.9095  steps 100 | 注入 17   ← run2/run3 都死在这一集
ep 2  | SPL 0.4825  NE 0.8050  steps 109 | 注入 18
ep 26 | SPL 1.0000  NE 0.5906  steps  81 | 注入 15
```

全程 `507017` **零次**，集表核对通过（少一集或多一集会让启动脚本失败）。

⚠️ **但这四个数还不能当结果用**，因为这一遍带着 `TASK_QUEUE_ENABLE=0`，而那个开关：

- **没有过 H1**。它改的是算子下发方式（torch_npu 把下发放到后台队列线程），理论上数值中性，
  旁证是 ep 1 在 `d0bd852`（无开关）和 `ffbe22e`（无开关）两遍逐位相同、run4（有开关）也是
  同一组数——但"ep 1 相同"只是一集，不是"同机两遍 4 集逐调用相同"那条判据。
- **还不是部署设置**。要用它就得写进启动脚本并记进 `servers.json`，否则下一次谁不设它，结果
  就又不是同一条路径。
- 它为什么能避开 AICPU 超时，**也还没有解释**。所以 §0.5 那条"卡在第 2 集第一次建图"的根因
  仍然没定，只是现在有了一个能绕过去的开关。

⚠️ **这个开关后来被量了，而且被撤掉了**（下面"这一轮落地的两项"）：它在三类调用上都比平台默认
的 `=1` 慢 7.0%–7.7%，数值中性有实测。**于是这一遍 4/4 是在一个已经不再使用的配置下跑出来的**，
这四个数作为"结果"的问题比上面写的更大一层：下一遍金丝雀换了配置，4/4 要重新拿。

### ✅ 单次调用的成本：按调用类别量清了，视觉塔占 83%，其中一半是重复的

**量法.** `scripts/tools/probe_plan_latency.py`（新增）在**服务器本机**驱动 NavAgent，不走隧道，
帧由种子生成（两臂看到同样的像素）。每遍 14 次规划调用 / 50 步；**每个调用类别各自丢掉第一次**，
因为第一次要付算子编译。`compare` 子命令逐字段比两遍决定了什么，并且**在比到 0 次规划调用时拒绝
给结论**（上一轮一个临时版本就是这么报出过一次空的 "identical"）。

**先把"一次调用"拆开——它不是一个数。** 三类调用差 5 倍：

| 调用类别 | 做了什么 | `model_rpc` 中位数 | 视觉塔过几遍 |
|---|---|---|---|
| `native_actions` | 只跑慢系统第一轮，直接给动作块 | **2349 ms** | 1 |
| `trajectory` | 第二轮 + 系统 1（还没有位姿） | **9107 ms** | 3 |
| `trajectory+ppa` | 第二轮 + 系统 1 + PPA 注入 | **11079 ms** | 4 |

同配置两遍的差 < 0.3%。里程计查询不在 `model_rpc` 里，它另外加在规划步上，而且**随地图增长**：
第 6 次调用 1.0 s → 第 13 次 3.1 s（50 步之内）。不含规划的步只要约 52 ms。

⚠️ **上一稿"稳态 4.41–5.03 s / 预热 2.77–2.84 s"退役。** 它不是上面三类里的任何一个，而且
现在已经无法归类——按算子次数反推，当时 profile 的那次调用**视觉塔只过了一遍**，属于最轻的那类。
延迟以后一律按类别引用，不要再引用单一的"稳态"数。

**服务端自己的阶段账（开计时，`trajectory+ppa`，n=5）.** 视觉塔的过数由服务端自己数出来并写进
响应（`vision_tower` 字段，`src/models/qwen2_5_vl_vision_count.py`），不再靠估：

| 阶段 | ms | 占比 | 这一阶段里的视觉塔形状 |
|---|---|---|---|
| `system2_turn2_generate` | 3318 | 30.1% | `8620x10` |
| `system1_condition_latents` | 2907 | 26.3% | `8620x10` ← **和上一行同一份输入** |
| `system2_turn1_generate` | 1946 | 17.6% | `7056x9` |
| `ppa_history_memory` | 1935 | 17.5% | `7056x9` ← **和上一行同一份输入** |
| `system1_nextdit_sampling` | 403 | 3.6% | — |
| 两轮 prep + 请求解码 | 512 | 4.6% | — |
| `ppa_bridge` / `future_heatmap_diagnostics` / `trajectory_to_actions` | 各 ≤ 4 | ≈0% | — |

每次调用的形状序列是 `['7056x9', '8620x10', '7056x9', '8620x10']`，5 次调用完全一样。

**第 3、4 遍重复的是第 1、2 遍的同一组图像**——两条证据强度不同，分开说：
- **第 4 遍（`system1_condition_latents`）是同一个张量对象**：`generate_latents` 拿到的就是
  第二轮那个 `inputs["pixel_values"]`。这一条**按构造成立**，不需要实测。
- **第 3 遍（PPA 历史头）是同一组 PIL 重新过了一遍图像处理器**。形状对得上，代码路径也对得上，
  但"逐字节相同"还**没有实测**——它取决于图像处理器是确定性的（极可能，但没验）。
  真要用这一条做复用，必须在代码里用 `torch.equal` 实际比过，不能靠这个推断。

按阶段减去里面的语言模型前向，一遍 `7056x9` 约
1.8–1.9 s、一遍 `8620x10` 约 2.6–2.9 s，四遍合起来约 9.2 s —— **视觉塔占这一类调用的 83%，
其中约 4.6 s 是重复的。** 扩散采样（`num_inference_steps: 10`、`num_sample_trajs: 32`）
只占 3.6%，不是靶子。

### ✅ 这一轮落地的三项（都不改数值，都实测过）

| 改动 | `native_actions` | `trajectory` | `trajectory+ppa` | 数值证据 |
|---|---|---|---|---|
| 撤掉 `TASK_QUEUE_ENABLE=0` | −187 ms | −787 ms | −882 ms | 逐字段相同（4 对） |
| 视觉窗口 mask 向量化构造 | −129 ms | −356 ms | −541 ms | 逐字段相同 |
| `generate_latents` 复用前一遍视觉塔 | −14 ms | **−2758 ms** | **−2749 ms** | 逐字段相同 + 6/6 重算逐位相等 |
| **合计（对金丝雀那一遍的配置）** | **−331 ms (12.4%)** | **−3900 ms (38.1%)** | **−4172 ms (33.4%)** | |

绝对值：`native_actions` 2666 → **2335 ms**；`trajectory` 10250 → **6349 ms**；
`trajectory+ppa` 12502 → **8330 ms**。

**1) `TASK_QUEUE_ENABLE=0` 该撤.** 预注册的问题是"修了 1.6 万次同步之后这个开关还有收益吗"。
ABBA 两遍、两张卡上**分别新起**的两套服务端、逐类别比：

| 调用类别 | `=1` A / B | `=0` A / B | `=1` 快 |
|---|---|---|---|
| `native_actions` | 2478 / 2479 | 2665 / 2666 | **7.0%** |
| `trajectory` | 9460 / 9465 | 10301 / 10198 | **7.7%** |
| `trajectory+ppa` | 11626 / 11613 | 12513 / 12490 | **7.0%** |

开关确认是**逐进程**生效的（读 `/proc/<pid>/environ`，不是只写在启动壳里）。四对两两比较
全部 `IDENTICAL`（各 14 次调用 × 13 字段 + 50 步 × 5 字段）：同臂两遍两对、**跨臂**两对。

- **判据回答：落在 5%–10% 那一档 → 按预注册算"没测出来"**（那一档写的是"加大 N 再判"）。
  **但这个"没测出来"是判据带定错了，不是测量不够**：我把 5–10% 当统计不确定区间写的，而实测的
  不确定度（同臂两遍 < 0.3%，n=16/8/20）远在 5% 以下，加大 N 不会把 7% 推进任何一档。
  事后改判据没有约束力，所以如实记成"没测出来"。
- **决策（工程决策，不是假设检验结论）：撤掉。** 三条理由都不依赖那个带——它在三类调用上都更慢、
  `=1` 是平台默认值（少一个未认证开关）、数值中性有实测。启动脚本现在把它打进日志，
  设成 `0` 时警告。
- ⚠️ **边界：这不覆盖 AICPU 超时。** 这个开关当初就是为了绕开那次挂死才加的，而这次的探针
  从不重置 episode，根本没走到"第 2 集第一次建图"。**撤掉之后必须重跑 4 集金丝雀**，
  在那之前不能说挂死不会回来。

**2) 视觉窗口 mask 向量化构造**（`src/models/qwen2_5_vl_vision_mask.py` 的 `_window_mask`）。
上一轮把边界从"每个切片回读一次"改成"`.tolist()` 一次读完"之后，剩下的开销是**下发**：
那个循环每个窗口发一次切片赋值，每次只写几个字节却要约 40 µs 主机时间，32 层重复一遍。
改成比较"窗口 id"——`ids[:, None] == ids[None, :]`——不管多少窗口都是四个 kernel。
布尔图样完全相同，所以是**同一个张量**；窗口没盖满序列时（有位置不属于任何窗口，它对所有位置
包括自己都必须是 False，id 比较表达不出来）退回原来的循环。
和旧代码比：14 次调用 × 13 字段 + 50 步 **`VERDICT: IDENTICAL`**。

### ✅ 第三项：`generate_latents` 不再重跑视觉塔（4 遍 → 3 遍）

`system1_condition_latents`（2907 ms / 26.3%）里那一遍 `8620x10` 拿到的就是第二轮
`inputs["pixel_values"]` **同一个张量对象**。现在它由第二轮那一遍的输出直接应答。

**为什么只有这一遍能这么做.** 视觉塔是被 ViT 第 7/15/23/31 块上的 forward hook 看着的
（`native_single_view_feature_extractor.py` 在加载时注册），所以返回缓存张量就跳过了塔里
所有块、连带跳过那些捕获。**这一遍可以，别的不行**，理由是逐行查过的：`_vit_captures`
全仓库只有三处读（`native_single_view_feature_extractor.py:170`、`:174`、`:202`），
三处都从 `extract_from_pixels` 进去，而它先 `clear()`（`:146`）再跑塔，捕获没填就抛
"visual hooks did not fire"（`:170-172`）。所以读的人只会看到**自己那一遍**的捕获，
或者直接报错，**永远看不到被跳过那一遍的残值**。第 3 遍就是那个读的人，它从不被应答。

复用的范围由调用方画死，不靠推断：`record_scope()` 圈住慢系统两轮 generate，
`serve_scope()` 只圈住那一次 `generate_latents`。**scope 之外一律不应答**，
所以新调用方不会"顺手"捡到它，必须自己开口要。

**两道没那么显然的闸**（都是评审逼出来的）：
- **只靠 `torch.equal` 是空的。** 部署路径上第 4 遍收到的就是第 2 遍记下的同一批对象，
  比较等于自己和自己比。所以命中还要求记下的像素 / grid / 输出三者的 `_version` 没变，
  而且比较走整数视图——NaN 和自己相等、`-0.0` 和 `0.0` 不相等，**承诺的是位，不是浮点相等**。
- **adapter 检查要查"装了没有"，不是"有没有这个 API"。** 第一版查
  `hasattr(module, "disable_adapters")`，而 transformers 的 `PreTrainedModel` 无条件带这个方法
  ——于是它在**每一台**真服务端上都拒绝，把复用悄悄全关掉了，只有日志那一行露了馅。
  现在查 `_hf_peft_config_loaded`、非空 `peft_config`、以及被注入的层；
  那次误判本身有一条回归测试。

**证据.**
- **位相等是实测的，不是论证的**：`HEATMAPVLN_QWEN_VISION_REUSE_VERIFY=1` 让每次命中都
  **重算一遍**再 `torch.equal`。10 次调用里 6 次命中，**6 次 verified、0 次 mismatch**。
  这同时说明这台 NPU 上同输入的视觉塔是逐位确定的。
- **决策不变**：同驱动同种子，开/关复用各 14 次规划调用，13 个决定字段 + 50 步 × 5 字段
  **`VERDICT: IDENTICAL`**。
- 服务端自报的过数序列也照着设计走：`trajectory+ppa` 是
  `['record','record','unscoped','verified']`（第 3 遍 `unscoped`，确实没被应答），
  `trajectory` 是 `['record','record','verified']`。

⚠️ 这三项都**没有过 H1**，和别的部署设置一样。这一轮的证据是"同机、同驱动、合成帧、
14 次调用逐字段相同"，用的是 `probe_plan_latency.py compare`，**不是** 4 集金丝雀。

### ⚠️ 还剩的一块：第 3 遍（PPA 历史头）那 1.8–1.9 s，和注意力

第 4 遍已经做掉了（上一节）。**剩下第 3 遍：PPA 历史头那一遍 `7056x9`，约 1.8–1.9 s**，
只出现在 `trajectory+ppa` 这一类（部署里约 17% 的规划调用：金丝雀 57 次注入 / 338 步）。

它和第 1 遍是同一组图像，但**不能用同一个办法**：它的捕获正是被读的那一份，跳过塔就没有捕获，
`extract_from_pixels` 会直接抛。能做的两条路都更重：
- **调用方交接**：第 1 遍跑完之后把 `_vit_captures` 的引用和合并输出一起交给 PPA 头，
  让它别再跑塔。要改 `native_single_view_feature_extractor` 的契约，而且
  ⚠️ 那个捕获字典是**进程级**的（挂在唯一的 extractor 上），`--workers > 1` 时两个请求会互相踩，
  所以这条路必须同时要么加进程锁、要么在 `workers != 1` 时拒绝。默认是 1，但没有东西强制它。
- **把捕获一起回放**：缓存同时记下每个带 hook 的块的输出，命中时按序重新触发那些 hook。
  通用做法要读 torch 的私有 hook 字典并代替别人触发 hook——评审（含写这个设计的那一位）
  自己的结论是**过于取巧**，除非 hook 的主人显式声明可回放。

两条都要自己的认证跑。考虑到它只覆盖 17% 的调用、而且现在最重的那类已经从 12.5 s 降到 8.3 s，
优先级低于把金丝雀重跑一遍（§9 第 3 步）。

**再往后**：剩下那两遍里，注意力是 dense `[1,S,S]` 布尔 mask + math 路径 SDPA，
而视觉注意力是**按图像分块对角**的（28 层还要在图像内部再按窗口分块）。
`S=8620` 时算出来的分数矩阵绝大部分会被 mask 掉。换成按窗口/按图像分块算**数学上等价、
但归约顺序变了，最后几位会动**，要单独认证。`npu_fusion_attention`（`input_layout='TND'`、
`actual_seq_qlen` 给边界、不传 mask）是这台机器上现成的做法。

⚠️ 顺带纠正两个被传开的数：
- `36ac0aa` 关掉的融合 MHA fastpath**不是**那 1,129 ms 未融合注意力的原因。
  `torch.backends.mha.set_fastpath_enabled(False)` 只管 `nn.TransformerEncoder`
  （系统 1 条件编码器、历史头），和 Qwen 视觉塔的 `F.scaled_dot_product_attention` 无关。
- "12 视角 / 9408 patch / 147 窗口"不是部署形状。实测的部署形状是 `7056x9` 和 `8620x10`。
  mask 微基准里的 9408 只是一个比例尺，不是单次调用的成本。

---

## 0.6 2026-10-10 12:0x–12:15：AICPU 超时的根因拿到了（设备侧日志）

§0.5 留下的那两条线索指向"建图图里某个具体算子"。**不是算子，是内存**。设备侧日志
（`~/ascend/log/debug/plog/plog-<pid>_*.log`，Python 的 traceback 里看不到）给出完整链条：

```
12:07:40  GE device_allocator: [Malloc][Memory] failed, rt_ret:207001, size=831782912
          halMemAlloc failed ... caching_mem_allocator: Failed to apply for memory
12:09:55  RUNTIME: Device Aicpu oom, ret=0x7110012  -> 之后每 1.1 秒重试一次，永远
12:14:xx  还在重试；流不再排空，于是下一次主机同步等满 CANN 的约 9 分钟超时，
          在它恰好停在的那一行报 507017
```

**CANN 的图引擎要自己的约 800 MiB 设备内存**来跑 20 视角的建图初始化，而 PyTorch 的缓存
分配器已经把整张卡拿走了——一集之后空载仍是 **65521 / 65536 MiB**，而活着的张量只有约
40 GB。图引擎拿不到，AI CPU 报 oom 并无限重试，表现为"卡住"。

这一条把 §0.5 的三个观察全部解释掉了：

- **为什么总是第 2 集的第一次建图前向**：第 1 集把缓存涨满，第 2 集的建图初始化是之后
  第一个需要图引擎工作区的地方；
- **为什么 `task_id` 会重复**：同一张图里同一个任务，每次都在同一处要不到内存；
- **为什么喂合成噪声也能复现、和并发无关**：只要先跑满一集，和跑什么无关。

### `rope.py:183` 是替罪羊，更正 `81d063b` 的提交信息

两次 traceback 都停在 `max_position = int(positions.max()) + 1`，因为那是**整个前向里第一次
主机同步**——流一停，谁先同步谁背锅。把它去掉之后第二集照样挂，只是日志里连一行都不再出现。

那个改动**保留**，因为它本身是对的：真机上 1/2/8/20 视角、有无特殊 token 偏移共 8 个形状
全部 `torch.equal`，最大绝对差 0.0；而且确实省掉每次前向 54 次主机等待（提交信息里写的"约
80 次"也偏高：anyview 分支是 40 层 ViT-g、`rope_start=13`，27 层 × q/k = 54；metric 分支
`rope_start=-1`，没有 rope）。

⚠️ **它有个副作用**：上界取 token 轴长度，而全局注意力块的 `pos_nodiff` 精确最大值只有 2，
token 轴却是 S×N（20 视角时 20740）。频率表按 `seq_len` 缓存，S 在一集内会变，所以每个不同
视角数都新建一张约 5 MB 的表。视角数有界（≤ 约 21），所以在约 200 MB 封顶、不是无界泄漏，
但方向是错的——正确做法是在 `_prepare_rope` 里用主机侧已知的 h、w 算出上界传进来。**没做。**

### 处置（`81dfa08`，两条都不改变任何数值）

- 里程计后端在 `reset()` 里把缓存块还回去：刚好在丢掉 keyframe memory 之后、下一次建图
  初始化之前，也就是缓存最大、图引擎马上要申请的那一刻。只对 NPU 生效，CUDA 路径不碰。
- 启动脚本设 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（可覆盖，进日志）。那 62 GB
  里大部分是碎片而不是活张量，这正是这个开关的用途。

### 同一类还没做的（审计查出来，全在这条路径上）

| 位置 | 是什么 | 能否逐位相同地去掉 |
|---|---|---|
| `utils/io/output_processor.py` ×6 + `model_zoo.py` ×4 | 每次建图前向把整个输出搬到主机再搬回（20 视角约 1200 万 float32 来回） | **能**，部署配置下中间没有任何主机算术 |
| `slam/pipeline.py:26-28` | 对整个置信度体 reduce + 拷回主机，只为打印一行日志，就在挂死的那个前向里 | **能**，这行没有消费者 |
| `da3.py:393/414/435`、`:162/:164` | assert、`.item()`、`min(tensor, float)` | **能**（`torch.clamp_max` 等） |
| `alignment.py:114` | `randperm(约 400 万)` 抽 10 万样本，每次前向一次 | **不能**，换 `randint` 会改变抽样 |

---

## 0.7 ✅ 2026-10-10 12:24–12:35：这台机器上第一次把 4 集干净跑完

`81dfa08` 之后，同样那 4 集、钉死的集表、`SHARD_RETRIES=0`、**2 槽并发**
（`exp22_smoke_memfix`，服务端 `20261010_121957_1215008`，commit 81dfa08）：

| 集 | 结果 |
|---|---|
| zsNo4HB9uLZ/1 | success 1.0、SPL 0.9649、NE 0.1478、48 步、慢系统 14 / 轨迹 11 |
| zsNo4HB9uLZ/25 | success 1.0、SPL 0.8916、NE 0.9095、100 步、29 / 19 |
| zsNo4HB9uLZ/2 | success 1.0、SPL 0.4825、NE 0.8050、109 步、34 / 22 |
| zsNo4HB9uLZ/26 | success 1.0、SPL 1.0000、NE 0.5906、81 步、23 / 17 |

**干净的凭据**：两个分片各只有一次客户端启动（`Episodes already done: 0` 各一次，没有重复块
或残块），`COMPLETE`，没有 `RETIRED`，服务端日志里 `507017` / `Aicpu oom` 各 0 次。

关键的是 **`zsNo4HB9uLZ/25` 跑完了**——它是之前两次金丝雀死掉的那一集，两次都死在第 20 帧的
建图初始化。每个槽都连着跑了两集，所以"跨集残留"这条也走通了。

⚠️ **这不是 EXP-22 的 P1**，别当成 H1 过了：它在不同的 commit 上、当时只有 2 槽、而且是作为
修复的验证跑的，不是按 P1/P2 成对跑的。**H1 仍然没有过**，要过得按 §判据 重新跑一对。
它能支撑的只有一句话：**那个挂死被修好了，而且这台机器能把 4 集连着跑完**。

### 顺带：6 槽并发第一次跑起来了

同一套服务端扩到 6 槽（卡 0–5，一卡一槽）驱动 EXP-21 的敏感性点时的读数：

| 项 | 值 |
|---|---|
| 910B 负载 | 10.75（192 核） |
| 卡 0–5 HBM | 26–41 GB / 65.5 GB（空载基线 3.4 GB） |
| 卡 6–7 | 3.4 GB，空着（留给服务端侧覆盖的 S / M 两组） |
| 4090 负载 | 20（128 核，还有别人的任务在跑） |

**为什么是 6 槽而不是 8**：客户端启动脚本要求每槽一个不重复的 GPU id
（`run_ppa_r2r_val_unseen_cuda.sh:186`），而 4090 只有 6 张卡。再起第 8 个服务端也没有客户端
能驱动它。

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

⚠️ **这一行是在没设 `PYTHONPATH` 的情况下量的，所以它少算了。** `vla_rpc` 不在包路径上时，
几个测试文件会被 `pytest.importorskip("vla_rpc")` 整体跳过（上面那 35 skipped 里就有它们）。
把 `PYTHONPATH="$ROOT/rpc/src:$ROOT/HeatmapVLN"` 加上再跑，这台机器的基线是：

```
14 failed, 28 skipped, 25 errors（第二轮加的测试之前）
```

多出来的那两条是 `tests/test_rpc_pano_two_phase.py::test_two_phase_rpc_skips_system1_until_after_real_recenter`
和 `::test_internnav_rpc_runs_second_lookdown_generation`，签名是
`AttributeError: 'HeatmapVLNRuntime' object has no attribute 'system2_cognition_arm'`——
**CLAUDE.md §5.1 那一类**：`164c67d` 给 `plan_panoramic` 加了一个分支
（`rpc_model_server.py:1327`），而这两个测试用 `object.__new__` 造桩，桩没跟上。
干净 HEAD 加同一个 `PYTHONPATH` 复现一模一样的两条，所以**不是这一轮引入的**。

⚠️ **不要顺手补这个桩。** 试过了：补上 `system2_cognition_arm = False` 之后，下一个缺的是
`ppa_online_amb3r`（`rpc_model_server.py:1708`），而这一个不是机械填值——部署里它是 `True`
（启动脚本传 `--require_ppa_online_amb3r`），而这两个测试是按 PPA 之前的两阶段协议写的
（看它们设的 `has_nextdit` / `pano_latent_adapter`）。填 `False` 能让它们过，但那是在测一个
部署根本不用的配置；填 `True` 会把它们送进没写过的分支。**决定这两个测试该断言什么，是关于
旧两阶段协议的判断，不是补桩**——和 CLAUDE.md §4 里那条 `_dataloader_in_order_kwargs`
collection error 同类，单独一件活。所以这台机器在 `PYTHONPATH` 设对时的基线就是
**14 failed / 25 errors**。

以后跑全量**一定要设 `PYTHONPATH`**，否则会把真失败藏成 skip。

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

## 8. 速度：这一节的账已经重写，见 §0.5

**本节原来那套"剩下的 4 倍"的归因作废**，不要再引用它，也不要再引用它里面的
"11.2 秒一次调用 / 对 4090 的 2.1 倍"。两个理由：

1. 它把"一次调用"当成一个数。实测是三类、差 5 倍（§0.5 的表）：`native_actions` 2349 ms、
   `trajectory` 9107 ms、`trajectory+ppa` 11079 ms。11.2 s 只对得上最重的那一类。
2. 它把成本记在"两轮解码 / 条件潜变量 / 历史头"这三个**阶段**上，但服务端自己数出来的
   视觉塔过数说明：那三个阶段的时间**绝大部分是同一个视觉塔跑了四遍**，而第 3、4 遍和
   第 1、2 遍的同一组图像。靶子是"少跑两遍塔"，不是"换三个阶段的算子"。

关掉融合注意力 fastpath 那条仍然成立（那条路径没有 NPU 实现、会悄悄回落 CPU：单次前向
1.35 s vs 0.0037 s，**365 倍**），但要注意它管的是 `nn.TransformerEncoder`
（系统 1 条件编码器、历史头），**不是** Qwen 视觉塔的注意力。

下面两笔"能换速度但要重新认证"的账照旧有效：

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

   ⚠️ **这一遍要在撤掉 `TASK_QUEUE_ENABLE=0` 之后跑**（§0.5）。那个开关是上一次 4/4 唯一
   带着的东西，它更慢，而且它当初就是为了绕开 AICPU 挂死才加的——所以撤掉之后挂死会不会
   回来，只有重跑金丝雀才知道。如果回来了，那正好把第 7 步的根因问题变成可复现的。

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
6. 再决定要不要把 EXP-21 搬过来——前提是先在昇腾上跑出自己的 A0。
   ⚠️ **"约 22 小时"这个估计作废**，它是按"一次调用 11.2 秒"×集数算的，而一次调用分三类、
   每类不同（§0.5）。要重估就用金丝雀实测的每步墙钟，不要用单次调用数乘。
7. 速度：§0.5 末尾那块"每份输入只过一遍视觉塔"是目前最大的一块（最重那类调用的约 4.6 s）。
   它**不堵着**第 3–6 步，可以并行做，但它要自己的认证跑，所以不要和金丝雀混在一遍里。
8. ~~AICPU 超时：从"真实帧能不能复现"开始查~~ —— 已复现（§0.5）：真实帧、单进程、机器上没别的任务，卡在**第 2 集的第一次建图前向**，`task_id:34378`。**第 3、4 步现在被它堵着**，下一步应当从这里开始：先看第 1 集到第 2 集之间 AMB3R 侧留下了什么状态，再看那个 AI CPU 算子是哪一个。
