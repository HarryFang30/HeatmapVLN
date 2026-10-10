# EXP-22 运行手册：昇腾 910B 起步（P1 / P2 / P3 / A0）

判据和设置在 [`README.md` 的 EXP-22 节](README.md#exp-22-昇腾-910b-起步这台机器的数能不能报能不能和-4090-比)，
**这份只记怎么跑**。部署本身见 [`../ops/deploy_ascend_910b.md`](../ops/deploy_ascend_910b.md)，
未决问题见 [`../ops/ascend_910b_open_problems.md`](../ops/ascend_910b_open_problems.md)。

认证的 commit：**`8766f9b`**。两台机器必须是同一个 commit 且干净——客户端会核对
`servers.json` 的 `repo_commit` 与自己的 `git rev-parse HEAD`，不一致就拒绝，而
`PPA_EVAL_ALLOW_COMMIT_MISMATCH` 在本条里一律不设。

地址、端口、私钥一律走环境变量，**不进仓库**：`PPA_TUNNEL_HOST`、`PPA_TUNNEL_PORT`、
`PPA_TUNNEL_DIR`（私钥和 `known_hosts` 放这里）。

---

## 0. 一次性准备

### 0.1 客户端那台要有一个真正的 git 检出

客户端的 commit 是用 `git rev-parse HEAD` 读的（`scripts/run_ppa_r2r_val_unseen_cuda.sh:287`），
所以 `git archive` 导出的 `src_<sha>/` 目录**不行**——那里没有 `.git`，读出来是 `unknown`，
和服务端的 commit 对不上，直接被拒。4090 容器连不到 GitHub，所以走 git bundle：

```bash
# 笔记本上：增量 bundle（容器里那份部署检出的 commit 是它的祖先，所以很小）
git bundle create /tmp/hv.bundle ^<容器里已有的 commit> refs/heads/<本次分支>
ssh 6024_fjl "docker exec -i fjl-habitat bash -c 'cat > /workspace/npu_eval/hv.bundle'" < /tmp/hv.bundle
```

容器里（**不要动部署检出 `/workspace/HeatmapVLN`**，只读它的对象）：

```bash
cd /workspace/npu_eval
git clone --local --no-checkout /workspace/HeatmapVLN repo_exp22
cd repo_exp22
git fetch ../hv.bundle 'refs/heads/*:refs/remotes/bundle/*'
git checkout -B exp22 8766f9b
git rev-parse HEAD; git diff --quiet HEAD -- && echo clean
chown -R 1015:1015 /workspace/npu_eval/repo_exp22      # 容器里是 root 写的
```

> 2026-10-10 那次这个目录叫 `repo_e6c3844`（建的时候是那个 commit，后来 fast-forward 到了
> `8766f9b`）。名字不影响任何检查，`exp22_lists/manifest.json` 里记的是工具的 sha256。

### 0.2 钉死的 4 集集表

```bash
cd /workspace/npu_eval/repo_exp22
/opt/conda/bin/python scripts/tools/make_episode_lists_from_run.py \
  --reference /workspace/eval_runs/canary_cuda_seed42 \
  --cohorts /workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/cohorts \
  --shards 0,1 --expect-per-shard 2 \
  --out /workspace/eval_runs/exp22_lists
```

得到 `shard_00.json`（zsNo4HB9uLZ 的 1、25）、`shard_01.json`（2、26）和 `manifest.json`。
工具会拒绝一个被续跑或重启过的参照，所以跑之前自己先看一眼也无妨：
`runtime/` 下只能有一个目录，每个分片日志里 `Episodes already done: 0` 只能出现一次，
每个 `progress.json` 正好 2 行且不重复。10-10 实测参照合格，sha256 与台账 §4 记的一致。

**不设 `PPA_EVAL_MAX_EPISODES_PER_SHARD`。** 集数上限只对"新跑的集"计数，重启会再给一批新集。

---

## 1. 每一遍开跑前的检查（910B）

```bash
R=$HOME/work/zhr/zhr_1/HeatmapVLN
git -C $R rev-parse HEAD; git -C $R status --porcelain          # 8766f9b，且干净
env | grep -E '^(TASK_QUEUE|PPA_|ASCEND_RT|HEATMAPVLN|INTERNNAV|DA3_)'  # 应该什么都没有
ps -eo pid,etime,args | grep -E 'rpc_model_server|rpc_amb3r_vo_server' | grep -v grep
npu-smi info | awk -F'|' '/0000:/ {print $4}'                  # 8 张卡都该是约 3.4 GB
ss -ltn | grep -E '5240|5250' || echo "端口空着"
bash $R/scripts/ascend/check_amb3r_patch.sh $HOME/work/zhr/zhr_1/amb3r   # ok 且无 WARN
```

`PYTHONPATH` 会有 CANN 自己的那几段（登录时 `set_env.sh` 设的），那是正常的：启动脚本把
我们的路径放在**前面**。镜像还塞了一个 `/home/ma-user/infer/model/1`，里面没有 `.py`，无害。

**P1–P3 期间这台机器上只跑这一套服务端**（判据要求；原因是 AICPU 超时还没定性）。

---

## 2. 起服务端（P1 / P2 各一次，各自新起）

没有 tmux，所以 `nohup setsid` + 重定向；**停的时候用 `kill -TERM <pid>`，不要 `-9`**
（`-9` 不走 trap，卡和端口都不会放，`STOPPED` 也不会写）。

```bash
cd $HOME/work/zhr/zhr_1/HeatmapVLN
LOG=$HOME/work/zhr/zhr_1/logs/exp22_p1_servers.log      # P2 换成 p2
nohup setsid env \
  -u TASK_QUEUE_ENABLE -u ASCEND_RT_VISIBLE_DEVICES \
  -u INTERNNAV_BACKBONE -u INTERNNAV_MODEL_PATH -u HEATMAPVLN_LLM_MODEL_PATH \
  -u HEATMAPVLN_QWEN_VISION_REUSE -u HEATMAPVLN_QWEN_VISION_COUNT -u HEATMAPVLN_QWEN_VISION_MASK_PATCH \
  -u PPA_EVAL_TIMING -u PPA_EVAL_BRIDGE_OFF -u PPA_EVAL_NUM_SAMPLE_TRAJS -u PPA_EVAL_NUM_INFERENCE_STEPS \
  -u PPA_NPU_VO_DEVICES -u PPA_NPU_PROFILE_DIR \
  bash scripts/ascend/run_ppa_servers_npu.sh >"$LOG" 2>&1 </dev/null &
```

为什么要 `-u` 这么多：`INTERNNAV_BACKBONE` / `HEATMAPVLN_LLM_MODEL_PATH` 任一存在都会
**悄悄换掉权重**（`src/config_schema.py`），而三个视觉开关要让启动脚本的默认值说话，
不要让启动它的那个 shell 说话。

启动完该在日志里看到（缺一条启动脚本就不让跑）：

```
[ppa-npu] qwen_vision_reuse=1 mask_patch=<on for npu> pass_count=<on for npu>
[ppa-npu] task_queue_enable=<platform default>          # 不是 0，也没有 WARN
[ppa-npu] runtime=.../servers/<STAMP> instance=<STAMP>-<16 位随机>
[ppa-npu] repo=... commit=8766f9b6f96d dirty=0
[amb3r-patch] ok   tree matches the certified base 09f1b2f outside the two patched files
[amb3r-patch] ok   AMB3R tree ... carries the NPU patch
[ppa-npu] npu=0 used=3406MiB free enough                # 读到真实 HBM，不是 0
[ppa-npu] slot=0 ... model=127.0.0.1:52400 vo=127.0.0.1:52500 ready
[ppa-npu] all servers ready; servers.json=.../servers.json
```

P1 与 P2 的 `instance` 必须不同——这正是"两套分别新起的服务端"的凭据。

## 3. 隧道（4090 容器内）

```bash
cd /workspace/npu_eval/repo_exp22
nohup setsid env PPA_TUNNEL_HOST=<地址> PPA_TUNNEL_PORT=<端口> PPA_TUNNEL_DIR=/workspace/ppa_tunnel \
  bash scripts/ascend/start_tunnel.sh 1 >/workspace/ppa_tunnel/exp22_tunnel.log 2>&1 </dev/null &
```

起之前把上一轮的隧道按 PID 停掉（`pkill -f` 会连自己这条 `bash -c` 一起打中）。
确认：容器里 `127.0.0.1:52400` 和 `52500` 都能连上。

## 4. 把 910B 的运行目录拷到这一遍的输出根下

必须在**服务端还活着的时候**拷（退出时 trap 会写 `STOPPED`，客户端见到 `STOPPED` 就拒绝），
而且要在**容器里**解包——`/home/fangjialei/eval_runs/...` 是 root 建的，宿主机那个用户写不进去。
分两步，不要把脚本和 tar 拼成一条 stdin（bash 会把后面的二进制一起吞掉）：

```bash
# 笔记本：拉下来（这一步顺带断言服务端还在跑）
ssh modelarts_nb "cd ~/work/zhr/zhr_1/servers/<STAMP> && test -s servers.json \
  && test ! -e STOPPED && test ! -e RETIRED && tar -czf - ." > /tmp/p1_servers.tgz
# 笔记本：送进容器，再解
ssh 6024_fjl "docker exec -i fjl-habitat bash -c 'cat > /workspace/npu_eval/p1_servers.tgz'" < /tmp/p1_servers.tgz
ssh 6024_fjl "docker exec fjl-habitat bash -c '
  D=/workspace/eval_runs/exp22_p1_seed42/npu_servers; mkdir -p \$D
  tar --no-same-owner -C \$D -xzf /workspace/npu_eval/p1_servers.tgz
  chown -R 1015:1015 /workspace/eval_runs/exp22_p1_seed42'"
```

拷完核对 `servers.json`：`schema=heatmapvln-npu-servers-v2`、`device=npu`、
`repo_commit=8766f9b…`、`repo_dirty=0`、`timing=0`、`num_sample_trajs=""`、
`num_inference_steps=""`、`vo_rng_seed=0`、`server_instance` 非空。

## 5. 跑客户端（P1）

```bash
ROOT=/workspace/eval_runs/exp22_p1_seed42
REPO=/workspace/npu_eval/repo_exp22
cd $REPO
nohup setsid env \
  -u PPA_EVAL_MAX_EPISODES_PER_SHARD -u PPA_EVAL_ALLOW_CAPPED_RESUME \
  -u PPA_EVAL_ALLOW_COMMIT_MISMATCH -u PPA_EVAL_VO_GPU_DEVICES \
  -u PPA_EVAL_NUM_SAMPLE_TRAJS -u PPA_EVAL_NUM_INFERENCE_STEPS -u PPA_EVAL_BRIDGE_OFF \
  -u PPA_EVAL_NUM_HISTORY -u PPA_EVAL_RPC_TIMEOUT_MS -u PPA_EVAL_RUN_STAMP \
  -u PPA_EVAL_MODEL_PORT_BASE -u PPA_EVAL_VO_PORT_BASE -u PPA_EVAL_CONFIG -u PYTHONPATH \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  PPA_EVAL_REPO=$REPO \
  PPA_EVAL_EXTERNAL_SERVERS=1 PPA_EVAL_EXTERNAL_SERVER_DIR=$ROOT/npu_servers \
  PPA_EVAL_OUTPUT_ROOT=$ROOT \
  PPA_EVAL_EPISODE_LISTS_DIR=/workspace/eval_runs/exp22_lists \
  PPA_EVAL_SHARDS=0,1 PPA_EVAL_GPU_DEVICES=4 PPA_EVAL_DISPLAY_BASE=601 \
  PPA_EVAL_PROTOCOL_SEED=42 PPA_EVAL_TIMING=0 PPA_EVAL_SHARD_RETRIES=0 \
  PPA_EVAL_ARM=ppa_refine_v2_online_amb3r_ascend910b \
  LP_NUM_THREADS=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  bash scripts/run_ppa_r2r_val_unseen_cuda.sh >"$ROOT/launcher.log" 2>&1 </dev/null &
```

- 输出根**每一遍都要是新的**：不带上限的跑法会直接 `--resume` 进已有的行里，启动脚本不拦。
- `PPA_EVAL_GPU_DEVICES` 只是给 Habitat 的占位（渲染走 llvmpipe，CPU），但三遍要保持同一张卡。
- 4 集约 30–40 分钟（10-10 实测一集约 7–9 分钟，一次规划调用约 8.3 秒）。

## 6. 跑完立刻检查（判定前）

```bash
# 客户端这边：每个分片只启动过一次，每个分片 2 行
for f in $ROOT/runtime/*/logs/client_shard_0*.log; do
  echo "$f starts=$(grep -c 'Episodes already done' $f)"; done
wc -l $ROOT/workers/shard_0*/progress.json
grep -E 'COMPLETE|ERROR' $ROOT/launcher.log | tail -3
# 910B 这边：没有退役、没有中毒标记、启动脚本是被 TERM 停的
ls ~/work/zhr/zhr_1/servers/<STAMP>/RETIRED 2>&1
grep -rl 'NPU device unusable after a failed request' ~/work/zhr/zhr_1/servers/<STAMP>/logs || echo NO_POISON
grep -c 507017 ~/work/zhr/zhr_1/servers/<STAMP>/logs/*.log
```

任何一条不对，这一遍就是**没测出来**，整对作废重跑——不拿残缺的一遍去比。
确认无误后再 `kill -TERM` 停服务端，然后把带 `STOPPED` 的完整目录再拷一份进输出根留档。

## 7. P2 和 P3

- **P2**：和 P1 逐字相同，只换输出根（`exp22_p2_seed42`）和服务端日志名。服务端要**重新起**，
  等卡 0 回到约 3.4 GB 再起（`initial_seed` 与进程相关，这正是要测的）。
- **P3**（H2）：和 P1 相同，但两端都开计时——服务端 `PPA_EVAL_TIMING=1` 起，客户端
  `PPA_EVAL_TIMING=1`；客户端会核对 `servers.json` 里的计时开关。输出根 `exp22_p3_timed_seed42`。
  完整性另判：计时文件行数要等于规划调用数，且 `scripts/tools/summarize_latency.py`
  不报服务端阶段缺失（缺失时它退出码 3）。

## 8. 判定

```bash
/opt/conda/bin/python /workspace/exp22/exp22_h1_compare.py \
  --p1 /workspace/eval_runs/exp22_p1_seed42 \
  --p2 /workspace/eval_runs/exp22_p2_seed42 \
  --lists /workspace/eval_runs/exp22_lists \
  --repo /workspace/npu_eval/repo_exp22 \
  --out /workspace/exp22/analysis/p1_vs_p2.json
```

脚本只用标准库（客户端那个解释器还要服务 P3 和 A0，不往里装东西），三档结论按判据写死：
两遍都干净 + 4 集每集 `all_identical` + 每集结局逐位相同 → **支持**；
都干净但有任何一处不同 → **否定**；任一遍不干净（重复块、残块、不足 4 集、槽位退役）
→ **没测出来**，并写明是哪一条。H2 用同一个脚本比 P3 与 P1（`--label p3_vs_p1`）。

---

## 附：这一轮踩到的三件事

1. **视觉塔复用原来是关着的。** 判据把它写成"默认开"，但代码里 `_on()` 要求环境变量正好是
   `"1"`，启动脚本又没设——H1 本来会认证一个没有这项优化的配置。`8766f9b` 让启动脚本默认
   设成 1 并打进日志，顺带加了两条证据行（计数器自己报的那行；以及模型加载时若因适配器
   把复用关掉，就拒绝起）。
2. **把脚本和 tarball 拼进同一条 stdin 会坏。** `bash -s` 读完脚本后 tar 拿到的是残缺的流，
   报 `Error is not recoverable`。分两步。
3. **容器里的目录是 root 的。** 宿主机那个登录用户没法往 `eval_runs/` 下写，所以解包要在容器里做，
   并且写完 `chown -R 1015:1015`。
