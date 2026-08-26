# 集群 Habitat/GLX 无头运行环境交接：Xvfb、XKB bundle 与 llvmpipe

更新日期：2026-08-03
适用环境：`finn_cci_c500` 开发机及其集群任务容器，`habitat_sim 0.1.7` 的 GLX 构建，`/mnt/afs/liwenhao/agent/370910109/envs/vlnce`

## 结论先行

当前集群上已经跑通并完成 1839 个 R2R `val_unseen` episode 的稳定方案是：

1. 不依赖作业镜像中的系统 X11/OpenGL 包，固定使用：
   `/mnt/afs/liwenhao/agent/370910109/tools/x11_headless_bundle_ubuntu22_20260801_v4`
2. 每个 Habitat client 使用一个独立 Xvfb display。
3. 使用 bundle 内的 Mesa GLX，并显式强制 `llvmpipe` 软件渲染。
4. 不强制 NVIDIA GLX；必须清掉可能继承的 `__GLX_VENDOR_LIBRARY_NAME` 和 `__EGL_VENDOR_LIBRARY_FILENAMES`。
5. Xvfb 和 Habitat client 使用不同的 bundle library path；不要把 bundle 全局注入模型 server。
6. 并发任务必须使用互不重叠的 display、RPC 端口和输出目录。

这套方案牺牲了一些渲染速度，但在当前容器和现有 `habitat_sim` GLX 二进制上是已完成全量评测验证的方案。不要根据旧文档把 `__GLX_VENDOR_LIBRARY_NAME=nvidia` 当作默认修复。

## 已验证的具体文件

稳定 bundle：

```text
/mnt/afs/liwenhao/agent/370910109/tools/x11_headless_bundle_ubuntu22_20260801_v4/
├── bin/Xvfb
├── bin/xdpyinfo
├── bin/glxinfo
├── bin/xkbcomp          # wrapper，强制使用 bundle XKB 数据
├── bin/xkbcomp.real
├── lib/                 # Xvfb/X11 工具所需完整运行库
├── mesa_lib/            # Habitat client 使用的较窄 Mesa/GL 运行库集合
├── dri/swrast_dri.so
├── share/X11/xkb/       # 完整 xkeyboard-config 数据，含 keycodes/evdev
├── share/fonts/misc/
├── manifest.json
└── manifest.sha256
```

bundle ABI 约束记录在 `manifest.json`：`x86_64`、host glibc `>= 2.35`。bundle 的 wrapper 和校验清单包含绝对路径，因此应直接复用上述固定目录，不要随意复制或改名。

2026-08-03 核对到的清单摘要：

```text
manifest.json   sha256 1a733a7fd63170ac87ded691a5d52f01121af318ce7804a9724f7e9a9e4c2a2c
manifest.sha256 sha256 c9d05c514cc1d93ad676f4ef5fe6447be9d9d7a9dc6a26af621f83e29e5916c1
XKB data files  294
```

完整、已验证的 8 卡启动实现：

```text
/mnt/afs/liwenhao/agent/370910109/evaluation_plans/
  stage3_priorfix_r2r_val_unseen_epoch2_8gpu_20260801/
  scripts/run_8gpu_rpc_eval.sh
```

当前评测端针对 GLX/import 顺序的保护代码：

```text
/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/scripts/evaluation/r2r_val_unseen.py
```

## 实际遇到的坑与根因

### 1. 开发机有 Xvfb，不代表集群作业镜像里也有

最初在开发机预检正常，但集群任务直接报：

```text
Missing command: Xvfb
```

根因是开发容器和作业容器的系统包不一致。`conda activate` 只能解决 Python 依赖，不能保证 `/usr/bin/Xvfb`、X11 字体、XKB 数据和 Mesa DRI 驱动都存在。

解决办法是使用固定 bundle 中的绝对路径，而不是 `command -v Xvfb` 找系统命令：

```bash
X11_BUNDLE=/mnt/afs/liwenhao/agent/370910109/tools/x11_headless_bundle_ubuntu22_20260801_v4
XVFB_BIN="$X11_BUNDLE/bin/Xvfb"
XDPYINFO_BIN="$X11_BUNDLE/bin/xdpyinfo"
GLXINFO_BIN="$X11_BUNDLE/bin/glxinfo"
```

### 2. 只带 Xvfb 不够，`magnum` 导入还会缺 `libOpenGL.so.0`

第二个错误发生在 Python 导入阶段：

```text
ImportError: libOpenGL.so.0: cannot open shared object file: No such file or directory
```

调用链是 `habitat` → `habitat_sim` → `magnum` → `_magnum`。这发生在真正创建 simulator 之前，所以必须在启动 Python 进程时设置动态库路径；在 Python 导入之后再改环境变量已经太晚。

v4 bundle 中包含 `libOpenGL.so.0`、GLX/Mesa 相关库和 `swrast_dri.so`。Habitat client 的关键环境是：

```bash
LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:${LD_LIBRARY_PATH:-}"
LIBGL_DRIVERS_PATH="$X11_BUNDLE/dri"
```

不要只补 `libOpenGL.so.0` 一个文件。GLVND、Mesa GLX、DRI 驱动和它们的 ABI 必须来自同一套 bundle，否则常见结果是导入成功、创建 GL context 时再崩溃。

### 3. v3 bundle 能找到 Xvfb，但缺完整 XKB 数据

v3 的失败日志是：

```text
The XKEYBOARD keymap compiler (xkbcomp) reports:
> Error: Can't find file "evdev" for keycodes include
XKB: Failed to compile keymap
Keyboard initialization failed.
Fatal server error:
Failed to activate virtual core keyboard: 2
```

对应证据：

```text
/mnt/afs/liwenhao/agent/370910109/model/
  eval_stage3_r2r_val_unseen_after_scale_stage3_r2r_epoch2_8gpu_rpcv2_x11bundle/
  runtime/20260801_141627_job1/logs/xvfb_0.log
```

`Errors from xkbcomp are not fatal` 这句话在该日志中具有误导性；紧接着 XKB 初始化失败，Xvfb 实际退出了。

v4 的修复不只是复制一个 `evdev` 文件，而是包含完整的 `share/X11/xkb`，并做了三件必须配套的事情：

1. `PATH` 以 `bundle/bin` 开头，使 Xvfb 调用 bundle 的 `xkbcomp` wrapper。
2. wrapper 给真实 `xkbcomp` 增加 `-I$BUNDLE/share/X11/xkb`。
3. Xvfb 显式传入 `-xkbdir "$BUNDLE/share/X11/xkb"`。

此外，v4 的 Xvfb 已将系统 `/var/lib/xkb/` 缓存路径改为 `/dev/fd/9/`。启动前必须为每个 rank 建立自己的缓存目录，并继承 fd 9：

```bash
mkdir -p "$xvfb_runtime/.xkb-cache"
(
  cd "$xvfb_runtime"
  exec 9<"$xvfb_runtime/.xkb-cache"
  # 随后 exec Xvfb
)
```

不要删除这段看起来“不像必需”的 fd 9 逻辑；它是为了避免依赖集群容器中不可写或不存在的 `/var/lib/xkb`。

### 4. 强制 NVIDIA GLX 在当前栈上不稳定

旧排障记录曾建议：

```bash
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

但当前 `habitat_sim 0.1.7` 是 GLX 构建，不是 EGL/headless 构建；在本集群的 Xvfb、numba/LLVM、NVIDIA GLX 组合下，这一路径出现过不可由 Python 捕获的 X11 `BadWindow`/进程退出。即使通过提前创建 simulator 调整 import 顺序，也不适合作为长时间采集任务的默认路径。

当前稳定方案在启动 Xvfb、`glxinfo` 和 Habitat client 时都显式清理 vendor override：

```bash
env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES ...
```

评测 Python 也有保护：除非明确设置 `HEATMAPVLN_ALLOW_NVIDIA_GLX=1`，否则会移除继承到的 NVIDIA GLX override。

### 5. llvmpipe 是 CPU 渲染，不会因为多给 GPU 就变快

稳定配置为：

```bash
LIBGL_ALWAYS_SOFTWARE=1
GALLIUM_DRIVER=llvmpipe
MESA_LOADER_DRIVER_OVERRIDE=swrast
LIBGL_DRIVERS_PATH="$X11_BUNDLE/dri"
```

预检必须看到：

```text
OpenGL renderer string: llvmpipe (...)
```

如果 renderer 为空、是 NVIDIA、或不是 llvmpipe，应直接 fail-fast，不能继续采集。否则不同 rank 可能偷偷使用不同渲染路径，结果和速度都不可控。

llvmpipe 使用 CPU。`LP_NUM_THREADS` 是每个 Habitat client 的 llvmpipe 线程数；8 个 client 若各自设置为 8，就可能占用约 64 个渲染线程。应按集群实际分配的 CPU 核数设置，避免过度超卖：

```bash
LP_NUM_THREADS=${EVAL_LP_NUM_THREADS:-8}
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMBA_NUM_THREADS=1
```

GPU 仍用于 8 个模型 RPC server；Habitat 画面渲染本身走 CPU llvmpipe。

### 6. 多 rank 或多 agent 共享一个 display 会产生难复现故障

一个 Xvfb 被多个 Habitat client 共用时，任何一个 client 的 C/C++ 级崩溃都可能污染该 display。并行作业还会发生 display、RPC 端口和输出锁冲突。

规则如下：

- 每个 Habitat client 一个 display，例如 8 卡使用 `260..267`。
- 同时运行的另一个任务必须改用另一段，例如 `280..287`。
- 每个模型 server 一个 RPC 端口；并发任务也必须使用不同端口段。
- 每个任务使用不同 `EVAL_OUTPUT_ROOT`。
- 启动前用 `xdpyinfo` 检测 display；已占用时默认失败，不要静默复用。
- 只清理由当前脚本记录的 `XVFB_PIDS`；不要在共享节点执行 `pkill Xvfb` 或 `kill $(pgrep Xvfb)`。
- 用 `trap` 回收本任务的 client、server 和 Xvfb 子进程。

推荐并发参数示例：

```bash
export EVAL_DISPLAY_BASE=280
export EVAL_RPC_PORT_BASE=51400
export EVAL_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/<unique_eval_name>
```

### 7. X11 bundle 不应污染模型 server 的运行环境

我们采用 RPC 拆分：

- Qwen/System1/System2 模型 server：`qwen25` Python 环境，只绑定 GPU 和 RPC 端口，不设置 `DISPLAY`，也不注入 Mesa bundle。
- Habitat client：`vlnce` Python 环境，设置独立 `DISPLAY` 和 Mesa/llvmpipe 环境，通过 RPC 请求模型输出。

不要在 launcher 顶层执行：

```bash
export LD_LIBRARY_PATH="$X11_BUNDLE/lib:$LD_LIBRARY_PATH"
```

这会让 bundle 内的 `libstdc++`、LLVM、X11/Mesa 库同时污染 PyTorch、flash-attn、MACA/CUDA 和模型 server。正确做法是只对具体的 Xvfb/检测命令/Habitat client 使用命令级 `env ...`。

## 可复用的已知可行 Bash 核心代码

下面是从已完成全量评测的 launcher 中抽出的 X11 部分。采集脚本可以复用，但必须自行选择唯一的 `DISPLAY_BASE` 和 runtime/output 目录。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
X11_BUNDLE="$FJL_ROOT/tools/x11_headless_bundle_ubuntu22_20260801_v4"
XVFB_BIN="$X11_BUNDLE/bin/Xvfb"
XDPYINFO_BIN="$X11_BUNDLE/bin/xdpyinfo"
GLXINFO_BIN="$X11_BUNDLE/bin/glxinfo"
X11_DRI_PATH="$X11_BUNDLE/dri"
X11_FONT_PATH="$X11_BUNDLE/share/fonts/misc"
X11_XKB_PATH="$X11_BUNDLE/share/X11/xkb"

BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
X11_TOOL_LD_LIBRARY_PATH="$X11_BUNDLE/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
X11_CLIENT_LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"

NUM_SHARDS=8
DISPLAY_BASE="${EVAL_DISPLAY_BASE:?set a unique EVAL_DISPLAY_BASE}"
LP_THREADS="${EVAL_LP_NUM_THREADS:-8}"
RUNTIME_DIR="${EVAL_RUNTIME_DIR:?set a unique EVAL_RUNTIME_DIR}"

declare -a XVFB_PIDS=()
declare -a DISPLAYS=()

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup_xvfb() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${XVFB_PIDS[@]:-}"; do stop_pid "$pid"; done
  exit "$status"
}
trap cleanup_xvfb EXIT
trap 'exit 130' INT TERM

for required in \
  "$XVFB_BIN" "$XDPYINFO_BIN" "$GLXINFO_BIN" \
  "$X11_BUNDLE/bin/xkbcomp" "$X11_DRI_PATH/swrast_dri.so"; do
  [[ -x "$required" || -s "$required" ]] || {
    echo "Missing bundle component: $required" >&2
    exit 1
  }
done
[[ -d "$X11_XKB_PATH" ]] || { echo "Missing XKB tree: $X11_XKB_PATH" >&2; exit 1; }

for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  display_num=$((DISPLAY_BASE + rank))
  display_addr="127.0.0.1:${display_num}.0"
  DISPLAYS[$rank]="$display_addr"

  if env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
    LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
    DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    echo "DISPLAY already occupied: $display_addr" >&2
    exit 1
  fi

  xvfb_runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/xvfb"
  xvfb_log="$RUNTIME_DIR/logs/xvfb_${rank}.log"
  mkdir -p "$xvfb_runtime/.xkb-cache" "$RUNTIME_DIR/logs"

  (
    cd "$xvfb_runtime"
    exec 9<"$xvfb_runtime/.xkb-cache"
    exec env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
      PATH="$X11_BUNDLE/bin:$PATH" \
      LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
      LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
      LIBGL_ALWAYS_SOFTWARE=1 \
      GALLIUM_DRIVER=llvmpipe \
      MESA_LOADER_DRIVER_OVERRIDE=swrast \
      LP_NUM_THREADS="$LP_THREADS" \
      "$XVFB_BIN" ":$display_num" \
      -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
      -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH"
  ) >"$xvfb_log" 2>&1 &
  XVFB_PIDS[$rank]=$!

  ready=0
  for _ in $(seq 1 60); do
    if ! kill -0 "${XVFB_PIDS[$rank]}" 2>/dev/null; then
      echo "Xvfb rank=$rank exited during startup" >&2
      tail -100 "$xvfb_log" >&2 || true
      exit 1
    fi
    if env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
      LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
      DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" == 1 ]] || { echo "Xvfb rank=$rank did not become ready" >&2; exit 1; }

  renderer="$(env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
    LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
    LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    DISPLAY="$display_addr" timeout 120 "$GLXINFO_BIN" -B 2>/dev/null | \
    grep -F 'OpenGL renderer string:' | head -1 || true)"
  [[ "${renderer,,}" == *llvmpipe* ]] || {
    echo "Unexpected renderer on $display_addr: ${renderer:-missing}" >&2
    exit 1
  }
  echo "rank=$rank DISPLAY=$display_addr $renderer"
done

# 对第 rank 个 Habitat client 使用如下命令级环境：
rank=0
env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
  LD_LIBRARY_PATH="$X11_CLIENT_LD_LIBRARY_PATH" \
  LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
  DISPLAY="${DISPLAYS[$rank]}" \
  LIBGL_ALWAYS_SOFTWARE=1 \
  GALLIUM_DRIVER=llvmpipe \
  MESA_LOADER_DRIVER_OVERRIDE=swrast \
  LP_NUM_THREADS="$LP_THREADS" \
  HEATMAPVLN_ALLOW_NVIDIA_GLX=0 \
  HEATMAPVLN_PREINIT_GL=0 \
  HEATMAPVLN_PREINIT_EMPTY_GL=1 \
  HABITAT_GL_GPU_ID=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /mnt/afs/liwenhao/agent/370910109/envs/vlnce/bin/python -u <collector.py> <args...>
```

最后一条命令中的 `<collector.py> <args...>` 是占位符，不应原样执行。完整任务请复制上面列出的已验证 launcher，再替换数据、cohort、RPC server 和输出路径。

## 提交前最低限度检查

### 1. bundle 完整性

```bash
X11_BUNDLE=/mnt/afs/liwenhao/agent/370910109/tools/x11_headless_bundle_ubuntu22_20260801_v4
test "$(uname -m)" = x86_64
getconf GNU_LIBC_VERSION
sha256sum -c "$X11_BUNDLE/manifest.sha256"
```

`manifest.sha256` 使用绝对路径；校验应在 bundle 的固定原路径执行。

### 2. Xvfb 是否真的活着

不要只检查 PID；必须让 `xdpyinfo` 成功：

```bash
DISPLAY=127.0.0.1:260.0 \
LD_LIBRARY_PATH="$X11_BUNDLE/lib:${LD_LIBRARY_PATH:-}" \
timeout 5 "$X11_BUNDLE/bin/xdpyinfo" >/dev/null
```

### 3. renderer 是否确定为 llvmpipe

必须检查 `glxinfo -B` 的 renderer 行，而不是只检查 GLX extension 是否存在。

### 4. Habitat import 是否使用相同环境

```bash
env -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
  LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:${LD_LIBRARY_PATH:-}" \
  LIBGL_DRIVERS_PATH="$X11_BUNDLE/dri" \
  LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
  /mnt/afs/liwenhao/agent/370910109/envs/vlnce/bin/python -c \
  'import magnum, habitat_sim; print("magnum/habitat_sim import OK")'
```

这一步必须在作业容器中通过。开发机通过不能替代集群容器验证。

## 成功证据

v4 + 独立 Xvfb + llvmpipe 已经完成一轮 8 shard 全量 R2R `val_unseen` 评测：

```text
/mnt/afs/liwenhao/agent/370910109/model/
  eval_stage3_priorfix_r2r_val_unseen_epoch2_8gpu_rpcv2_x11bundle_v4/
  merged/manifest.json
```

该 manifest 的环境相关完成性证据为：

```text
status: passed
total_episodes: 1839
scene_count: 11
pairwise_disjoint: true
union_exact: true
missing_rows: 0
duplicate_rows: 0
```

对应的 8 个成功 Xvfb 日志位于：

```text
.../runtime/20260801_213940_job1/logs/xvfb_0.log
...
.../runtime/20260801_213940_job1/logs/xvfb_7.log
```

这些日志为空是正常现象：Xvfb 正常运行时没有 stderr；launcher 使用 `xdpyinfo` 和 `glxinfo` 做了主动健康检查。

## 给后续 agent 的最终建议

- 直接以 v4 launcher 为模板，不要从 v3 修补。
- 不要依赖作业镜像中的 `/usr/bin/Xvfb`、`/usr/share/X11/xkb` 或系统 Mesa。
- 不要默认开启 NVIDIA GLX；当前已验证基线是 llvmpipe。
- 不要把 bundle 的 `LD_LIBRARY_PATH` 全局 export 给模型 server。
- 不要共享 display，不要复用别人的输出目录，不要全局杀 Xvfb。
- 先 fail-fast 验证 Xvfb、XKB、renderer 和 Python import，再开始长时间采集。
- 比较不同模型时保持完全相同的 llvmpipe/bundle 和评测协议，避免把渲染栈变化误认为模型差异。
