# InternNav 评估环境搭建问题排查指南

本文档记录了在基于 InternNav 框架评估 InternVLA-N1 模型（VLN-CE R2R val_unseen 数据集）过程中遇到的所有问题及其解决方案。适用于在 InternNav 基础上搭建自定义模型时参考。

---

## 目录

1. [Habitat-Lab 版本兼容性（YACS vs Hydra）](#1-habitat-lab-版本兼容性yacs-vs-hydra)
2. [NumPy ABI/API 不兼容](#2-numpy-abiapi-不兼容)
3. [gym.spaces.Discrete(0) 断言错误](#3-gymspacesdiscrete0-断言错误)
4. [Transformers 版本不兼容（5.x vs 4.51.0）](#4-transformers-版本不兼容5x-vs-4510)
5. [flash_attn GLIBC 兼容性问题](#5-flash_attn-glibc-兼容性问题)
6. [diffusers 梯度检查点 API 变更](#6-diffusers-梯度检查点-api-变更)
7. [LongCLIP 模块缺失](#7-longclip-模块缺失)
8. [dinov2.py meta tensor 不兼容](#8-dinov2py-meta-tensor-不兼容)
9. [NextDiT FFN 维度不匹配](#9-nextdit-ffn-维度不匹配)
10. [dtw-python 依赖安装引发 NumPy 升级](#10-dtw-python-依赖安装引发-numpy-升级)
11. [habitat_sim 使用 llvmpipe 软件渲染（极慢）](#11-habitat_sim-使用-llvmpipe-软件渲染极慢)
12. [numba LLVM 与 NVIDIA GLX 冲突（致命 X11 错误）](#12-numba-llvm-与-nvidia-glx-冲突致命-x11-错误)
13. [Xvfb 显示状态损坏](#13-xvfb-显示状态损坏)
14. [HabitatVLNEvaluator 与 habitat-lab 0.1.7 不兼容](#14-habitatvlnevaluator-与-habitat-lab-017-不兼容)
15. [build_depthanythingv2 加载外部权重失败](#15-build_depthanythingv2-加载外部权重失败)

---

## 1. Habitat-Lab 版本兼容性（YACS vs Hydra）

### 问题描述

InternNav 代码库中的 `HabitatVLNEvaluator` 和相关配置文件（如 `vln_r2r.yaml`）是基于 habitat-lab 0.2.x+ 的 **Hydra** 配置系统编写的。但安装的 habitat-lab 版本是 **0.1.7**，使用的是 **YACS** 配置系统。两者 API 完全不同：

- Hydra: `@habitat.register_config`, `habitat.get_config()` 返回 OmegaConf 对象
- YACS: `get_config()` 返回 `CfgNode` 对象，使用 `cfg.defrost()` / `cfg.freeze()`

### 解决方案

不使用原有的 YAML 配置文件，改为在 Python 中**程序化构建 YACS 配置**：

```python
from habitat.config.default import get_config as get_habitat_default_config
from habitat.config.default import Config as CN

cfg = get_habitat_default_config()
cfg.defrost()

cfg.DATASET.TYPE = "R2RVLN-v1"
cfg.DATASET.SPLIT = "val_unseen"
cfg.DATASET.SCENES_DIR = "data/scene_data/mp3d_ce"
cfg.DATASET.DATA_PATH = "data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"

# ... 其他配置项 ...

# 自定义 measure 需要手动添加 CN 节点
cfg.TASK.ORACLE_SUCCESS = CN()
cfg.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"

cfg.freeze()
```

### 要点

- habitat-lab 0.1.7 没有 `get_agent_config` 函数（0.2.x 才有），不能直接 import
- 自定义的 Task Measure（如 `OracleSuccess`）需要通过 `CN()` 手动创建配置节点
- `TASK.POSSIBLE_ACTIONS` 必须包含所有需要的动作名称

---

## 2. NumPy ABI/API 不兼容

### 问题描述

`habitat_sim` 0.1.7 编译时使用的是旧版 NumPy（1.x），当环境中安装了 NumPy 2.x 时，会出现 ABI 不兼容错误：

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

此外，旧代码可能使用了 NumPy 1.x 中已废弃的别名（`np.float`, `np.int`, `np.bool`），在 NumPy 1.24+ 中会报错。

### 解决方案

1. **固定 NumPy 版本**：`pip install 'numpy==1.26.4'`
2. **添加兼容性补丁**（在脚本最顶部，任何其他 import 之前）：

```python
import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'bool'):
    np.bool = np.bool_
```

### 要点

- 安装 `dtw-python` 等依赖时可能会自动升级 NumPy，安装后需要立即降回 1.26.4
- 这些补丁必须在 `import habitat_sim` 之前执行

---

## 3. gym.spaces.Discrete(0) 断言错误

### 问题描述

habitat-lab 0.1.7 的 `VLNTask` 调用 `spaces.Discrete(0)` 来创建空动作空间，但 gym 0.26+ 不允许 `n=0`：

```
AssertionError: n (counts) have to be positive
```

### 解决方案

Monkeypatch `gym.spaces.Discrete`：

```python
import gym.spaces
_OrigDiscrete = gym.spaces.Discrete
class _PatchedDiscrete(_OrigDiscrete):
    def __init__(self, n, *args, **kwargs):
        if n == 0:
            n = 1
        super().__init__(n, *args, **kwargs)
gym.spaces.Discrete = _PatchedDiscrete
```

---

## 4. Transformers 版本不兼容（5.x vs 4.51.0）

### 问题描述

这是**最严重、影响面最广**的问题。InternVLA-N1 模型使用 `transformers==4.51.0` 训练（见模型 `config.json` 中的 `transformers_version` 字段）。若环境安装了 `transformers==5.x`，会遇到以下一系列级联错误：

| 错误 | 原因 |
|------|------|
| `AttributeError: 'Config' object has no attribute 'hidden_size'` | 5.x 将通用属性移到了嵌套的 `text_config` 中 |
| `TypeError: mm_token_type_ids` | 5.x 的 processor 添加了 `mm_token_type_ids` 字段 |
| `AttributeError: object has no attribute 'embed_tokens'` | 5.x 改变了模型内部属性层级结构 |
| `AttributeError: object has no attribute 'visual'` | 同上，`visual` 不再是顶级属性 |
| `BaseModelOutputWithPooling has no attribute 'shape'` | 5.x 改变了视觉模型的返回类型 |
| `get_rope_index` 签名变更 | 需要额外的 `mm_token_type_ids` 参数 |

### 解决方案

**降级 transformers 到模型训练时的版本**：

```bash
pip install 'transformers==4.51.0'
```

### 要点

- **始终检查模型 `config.json` 中的 `transformers_version` 字段**，并使用相同版本
- 不要尝试逐个修补 API 差异——改动太多，且容易引入难以发现的行为差异
- 降级 transformers 后，之前添加的大量兼容性 patch（config 代理、属性转发、视觉输出解包等）都可以删除

---

## 5. flash_attn GLIBC 兼容性问题

### 问题描述

`flash_attn` 编译时依赖 `GLIBC_2.32`，但系统的 glibc 版本较低：

```
GLIBC_2.32 not found (required by flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so)
```

即使不直接使用 flash attention（使用 SDPA），`transformers` 在导入 `Qwen2_5_VL` 模型时也会尝试 import `flash_attn` 相关模块。

### 解决方案

在脚本最顶部创建完整的 `flash_attn` stub 模块：

```python
import types as _types, importlib as _importlib

def _noop(*a, **kw):
    raise RuntimeError("flash_attn stub - use SDPA instead")

def _make_stub(name, attrs=None):
    m = _types.ModuleType(name)
    m.__spec__ = _importlib.machinery.ModuleSpec(name, None)
    m.__version__ = '2.8.3'
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m

_fa = _make_stub('flash_attn', {
    'flash_attn_func': _noop,
    'flash_attn_varlen_func': _noop,
})
_make_stub('flash_attn_2_cuda')
_make_stub('flash_attn.flash_attn_interface', {
    'flash_attn_func': _noop,
    'flash_attn_varlen_func': _noop,
})
_make_stub('flash_attn.bert_padding', {
    'index_first_axis': _noop,
    'pad_input': _noop,
    'unpad_input': _noop,
})
_fa_rotary = _make_stub('flash_attn.layers', {})
_make_stub('flash_attn.layers.rotary', {
    'apply_rotary_emb': _noop,
})
```

### 要点

- stub 模块必须有正确的 `__spec__` 属性，否则 `importlib.util.find_spec()` 会报 `ValueError`
- 必须提供 transformers 源码中 `from flash_attn import ...` 实际引用的所有函数名
- 模型加载时使用 `attn_implementation="sdpa"` 避免实际调用 flash attention

---

## 6. diffusers 梯度检查点 API 变更

### 问题描述

`NextDiTCrossAttn` 在初始化时调用 `self.model.enable_gradient_checkpointing()`，但 `diffusers` 0.36.0 改变了 `_set_gradient_checkpointing()` 的签名：

```
TypeError: LuminaNextDiT2DModel._set_gradient_checkpointing() got an unexpected keyword argument 'enable'
```

### 解决方案

在 `nextdit_crossattn_traj.py` 中用 try/except 包裹：

```python
if self._gradient_checkpointing:
    try:
        self.model.enable_gradient_checkpointing()
    except TypeError:
        pass  # 推理时不需要梯度检查点
```

### 要点

- 推理（eval）模式下不需要梯度检查点，安全跳过即可
- 如果需要训练，需要手动适配 diffusers 新版 API

---

## 7. LongCLIP 模块缺失

### 问题描述

`internnav/model/encoder/__init__.py` 无条件导入 LongCLIP 模块，但 `internnav/model/basemodel/LongCLIP/` 目录为空：

```
ModuleNotFoundError: No module named 'internnav.model.basemodel.LongCLIP.model'
```

### 解决方案

创建最小的 stub 文件：

```python
# internnav/model/basemodel/LongCLIP/__init__.py
# Stub

# internnav/model/basemodel/LongCLIP/model.py
class LongCLIP:
    pass
```

---

## 8. dinov2.py meta tensor 不兼容

### 问题描述

`transformers` 的 `from_pretrained` 使用 `torch.device("meta")` 进行模型初始化时，`dinov2.py` 中的 `torch.linspace(...).item()` 调用会失败：

```
RuntimeError: Tensor.item() cannot be called on meta tensors
```

### 解决方案

将 `torch.linspace` 替换为纯 Python 计算：

```python
# 原始代码
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

# 修复后
dpr = [float(drop_path_rate) * i / max(depth - 1, 1) for i in range(depth)]
```

---

## 9. NextDiT FFN 维度不匹配

### 问题描述

`NextDiTCrossAttnConfig` 的 `ffn_dim_multiplier` 默认值为 `None`，导致实例化的 FFN 层维度（1536）与 checkpoint 中保存的维度（1024）不匹配：

```
RuntimeError: size mismatch for model.layers.X.feed_forward.linear_X.weight
```

### 解决方案

将 `ffn_dim_multiplier` 的默认值从 `None` 改为 `2/3`：

```python
# nextdit_crossattn_traj.py 中的 NextDiTCrossAttnConfig
ffn_dim_multiplier: float = 2/3  # 原来是 None
```

---

## 10. dtw-python 依赖安装引发 NumPy 升级

### 问题描述

`NDTW` 指标需要 `dtw-python` 包。安装时 pip 会自动将 NumPy 升级到 2.x，破坏 `habitat_sim` 兼容性。

### 解决方案

安装后立即降级：

```bash
pip install dtw-python
pip install 'numpy==1.26.4'  # 必须立即降回
```

---

## 11. habitat_sim 使用 llvmpipe 软件渲染（极慢）

### 问题描述

**这是最耗时的问题之一**。`habitat_sim` 0.1.7 编译时未使用 `--headless` 参数，只支持 GLX（不支持 EGL headless GPU 渲染）。通过 Xvfb 运行时，默认使用 Mesa 的 **llvmpipe**（CPU 软件渲染器），导致：

- 渲染极慢（单个 `env.reset()` 可能需要数分钟）
- LLVM Pass Registry 错误信息疯狂刷屏（每秒数十万行，stderr 迅速膨胀到 GB 级别）
- 进程看似卡死，实际是在极慢地编译 shader

检查方法——如果看到以下渲染器信息，说明在用软件渲染：
```
Renderer: llvmpipe (LLVM 12.0.0, 256 bits) by Mesa/X.org
```

### 解决方案

设置环境变量强制 GLX 使用 NVIDIA GPU：

```bash
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY=:99  # Xvfb
```

正确的渲染器信息应该显示：
```
Renderer: NVIDIA GeForce RTX 4090/PCIe/SSE2 by NVIDIA Corporation
OpenGL version: 4.6.0 NVIDIA 580.126.09
```

### 要点

- `habitat_sim` 的 `cuda_enabled` 属性指的是 CUDA 传感器（GPU→GPU 数据传输），与渲染 GPU 无关
- `CUDA_VISIBLE_DEVICES` 与 `__GLX_VENDOR_LIBRARY_NAME=nvidia` 可能冲突，避免同时使用
- 如果编译 habitat_sim 时启用了 `--headless` 参数，可以使用 EGL 直接渲染，无需 Xvfb

---

## 12. numba LLVM 与 NVIDIA GLX 冲突（致命 X11 错误）

### 问题描述

**这是最隐蔽的问题**。`numba`（由 `habitat.core.env` 导入）使用自己的 LLVM 运行时。如果 `numba` 在 `habitat_sim` 创建 GL 上下文**之前**被导入，numba 的 LLVM 会破坏 NVIDIA GLX 的 shader 编译环境，导致致命的 X11 错误：

```
X Error of failed request:  BadWindow (invalid Window parameter)
  Major opcode of failed request:  147 ()
  Minor opcode of failed request:  3
  Resource id in failed request:  0x0
```

这个错误是 C 级别的 `exit()`，Python 无法捕获，进程直接终止。

### 解决方案

**在 numba 被导入之前，先创建一个临时的 `habitat_sim.Simulator` 来初始化 NVIDIA GL 上下文**：

```python
import habitat_sim

# 在任何可能触发 numba import 的代码之前执行
_dummy_cfg = habitat_sim.SimulatorConfiguration()
_dummy_cfg.gpu_device_id = 0
_dummy_agent = habitat_sim.agent.AgentConfiguration()
_dummy_agent.sensor_specifications = [habitat_sim.CameraSensorSpec()]
_dummy_sim = habitat_sim.Simulator(
    habitat_sim.Configuration(_dummy_cfg, [_dummy_agent])
)
_dummy_sim.close()

# 现在可以安全导入 habitat（会触发 numba import）
import habitat
```

### 要点

- `import habitat` → `habitat.core.env` → `import numba`，这个导入链会触发问题
- 单独 `import habitat.core.registry` / `import habitat.sims` / `import habitat.tasks` 都不会触发
- `from habitat.core.benchmark import Benchmark` 也会触发（因为它导入 `Env` → `numba`）
- 正确的导入顺序：`habitat_sim`（并创建 Simulator）→ `habitat`
- 这个 bug 只在使用 `__GLX_VENDOR_LIBRARY_NAME=nvidia`（NVIDIA GLX）时出现，Mesa llvmpipe 不受影响（但 llvmpipe 本身有性能问题）

---

## 13. Xvfb 显示状态损坏

### 问题描述

当进程因段错误（SIGSEGV, exit code 139）或 X11 错误崩溃时，Xvfb 的 display `:99` 可能进入损坏状态。后续所有使用该 display 的进程都会立即失败。

### 解决方案

重启 Xvfb：

```bash
kill $(pgrep Xvfb); sleep 1
Xvfb :99 -screen 0 1024x768x24 &
sleep 2
```

### 要点

- 每次遇到 X11 相关的莫名错误时，先尝试重启 Xvfb
- 建议在评估脚本的启动命令中添加自动重启 Xvfb 的逻辑

---

## 14. HabitatVLNEvaluator 与 habitat-lab 0.1.7 不兼容

### 问题描述

`internnav/habitat_extensions/vln/__init__.py` 导入了 `HabitatVLNEvaluator`，该类依赖 `get_agent_config`（habitat-lab 0.2.x+ 的 API）：

```
ImportError: cannot import name 'get_agent_config' from 'habitat.config.default'
```

### 解决方案

不通过包的 `__init__.py` 导入，而是使用 `importlib` 直接加载需要的模块：

```python
import importlib.util

def _load_module_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# 直接加载需要的模块，绕过 __init__.py
_load_module_from_file("vln_measures",
    os.path.join(base_dir, 'internnav', 'habitat_extensions', 'vln', 'measures.py'))
```

---

## 15. build_depthanythingv2 加载外部权重失败

### 问题描述

`InternVLAN1MetaModel.__init__` 调用 `build_depthanythingv2(config)` 时，该函数尝试从独立的 checkpoint 文件加载 DepthAnythingV2 权重。但在使用 `from_pretrained` 加载完整模型时，这些权重已经包含在 safetensors 中，外部 checkpoint 文件不存在导致加载失败。

### 解决方案

Monkeypatch `build_depthanythingv2` 为只创建模型结构、不加载权重的版本：

```python
import internnav.model.basemodel.internvla_n1.internvla_n1_arch as _arch_mod

def _patched_build_dav2(config):
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    DAv2_model = DepthAnythingV2(**model_configs['vits'])
    return DAv2_model.pretrained

_arch_mod.build_depthanythingv2 = _patched_build_dav2
```

---

## 完整启动命令参考

```bash
# 1. 启动 Xvfb
Xvfb :99 -screen 0 1024x768x24 &
sleep 2

# 2. 运行评估（关键环境变量）
export DISPLAY=:99
export __GLX_VENDOR_LIBRARY_NAME=nvidia  # 强制 NVIDIA GPU 渲染

python -u scripts/eval/eval_r2r_val_unseen.py \
    --model_path /workspace/InternNav_Model \
    --gpu_id 0 \
    --output_path ./logs/eval_r2r_val_unseen \
    2>eval_stderr.log  # 分离 stderr 避免日志污染
```

## 推荐的依赖版本

| 包 | 版本 | 说明 |
|---|------|------|
| habitat-lab | 0.1.7 | YACS 配置系统 |
| habitat-sim | 0.1.7 | 需注意渲染后端 |
| numpy | 1.26.4 | 必须固定，不能用 2.x |
| transformers | 4.51.0 | 与模型 config.json 一致 |
| flash-attn | 2.8.3 | 如 GLIBC 不兼容则用 stub |
| diffusers | 0.36.0 | 需 patch 梯度检查点 |
| gym | 0.26.x | 需 patch Discrete(0) |
| dtw-python | 最新 | 安装后需重新固定 numpy |

## 脚本中 patch 的执行顺序（至关重要）

```
1. flash_attn stub 注册       ← 最先，防止 import 失败
2. numpy 兼容性补丁            ← import habitat_sim 之前
3. habitat_sim GL 上下文预初始化 ← import numba/habitat 之前
4. gym.spaces.Discrete patch   ← import habitat 之前
5. 直接加载 vln_measures       ← 绕过 __init__.py
6. build_depthanythingv2 patch ← import 模型代码之前
7. 正常 import habitat 和模型代码
```
