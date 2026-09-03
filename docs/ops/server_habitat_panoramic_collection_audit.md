# Habitat Server Panoramic Collection Audit

本文用于在服务器的 Habitat / VLN-CE 环境里验证一件事：当前 `run_collect_panoramic.sh` 采出来的数据，是否真正满足 HeatmapVLN Stage1-S2 panoramic SFT 的训练要求。

重点不是先扩大数据量，而是先确认两个基础条件：

1. `env.reset()` 后实际加载的 episode 是否和脚本选中的 episode 一致。
2. 采集结果里是否保存了 Stage1-S2 需要的 `front_down` depth。

如果这两项有问题，Stage1-S2 或 Stage2 都无法从训练阶段自动补回来。

## 0. 已知上下文

服务器采集入口：

```bash
su intern
cd /home/intern/zhr/fjl/habitat/VLN-CE
set -o pipefail
source /home/intern/suyihan/miniconda3/etc/profile.d/conda.sh
conda activate dataset_collect

./run_collect_panoramic.sh \
  /home/intern/zhr/fjl/r2r_paronamic_data \
  train \
  5000 \
  0 2>&1 | tee /home/intern/zhr/fjl/habitat/VLN-CE/logs/collect_train_5000.log
```

当前 wrapper 最后实际调用：

```bash
python -m collect panoramic \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU"
```

Stage1-S2 训练配置依赖：

```yaml
panoramic_vlm_input: true
load_depth: true
load_lookdown_for_system2: true
compute_pixel_goal: true
system2_sft_protocol: internnav
pixel_goal_direction: front_down
system2_min_pixel_goal_len: 3
system2_sample_step: 4
system2_stop_oversample: 5
```

因此，训练端会用 `front_down` 视角计算 pixel-goal，并且期望最好能读到 `front_down` depth 做遮挡过滤。

## 1. 先确认服务器代码版本

在服务器上运行：

```bash
su intern
cd /home/intern/zhr/fjl/habitat/VLN-CE
source /home/intern/suyihan/miniconda3/etc/profile.d/conda.sh
conda activate dataset_collect

sed -n '1,220p' run_collect_panoramic.sh
sed -n '55,90p' collect/panoramic/collector.py
sed -n '160,215p' collect/panoramic/collector.py
```

重点看三处：

- `run_collect_panoramic.sh` 有没有传 `--depth-directions front front_down` 或 `--depth-directions all`。
- `collector.py` 里 `--depth-directions` 的默认值是不是 `["front"]`。
- `collector.py` 里是否存在 `env._current_episode = episode` 后马上 `env.reset()` 的逻辑。

如果 wrapper 没传 `--depth-directions`，并且默认值是 `["front"]`，则默认不会保存 `front_down` depth。

## 2. 验证采集结果是否缺 `front_down` depth

在服务器上运行下面脚本，直接检查已经采出来的 `.npz` chunk：

```bash
cd /home/intern/zhr/fjl/habitat/VLN-CE
source /home/intern/suyihan/miniconda3/etc/profile.d/conda.sh
conda activate dataset_collect

python - <<'PY'
from pathlib import Path
import json
import numpy as np

root = Path("/home/intern/zhr/fjl/r2r_paronamic_data/train")
clips = sorted(root.glob("*/*"))
clips = [p for p in clips if p.is_dir() and (p / "meta.json").exists()]

print("clip_count:", len(clips))
if not clips:
    raise SystemExit("No clips found")

need_keys = {
    "rgb_front",
    "rgb_right",
    "rgb_back",
    "rgb_left",
    "rgb_front_down",
    "pose_front_down",
    "depth_front_down",
}

checked = 0
missing_summary = {}
for clip in clips[:50]:
    chunk_files = sorted((clip / "chunks").glob("*.npz"))
    if not chunk_files:
        print("NO_CHUNKS", clip)
        continue

    with np.load(chunk_files[0]) as z:
        keys = set(z.files)

    missing = sorted(need_keys - keys)
    if missing:
        missing_summary[str(clip)] = missing

    if checked < 5:
        print("\nCLIP:", clip)
        print("chunk:", chunk_files[0].name)
        print("keys:", sorted(keys))
        if missing:
            print("missing:", missing)
        else:
            print("missing: []")

    checked += 1

print("\nchecked:", checked)
print("clips_with_missing_required_keys:", len(missing_summary))
if missing_summary:
    print("first_missing:", next(iter(missing_summary.items())))
PY
```

判断标准：

- 如果缺 `rgb_front_down`：lookdown 图像没采到，Stage1-S2 的输入不完整。
- 如果有 `rgb_front_down` 和 `pose_front_down`，但缺 `depth_front_down`：训练可以跑，但 pixel-goal 的遮挡过滤会退化，标签质量下降。
- 如果 `depth_front_down` 存在：这个风险基本解除。

## 3. 验证 `env.reset()` 是否发生 episode 错配

这是最关键的完整性检查。建议先不要采 5000，先跑 20 条小采样。

在 `collect/panoramic/collector.py` 的：

```python
env._current_episode = episode
observations = env.reset()
```

后面临时加上：

```python
actual = env.current_episode
selected_id = str(getattr(episode, "episode_id", ""))
actual_id = str(getattr(actual, "episode_id", ""))
selected_scene = str(getattr(episode, "scene_id", ""))
actual_scene = str(getattr(actual, "scene_id", ""))

print(
    "EPISODE_RESET_CHECK",
    "selected_id=", selected_id,
    "actual_id=", actual_id,
    "selected_scene=", selected_scene,
    "actual_scene=", actual_scene,
    flush=True,
)

if selected_id != actual_id or selected_scene != actual_scene:
    raise RuntimeError(
        "Episode mismatch after env.reset(): "
        f"selected_id={selected_id}, actual_id={actual_id}, "
        f"selected_scene={selected_scene}, actual_scene={actual_scene}"
    )
```

然后跑小规模：

```bash
cd /home/intern/zhr/fjl/habitat/VLN-CE
source /home/intern/suyihan/miniconda3/etc/profile.d/conda.sh
conda activate dataset_collect

./run_collect_panoramic.sh \
  /home/intern/zhr/fjl/r2r_paronamic_audit_reset \
  train \
  20 \
  0 2>&1 | tee logs/audit_reset_20.log
```

判断标准：

- 如果没有报错，并且日志中每条 `selected_id == actual_id`、`selected_scene == actual_scene`：episode reset 风险基本解除。
- 如果报错：当前采集逻辑不能信，之前采出来的训练数据很可能 instruction/reference_path 和实际图像轨迹错配。

如果触发错配，优先修采集逻辑。不要指望 Stage1-S2 或 Stage2 在训练时补救。

## 4. 修正 `front_down` depth 采集

如果第 2 节确认缺 `depth_front_down`，修改 `run_collect_panoramic.sh` 的最后调用：

```bash
python -m collect panoramic \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU" \
  --depth-directions front front_down
```

也可以用：

```bash
--depth-directions all
```

但 `all` 会保存 front/right/back/left/front_down 的 depth，更占磁盘。Stage1-S2 当前最关键的是 `front_down` depth；保留 `front` depth 也有利于兼容其他实验。

修完后跑 5 条小采样：

```bash
./run_collect_panoramic.sh \
  /home/intern/zhr/fjl/r2r_paronamic_audit_depth \
  train \
  5 \
  0
```

然后重新执行第 2 节的 key 检查，把 root 改成：

```python
root = Path("/home/intern/zhr/fjl/r2r_paronamic_audit_depth/train")
```

确认 `depth_front_down` 已存在后再正式采集。

## 5. 检查 `meta.json` 是否能被 Stage1-S2 正常使用

Stage1-S2 当前会读取 `meta.json` 里的 `instruction` 作为唯一语言监督。`trajectory_id` 可用于数据审计，但不会参与训练文本生成。

运行：

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/home/intern/zhr/fjl/r2r_paronamic_data/train")
clips = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())

empty_instruction = []
for clip in clips[:200]:
    meta = json.loads((clip / "meta.json").read_text())
    instr = meta.get("instruction", "")
    if not instr:
        empty_instruction.append(str(clip))

print("checked:", min(200, len(clips)))
print("empty_instruction:", len(empty_instruction), empty_instruction[:5])
PY
```

判断标准：

- `empty_instruction > 0`：这些样本不能作为有效 SFT 数据。

## 6. 检查 action 和轨迹长度

运行：

```bash
python - <<'PY'
from pathlib import Path
import json
import numpy as np

root = Path("/home/intern/zhr/fjl/r2r_paronamic_data/train")
clips = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())

lengths = []
bad = []
for clip in clips[:500]:
    meta = json.loads((clip / "meta.json").read_text())
    traj = np.load(clip / "trajectory_3d.npy")
    acts = np.load(clip / "discrete_actions.npy")
    lengths.append(len(traj))
    if len(traj) != int(meta.get("num_frames", -1)):
        bad.append((str(clip), "num_frames", len(traj), meta.get("num_frames")))
    if len(acts) not in {len(traj), max(0, len(traj) - 1)}:
        bad.append((str(clip), "actions_len", len(acts), "traj_len", len(traj)))

print("checked:", min(500, len(clips)))
print("min_len:", min(lengths) if lengths else None)
print("max_len:", max(lengths) if lengths else None)
print("avg_len:", sum(lengths) / len(lengths) if lengths else None)
print("bad:", bad[:10], "count=", len(bad))
PY
```

Stage1-S2 的有效样本数量强依赖轨迹长度。`system2_sample_step: 4` 会每隔 4 帧抽一次，短轨迹贡献很少。

## 7. 估算 5000 条大概能产生多少 Stage1-S2 SFT 样本

这是粗略估算，不等同于训练端最终样本数，因为 pixel-goal 是否可见还要经过投影和遮挡判断。

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/home/intern/zhr/fjl/r2r_paronamic_data/train")
clips = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())

sample_step = 4
min_history = 5
stop_oversample = 5

est_total = 0
lengths = []
for clip in clips:
    meta = json.loads((clip / "meta.json").read_text())
    n = int(meta.get("num_frames", 0))
    lengths.append(n)
    temporal = max(0, len(range(min_history, max(min_history, n - 1), sample_step)))
    est_total += temporal + stop_oversample

print("clips:", len(clips))
print("avg_frames:", sum(lengths) / len(lengths) if lengths else None)
print("estimated_upper_bound_sft_samples:", est_total)
print("estimated_samples_per_clip:", est_total / len(clips) if clips else None)
PY
```

经验判断：

- 小于 50k SFT samples：主要只能验证格式和 loss，grounding 不可靠。
- 50k 到 150k：可以做第一轮训练，但 val_unseen 波动可能很大。
- 150k 到 350k：比较像 R2R-only Stage1-S2 的认真训练规模。
- 350k 以上：更接近稳定训练，但仍明显小于 InternNav 论文的多数据源规模。

5000 条 instruction episode 通常大约落在 8 万到 12 万级别 SFT samples，前提是轨迹长度正常、episode 对齐正确、pixel-goal label 质量正常。

## 8. 最终判定表

| 检查项 | 通过标准 | 如果失败 |
|---|---|---|
| Episode reset 对齐 | `selected_id == actual_id` 且 scene 一致 | 采集数据可能错配，必须修 collector 后重采 |
| `rgb_front_down` | chunk key 中存在 | lookdown 输入缺失，Stage1-S2 不成立 |
| `pose_front_down` | chunk key 中存在 | 无法在 lookdown 视角可靠投影 pixel-goal |
| `depth_front_down` | chunk key 中存在 | 不会崩，但 pixel-goal 遮挡过滤退化，建议重采 |
| instruction | 非空 | SFT prompt 无效 |
| 样本数量 | 至少 50k SFT samples，正式建议 150k+ | 只能做 smoke/debug，不适合下结论 |

## 9. 推荐执行顺序

1. 不改任何代码，先跑第 2 节检查已有数据 key。
2. 给 collector 加第 3 节的 reset 断言，跑 20 条 audit。
3. 如果 reset 错配，先修 collector，之前数据不要继续用于正式训练判断。
4. 如果缺 `depth_front_down`，改 wrapper 加 `--depth-directions front front_down`。
5. 跑 5 条 audit，确认 `depth_front_down` 出现在 chunk 中。
6. 跑第 5、6、7 节检查 meta、长度和估算样本数。
7. 全部通过后，再考虑采 5000 / 10000 / full train。

## 10. 对 5000 条的结论口径

如果 episode 对齐和 `front_down` depth 都通过，5000 条可以作为 Stage1-S2 的第一轮训练数据，用来确认：

- SFT 格式正确；
- loss 能下降；
- 模型能学到 STOP / turn / lookdown pixel-goal 的基本输出格式；
- val_seen 是否出现初步改善。

但 5000 条不适合用来判断 Stage1-S2 的上限，也不适合证明“前向图到全景图的风险已经完全被弥补”。正式 R2R-only 建议至少 10000 条，最好采完整 R2R train instruction episodes。

如果 episode reset 错配或缺 `front_down` depth，则 5000 条不是“少但可用”，而是“监督信号有结构性问题”。这类问题需要重采，不能靠 Stage2 修。
