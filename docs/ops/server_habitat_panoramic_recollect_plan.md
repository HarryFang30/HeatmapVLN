# Habitat Panoramic Recollection Plan

服务器验证已经确认：旧的 panoramic 数据存在结构性问题，不能作为 Stage1-S2 的有效监督数据。

结论：

- 旧数据必须隔离或删除，不要用于正式训练。
- 必须修 `collect/panoramic/collector.py` 的 episode 驱动方式。
- 必须让 `run_collect_panoramic.sh` 保存 `front_down` depth。
- 修复后先跑 20 条 audit，再正式重采。

## 1. 旧数据处理

旧目录：

```bash
/home/intern/zhr/fjl/r2r_paronamic_data
```

建议不要在旧目录上续采。旧数据里 `meta.json` 的 instruction/reference_path 和真实观测轨迹已经系统性错配，同时缺 `depth_front_down`。

建议改用新目录，顺便修正拼写：

```bash
/home/intern/zhr/fjl/r2r_panoramic_data_v2
```

如果磁盘紧张，可以先移动旧目录：

```bash
mv /home/intern/zhr/fjl/r2r_paronamic_data \
   /home/intern/zhr/fjl/r2r_paronamic_data_bad_episode_mismatch
```

## 2. 修 wrapper：保存 `front_down` depth

编辑：

```bash
/home/intern/zhr/fjl/habitat/VLN-CE/run_collect_panoramic.sh
```

把最后的调用从：

```bash
python -m collect panoramic \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU"
```

改成：

```bash
python -m collect panoramic \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU" \
  --depth-directions front front_down
```

`--depth-directions all` 也可以，但会额外保存 right/back/left depth，占用更多空间。Stage1-S2 当前最关键的是 `front_down`。

## 3. 修 collector：不要再用外部 episode 驱动 reset

核心原则：

```python
observations = env.reset()
episode = env.current_episode
```

所有 `scene_name`、`episode_id`、`instruction_text`、`trajectory_id`、`reference_path` 都必须从 `env.current_episode` 获取。

不要再使用这个模式：

```python
episode = dataset.episodes[episode_idx]
env._current_episode = episode
observations = env.reset()
```

因为 `env.reset()` 会用 Habitat 自己的 `_episode_iterator` 覆盖 `_current_episode`。

### 建议替换的主循环结构

在 `collect/panoramic/collector.py` 中，把主循环改成 reset-driven。下面是核心结构，按服务器文件实际缩进合并：

```python
seen_reset_keys = set()

while clip_id <= args.num_clips:
    try:
        observations = env.reset()
        episode = env.current_episode

        scene_name = episode.scene_id.split("/")[-1].replace(".glb", "")
        ep_key = f"{scene_name}:{episode.episode_id}"

        if ep_key in seen_reset_keys:
            print("  Stop: Habitat episode iterator cycled through all episodes")
            break
        seen_reset_keys.add(ep_key)

        if str(episode.episode_id) in collected_ids:
            print(f"  Skip already collected episode {episode.episode_id}")
            continue

        print(f"\nClip {clip_id}/{args.num_clips}  scene={scene_name}  ep={episode.episode_id}")

        ok = True
        if episode.goals is None or len(episode.goals) == 0:
            ok = False
        if episode.reference_path is None or len(episode.reference_path) == 0:
            ok = False
        if not ok:
            print("  Skip: missing goals/reference_path")
            stats["failed"] += 1
            continue

        instruction_text = ""
        if episode.instruction is not None and hasattr(episode.instruction, "instruction_text"):
            instruction_text = episode.instruction.instruction_text or ""
        trajectory_id = getattr(episode, "trajectory_id", None) or "unknown"

        sim = env.sim
        if not hasattr(env, "_last_scene") or env._last_scene != scene_name:
            follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)
            env._last_scene = scene_name

        # 后面沿用原来的 clip_dir、record_frame、ShortestPathFollower、save meta 逻辑。
        # reference_path 必须来自 reset 后的实际 episode。
        reference_path = episode.reference_path

    except Exception as e:
        ...
```

注意：如果原代码里还有 `ep_ptr`、`all_indices`、`episode = dataset.episodes[episode_idx]`，这些不应该再参与主循环。可以保留前面的 dataset/scene 统计打印，但不能用外部 episode 元数据写 `meta.json`。

### 更严格的 resume key

当前代码只用 `episode_id` 判断是否已采集。R2R 通常够用，但更稳的是用 `scene_id + episode_id`。

读取旧 meta 时可以改成：

```python
collected_ids = set()
for mf in split_dir.rglob("meta.json"):
    try:
        meta = json.load(open(mf))
        eid = meta.get("episode_id")
        scene = meta.get("scene_id")
        if eid is not None and scene is not None:
            collected_ids.add(f"{scene}:{eid}")
    except Exception:
        pass
```

主循环中对应：

```python
ep_key = f"{scene_name}:{episode.episode_id}"
if ep_key in collected_ids:
    continue
...
collected_ids.add(ep_key)
```

如果你直接使用全新输出目录，这个改动不是第一优先级；第一优先级是 reset-driven。

## 4. 修复后必须跑的 audit

先采 20 条：

```bash
su intern
cd /home/intern/zhr/fjl/habitat/VLN-CE
source /home/intern/suyihan/miniconda3/etc/profile.d/conda.sh
conda activate dataset_collect

./run_collect_panoramic.sh \
  /home/intern/zhr/fjl/r2r_panoramic_audit_v2 \
  train \
  20 \
  0 2>&1 | tee logs/audit_recollect_v2_20.log
```

然后检查 chunk key：

```bash
python - <<'PY'
from pathlib import Path
import json
import numpy as np

root = Path("/home/intern/zhr/fjl/r2r_panoramic_audit_v2/train")
clips = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())
print("clips:", len(clips))

required = {
    "rgb_front",
    "rgb_right",
    "rgb_back",
    "rgb_left",
    "rgb_front_down",
    "pose_front_down",
    "depth_front_down",
}

bad = []
for clip in clips:
    chunks = sorted((clip / "chunks").glob("*.npz"))
    if not chunks:
        bad.append((str(clip), ["NO_CHUNK"]))
        continue
    with np.load(chunks[0]) as z:
        keys = set(z.files)
    missing = sorted(required - keys)
    if missing:
        bad.append((str(clip), missing))

print("bad:", len(bad))
print("first_bad:", bad[:3])
assert not bad

for clip in clips[:3]:
    meta = json.loads((clip / "meta.json").read_text())
    print(clip, meta["scene_id"], meta["episode_id"], meta["data_format"]["depth_directions"])
PY
```

通过标准：

- 20 条都成功。
- 每条都有 `depth_front_down`。
- `meta["data_format"]["depth_directions"]` 至少包含 `front` 和 `front_down`。
- 日志里不再出现 selected episode 与 actual episode 的 mismatch。

## 5. 正式重采命令

建议先采 5000 条新数据：

```bash
./run_collect_panoramic.sh \
  /home/intern/zhr/fjl/r2r_panoramic_data_v2 \
  train \
  5000 \
  0 2>&1 | tee logs/collect_train_5000_v2.log
```

如果 5000 条训练趋势正常，再扩到 10000 或完整 R2R train instruction episodes。

## 6. 对数据量的重新判断

服务器实测平均轨迹长度约 215 帧，因此修复后 5000 条不再是“明显偏小”的规模。按 `system2_sample_step=4` 和 `system2_stop_oversample=5` 粗估，5000 条约 28-29 万 SFT 上界。

这已经足够做一轮认真的 R2R-only Stage1-S2 实验。区别是：

- 5000 条：可以判断 pipeline 和 Stage1-S2 是否有明显收益。
- 10000 条或完整 train：更适合判断最终上限和 val_unseen 稳定性。

但旧的 6764 条因为 episode 错配和缺 `front_down` depth，不能计入有效训练规模。
