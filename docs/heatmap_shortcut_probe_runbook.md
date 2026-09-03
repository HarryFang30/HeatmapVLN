# History Head shortcut probes: full / vision-only / pose-only / no-input

## What this answers

The History Head predicts, for every past observation, a four-direction
heatmap of where that observation now lies relative to the current view. It
receives three things: the current front image, one front image per history,
and the relative pose of each history. Relative pose alone almost determines
the answer geometrically, so a good score is not by itself evidence that the
head reads the images.

These probes separate the two. One frozen backbone, one byte-identical fresh
head and one training budget are shared by four input regimes:

| Regime | Current + history images | Relative pose |
|---|---|---|
| `full` | real | real |
| `vision-only` | real | constant |
| `pose-only` | all-black | real |
| `no-input` | all-black | constant |

`no-input` measures what the label prior alone buys, so it is the floor every
other regime is read against. `pose-only` measures the geometric shortcut.
`vision-only` measures whether the images carry the answer at all.
`full` measures the two together.

**The claim this can support** is that the task is *identifiable from vision*
under a matched budget, not that the deployed head does in fact use vision.
The second claim needs the intervention arm below, run against a trained head.

## Pose ablation is a constant, not a removal

The deployed head (`internnav_single_view`) has no representation for an
absent pose; the trajectory attention always consumes a `[K, 4]` tensor. The
ablation therefore feeds the identity relative pose `(0, 0, cos 0, sin 0)` in
every slot. It is in-distribution and carries no information about where any
history lies, which is what the ablation needs. The legacy panoramic head,
which does accept `None`, keeps its original behaviour.

## Interventions on the trained `full` probe

`full` is additionally evaluated, without retraining, under six perturbations:

| Condition | What changes | Reads as |
|---|---|---|
| `zero-pose` | pose replaced by the constant | pose dependence at test time |
| `blank-images` | all RGB zeroed | image dependence at test time |
| `history-shuffle` | history images reversed, poses and targets untouched | whether slot *k*'s answer uses slot *k*'s image |
| `current-shuffle` | current image taken from another sample | whether the answer is anchored in the current view |
| `pose-conflict` | poses rolled by one slot | pose dependence, without changing the image set |
| `pose-conflict-shifted-target` | poses **and** targets rolled together | control: a pose-driven head should score near baseline here |

## Matched contract

`summarize_heatmap_shortcuts.py` refuses to tabulate unless every probe agrees
on architecture, seed, initial head hash, trainable head size, train budget,
and the exact ordered train and validation sample sets, with zero trainable
backbone tensors and no adapter tensors on the single-view backbone. Train and
validation scenes are disjoint by the dataset's deterministic MD5 scene split
(54 train scenes / 5177 clips, 7 val scenes / 823 clips on the random-walk
corpus).

## Budget and why

Measured on one C500 at `K=8`: about 1.4 s per training step and 1.5 s per
evaluated sample. The default is 12000 steps at batch 1 with each selected
sample presented exactly once, plus 400 validation samples. `full` also pays
for six extra conditions, so it is the slowest probe at roughly six hours.

12000 updates is the same order as the production head's optimizer-step count,
so these probes are undertrained in samples seen rather than in gradient steps
taken. Read them as a learning-rate-of-acquisition comparison between input
regimes, not as achievable ceilings.

## Website submission (8 GPUs)

Four regimes times two seeds fills eight GPUs exactly. The second seed is what
separates a real gap from probe noise. Resubmitting the identical command
skips any probe whose `report.json` already exists.

```bash
cd /mnt/afs/liwenhao/agent/370910109/HeatmapVLN

export SHORTCUT_ARCHITECTURE=internnav_single_view
export SHORTCUT_CONFIG=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/configs/train_heatmap_internnav_single_view_8gpu.yaml
export SHORTCUT_DATA_ROOT=/mnt/afs/liwenhao/agent/370910109/data/heatmap_randomwalk_train_v1
export SHORTCUT_OUTPUT_ROOT=/mnt/afs/liwenhao/agent/370910109/model/heatmap_shortcut_probe_v1
export INTERNNAV_MODEL_PATH=/mnt/afs/liwenhao/agent/370910109/InternNav-Model

export SHORTCUT_SEEDS=42,1337
export SHORTCUT_NUM_HISTORY=8
export SHORTCUT_TRAIN_STEPS=12000
export SHORTCUT_TRAIN_SAMPLES=12000
export SHORTCUT_VAL_SAMPLES=400
export SHORTCUT_GPU_DEVICES=0,1,2,3,4,5,6,7

bash scripts/run_heatmap_shortcut_diagnostic_8gpu_mxc500.sh
```

## Output

```
<SHORTCUT_OUTPUT_ROOT>/seed_<seed>/<mode>/report.json   per-probe metrics + contract
<SHORTCUT_OUTPUT_ROOT>/seed_<seed>/<mode>/head_final.pth trained probe head
<SHORTCUT_OUTPUT_ROOT>/seed_<seed>/task3_summary.csv     one row per mode/condition
<SHORTCUT_OUTPUT_ROOT>/seed_<seed>/task3_summary.json    rows plus deltas vs full
<SHORTCUT_OUTPUT_ROOT>/_logs/<run tag>/                  per-probe stdout
```

Primary metrics: `joint_pck8` and `joint_median_pixel_error` count a wrong
view as a failed localization, so they are the honest end-to-end numbers.
`visible_view_accuracy` and `visibility_auroc` isolate the which-direction
decision; `pck8` and `median_pixel_error` isolate where-in-the-image accuracy
on views the ground truth marks visible.

## Reading the result

- `vision-only` at the `no-input` floor means the images carry nothing the
  head can use under this budget. Section 3.3 must then stay with the
  conservative wording.
- `vision-only` clearly above `no-input`, and `full` above `pose-only`, is the
  evidence needed to say the head localizes history from vision.
- `pose-only` near `full` with `vision-only` at the floor means the head is a
  pose projector and the visual claim is unsupported.
- Always report both seeds. A gap smaller than the seed-to-seed spread is not
  a result.

## Legacy panoramic arm

The original LoRA panoramic stack is still runnable and unchanged. It needs a
frozen Stage1-S2 checkpoint carrying all 224 LoRA tensors:

```bash
export SHORTCUT_ARCHITECTURE=legacy_panoramic
export SHORTCUT_CONFIG=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN/configs/train_config_internnav_8gpu.yaml
export SHORTCUT_CHECKPOINT=<stage1_s2 run>/checkpoints/latest.pth
```

It measures a stack the current method section no longer describes, so it is
not a substitute for the single-view arm.
