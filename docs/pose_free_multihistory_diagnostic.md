# Pose-free multi-history diagnostic branch

## Scientific contract

The diagnostic model receives a current four-view panorama and `K` historical
panoramas.  It predicts one four-view heatmap and one four-view visibility
vector for every history item:

```text
current VLM patches [B, 4, hc, wc, C]
history VLM queries [B, K, Cq]
                  │
                  ▼
shared normalized query × patch matching
                  │
                  ├── heatmaps   [B, K, 4, H, W]
                  └── visibility [B, K, 4]
```

The main diagnostic branch must not receive relative pose, coordinates,
distance, bearing, or a history-slot embedding.  Exact relative pose may still
be used to create labels and to run a separately named geometry-only baseline.

`PoseFreeHistoryMatcher` enforces the model-side part of this contract:

- relative pose is absent from its API and state;
- every history uses the same projections and visibility readout;
- no operation mixes history items;
- permuting histories permutes predictions in exactly the same way;
- changing one history query cannot change another history prediction;
- padded history slots are explicitly zeroed and returned with a mask;
- the 3584-to-64 default head has fewer than 0.5M trainable parameters.
- pose-free integration registers no ViT hooks, so backbone-gradient training
  does not retain an unused ViT computation graph.

These architectural properties are necessary but not sufficient for visual
grounding.  The dataset must independently randomize history order and balance
target direction/distance, and the evaluation must include single-anchor swaps,
current-image shuffles, and history-image shuffles.

## Current-stack audit

The existing stack is not yet a valid implementation of this experiment:

1. `TrajectoryGuidedAttention` consumes `history_rel_poses`, so its score cannot
   be primarily attributed to visual reasoning.
2. The legacy decoder contains two DPT fusion blocks, a transformer coarse
   module, and a deconvolutional fine head.  It is large enough to learn strong
   task priors independently of the VLM.
3. Compact `FeatureExtractor` capture retains current-view patch tokens and
   history text-anchor states, but discards direct historical visual patches.
4. A history anchor is correctly placed after its four historical images, but
   Qwen is causal: anchor `i` also sees all earlier historical panoramas.  The
   anchor text additionally contains the history number.  Therefore the VLM
   query itself is not structurally single-history-independent.
5. Pipeline and Qwen integration already skip HeatmapVLN hooks for Stage2 input
   without panoramic histories.  A diagnostic-only decoder can therefore be
   selected without changing Stage2/Stage3 input or output contracts.

## Minimal safe integration

Keep the production default unchanged and add an explicit diagnostic decoder
mode, for example `heatmap_decoder_mode: legacy | pose_free_matcher`, defaulting
to `legacy`.

For compact panoramic batches, the pose-free path should use the deepest hooked
LLM current-view tensor directly:

```python
current_patches = llm_layer_tensors[max(self.llm_layer_indices)]
history_queries, history_mask = pad_history_queries(history_queries_batch)
result = self.pose_free_matcher(
    current_patches=current_patches,
    history_queries=history_queries,
    history_mask=history_mask,
)
```

The pose-free path must raise if its outer caller supplies
`history_rel_poses`; silently accepting and ignoring pose makes experiment
configuration errors difficult to detect.  The legacy path can continue to use
pose unchanged.  The result already exposes the existing public keys
`heatmaps` and `visibility`, so no downstream Stage2/Stage3 I/O change is
required.

For the first pilot, train and evaluate only fixed `K=4` diagnostic batches.
The existing variable-length helper remains available for smoke tests and
padding-contract checks.

## Query isolation follow-up

The first integration can reuse the existing post-history anchor queries, but
the attribution report must disclose their causal-prefix limitation.  The
stronger version should remove history numbers from anchor text, randomize
history order on every presentation, and capture or pool the four visual-token
grids belonging to each history independently.  That change should be opt-in
for the diagnostic collator and feature capture; it must not alter the Stage2
front-view/video prompt.

## Kill-or-continue checks

Proceed to joint Stage1-S2 retention training only if all of these hold on a
scene-disjoint, direction-balanced set:

- learned LoRA + shared matcher beats frozen VLM + the same matcher;
- disabling trained LoRA while keeping the head fixed causes a material drop;
- swapping only history `i` breaks output `i` while leaving other outputs
  stable;
- current-image shuffle and history-image shuffle both cause material drops;
- every target-direction and lag bin contributes successes;
- a no-image/blank prior and a geometry-only baseline do not explain the gain.

## Strict-B=1 pilot result (2026-07-14)

The original four-chain `B=4` execution was invalidated after identical blank
chains produced stable row-specific outputs.  The corrected pilot executes
four physically separate `B=1` Qwen forwards and regroups their outputs only
afterward.  The clean execution passed all structural gates:

- blank inputs, raw queries, projected queries, current patches, visibility,
  and heatmaps were bitwise identical across the four calls;
- reversing the four histories and undoing the permutation recovered both
  visibility and heatmaps bitwise exactly for all 40 validation samples;
- swapping history `i` changed only output `i`; all 480 untargeted comparisons
  had exactly zero heatmap, visibility, and peak displacement;
- heatmap loss reached all 168 trainable LoRA tensors through layer 20, while
  the head-only branch had exactly zero LoRA gradient and drift.

The bounded 512-step (four-epoch) anchor-token pilot nevertheless failed the
predeclared grounding gate.  On 40 scene-disjoint source samples (160 history
targets), the jointly trained head and LoRA obtained:

```text
anchor identity             0.21875  (chance = 0.25)
visible-view accuracy       0.09375
conditional PCK@8           0.33750
true joint PCK@8            0.03750
standard - history shuffle -0.05000  identity
standard - current shuffle -0.03125  true joint PCK@8
standard - targeted swap   -0.02500  identity
```

The factorial attribution also failed: joint minus head-only identity was
`0.0000` (95% paired-bootstrap CI `[-0.06875, 0.06875]`), trained versus
Stage1-S2 LoRA under the same joint head was `-0.00625`
(`[-0.04375, 0.03125]`), and joint LoRA transferred to the head-only head was
only `+0.01875` (`[-0.05000, 0.08125]`).  Thus this checkpoint must not be used
as evidence that history-grounding ability entered the VLM, and it must not
gate a new Stage2/Stage3 causal comparison.

The final query audit narrows the failure.  Joint LoRA increased raw history
query separation (mean off-diagonal cosine `0.8544 -> 0.5889`; Euclidean
distance `91.61 -> 153.43`), but the resulting heatmaps remained highly
similar (cosine `0.9484`) and the differences were not aligned with the true
history identity.  The failure is therefore not missing gradient or occurrence
mapping.  The anchor-token objective learned differences without the required
semantic correspondence.

The next bounded route is to pool each history panorama's layer-20 visual
tokens instead of using the trailing text-anchor token, and to add a
target-grounded K-way identity loss after the four `B=1` outputs are regrouped.
Warm up the shared head, freeze it, and then train LoRA only; compare trained
and Stage1-S2 LoRA through that same frozen head before any downstream run.
