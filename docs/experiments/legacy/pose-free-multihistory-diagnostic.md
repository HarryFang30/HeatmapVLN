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
