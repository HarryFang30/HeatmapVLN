# HeatmapVLN Method Architecture — code-truth notes

This note records the implementation choices represented in
`heatmapvln_method_architecture.svg`. Existing architecture drawings and README
descriptions were treated as references only; the current code and the final
training/evaluation configuration were treated as authoritative.

## System-2

- Inputs are one instruction, at most eight historical panoramas, and one
  current panorama. Each panorama contains `front/right/back/left` RGB views.
- Qwen2.5-VL produces `view: ...` and `pixel: u v` as autoregressive language
  tokens. Coordinates are not produced by a regression head.
- The represented checkpoint uses rank-32, alpha-64 LoRA on
  `q_proj/k_proj/v_proj/o_proj` in all 28 LLM layers.
- The four trajectory queries are appended only after the structured goal. A
  second causal readout yields four 3584-D hidden states for System-1.
- `view: stop` bypasses System-1.

Primary code:

- `src/models/heatmap/input_constructor.py`
- `src/data/panoramic_tokenized_collator.py`
- `src/models/qwen2_5_vl/integration.py`
- `src/data/sliding_window_dataset.py`
- `src/data/trajectory_dataset.py`

## Auxiliary heatmap branch

- This is an optional spatial-grounding/pretraining branch, not an online input
  to the deployed System-2 → System-1 route.
- ViT hooks use layers `7/15/23/31`; LLM hooks use layers `6/13/20`.
- The production legacy decoder uses two-layer, four-head
  `TrajectoryGuidedAttention`, followed by the 513-channel fine-localization
  decoder and 64×64 four-view heatmaps.
- Depth and intrinsics are used for geometric supervision rather than as image
  inputs. Historical camera poses are converted to a 4-D relative-pose signal
  and enter `TrajectoryGuidedAttention`.

Primary code:

- `src/models/heatmap/heatmap_vln.py`
- `src/models/heatmap/dpt_lite_fusion.py`
- `src/models/heatmap/trajectory_attention.py`
- `src/models/heatmap/fine_localization.py`
- `src/models/heatmap/heatmap_vln_loss.py`
- `configs/train_heatmap_config.yaml`

## System-2 → System-1 bridge

- The final h1024 route uses `PanoLatentSpaceAdapter`:
  `3584 → 1024 → 3584`, GELU, and a residual identity connection.
- A separate frozen condition projector maps `3584 → 768 → 768`.
- The bridge receives no explicit view embedding, pixel-coordinate embedding,
  geometry token, or learned “view adjust” signal.
- In Stage-3 only the h1024 adapter is trainable.

Primary code:

- `src/models/adapters/pano_latent_adapter.py`
- `src/models/pipeline.py`
- `src/models/action/nextdit_action_head.py`
- `configs/train_stage3_pano_system1_h1024_8gpu.yaml`

## System-1

- The independent visual-memory path uses a 224×224 front-down image pair. The
  deployed `first_only` path duplicates the same observation for anchor and
  current slots.
- Frozen DINOv2 ViT-S/14 features pass through a three-layer Memory Encoder and
  a 32-query, three-layer Q-Former, producing 32 visual tokens of width 768.
- Four goal tokens are concatenated with 32 visual tokens.
- NextDiT uses 12 blocks of width 384 and six heads. Its self-attention and
  gated cross-attention branches run in parallel from the same normalized action
  state before they are merged; they are not a serial stack.
- The model predicts a velocity field for 32 delta poses
  `[Δx, Δy, Δyaw]`. Evaluation performs 10 Euler updates for 32 candidates and
  sends the mean XY path to the downstream executor.

Primary code:

- `src/models/action/nextdit_action_head.py`
- `src/models/action/nextdit/components.py`
- `src/models/action/nextdit/nextdit_traj.py`
- `scripts/training/train_loop.py`
- `scripts/evaluation/rpc_model_server.py`

## Deliberately excluded from the main route

- `GeometryAwarePanoToNextDiTAdapter`
- learned `View Adjust`
- `llm_projector 3584 → 896`
- future/pixel-goal RGB as the System-1 anchor
- heatmap output as an input to the bridge
- a serial self-attention → cross-attention depiction inside NextDiT
