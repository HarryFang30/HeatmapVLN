"""Check a flash-attn build against SDPA on the deployed Qwen2.5-VL vision tower.

Three questions, answered on the machine the build is for:

1. Does it import, and does transformers 4.51.0 see it (``is_flash_attn_2_available``,
   and modeling_qwen2_5_vl's module-level ``flash_attn_varlen_func`` / rotary import)?
2. Kernel: ``flash_attn_varlen_func`` against SDPA with the block mask, on the real
   window partition of the turn-2 shape, forward and backward.  Both are measured
   against an fp32 SDPA reference, so the question is "is flash-attn as close to exact
   as SDPA is", not "is it bit-identical to SDPA" -- it is not, nor should it be.
3. The real vision tower (InternNav_Model ``visual.*`` weights) at both deployed
   shapes: SDPA with upstream's mask loop, SDPA with the repo's window-id mask patch,
   and flash_attention_2 -- same weights, same input -- against an fp32 SDPA reference,
   with the time per tower pass of each (median of 10 after a warm-up).

The three implementations are switched by swapping each block's attention class:
``Qwen2_5_VLVisionSdpaAttention`` and ``Qwen2_5_VLVisionFlashAttention2`` share the
same parameters and differ only in ``forward``, so one set of weights serves all three.

Usage (inside the 4090 container, flash-attn on PYTHONPATH):
    PYTHONPATH=/workspace/fa_build/site CUDA_VISIBLE_DEVICES=5 \\
        /opt/conda/bin/python scripts/tools/check_flash_attn_vision.py --out fa_check.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
# Front views 384x384 and the lookdown 640x480, rounded to multiples of 28 by the
# processor: K=8 history + current is 9 front grids, turn 2 adds the lookdown.
SHAPES = {
    "turn1_7056x9": [[1, 28, 28]] * 9,
    "turn2_8620x10": [[1, 28, 28]] * 9 + [[1, 34, 46]],
}


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "rel_l2": ((a - b).norm() / b.norm()).item(),
        "min_token_cos": F.cosine_similarity(a, b, dim=-1).min().item(),
    }


def _timed(fn, n: int = 10) -> dict:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": round(statistics.median(ts), 2), "min_ms": round(min(ts), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/workspace/InternNav_Model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import flash_attn
    import flash_attn_2_cuda
    import transformers
    from flash_attn import flash_attn_varlen_func
    from safetensors import safe_open
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
    from transformers.utils import is_flash_attn_2_available

    # Loaded by path so the check does not import the whole src.models package.
    spec = importlib.util.spec_from_file_location(
        "qwen2_5_vl_vision_mask", REPO / "src/models/qwen2_5_vl_vision_mask.py"
    )
    mask_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_mod)

    out = {
        "flash_attn": flash_attn.__version__,
        "flash_attn_file": flash_attn.__file__,
        "ext_file": flash_attn_2_cuda.__file__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_sees_flash_attn_2": bool(is_flash_attn_2_available()),
        "modeling_has_varlen": M.flash_attn_varlen_func is not None,
        "modeling_has_rotary": M.apply_rotary_emb is not None,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    print(json.dumps(out, indent=1), flush=True)
    dev = torch.device("cuda:0")

    vcfg = M.Qwen2_5_VLVisionConfig(**json.load(open(f"{args.model}/config.json"))["vision_config"])
    index = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    state = {}
    for shard in sorted({f for k, f in index.items() if k.startswith("visual.")}):
        with safe_open(f"{args.model}/{shard}", framework="pt") as fh:
            for k in fh.keys():
                if k.startswith("visual."):
                    state[k[len("visual."):]] = fh.get_tensor(k)

    def tower(dtype):
        m = M.Qwen2_5_VisionTransformerPretrainedModel._from_config(
            vcfg, attn_implementation="sdpa", torch_dtype=dtype
        )
        m.load_state_dict(state, strict=True)
        return m.to(device=dev, dtype=dtype).eval()

    attn_cls = {
        "sdpa": M.Qwen2_5_VLVisionSdpaAttention,
        "flash_attention_2": M.Qwen2_5_VLVisionFlashAttention2,
    }

    def set_impl(m, impl):
        for blk in m.blocks:
            blk.attn.__class__ = attn_cls[impl]

    # --- kernel, on the real window partition of the turn-2 shape ---------------------
    torch.manual_seed(0)
    m16 = tower(torch.bfloat16)
    _, cu = m16.get_window_index(torch.tensor(SHAPES["turn2_8620x10"], device=dev))
    cu = torch.unique_consecutive(torch.tensor(cu, device=dev, dtype=torch.int32))
    bounds = cu.tolist()
    seq, heads = bounds[-1], vcfg.num_heads
    head_dim = vcfg.hidden_size // heads
    q, k, v = (
        torch.randn(seq, heads, head_dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    mask = mask_mod._window_mask(bounds, seq, dev)

    def sdpa(qq, kk, vv):
        t = lambda x: x.transpose(0, 1)[None]  # noqa: E731
        return F.scaled_dot_product_attention(t(qq), t(kk), t(vv), mask)[0].transpose(0, 1)

    max_len = max(bounds[i] - bounds[i - 1] for i in range(1, len(bounds)))
    o_fa = flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len)
    o_sd = sdpa(q, k, v)
    o_32 = sdpa(q.float(), k.float(), v.float())
    g = torch.randn_like(o_fa)
    g_fa = torch.autograd.grad(o_fa, (q, k, v), g)
    g_sd = torch.autograd.grad(o_sd, (q, k, v), g)
    g_32 = torch.autograd.grad(o_32, (q, k, v), g.float())
    kernel = {
        "seq": seq, "windows": len(bounds) - 1, "heads": heads, "head_dim": head_dim,
        "fwd": {"flash_attn": _cmp(o_fa, o_32), "sdpa_bf16": _cmp(o_sd, o_32)},
        "dq": {"flash_attn": _cmp(g_fa[0], g_32[0]), "sdpa_bf16": _cmp(g_sd[0], g_32[0])},
        "dk": {"flash_attn": _cmp(g_fa[1], g_32[1]), "sdpa_bf16": _cmp(g_sd[1], g_32[1])},
        "dv": {"flash_attn": _cmp(g_fa[2], g_32[2]), "sdpa_bf16": _cmp(g_sd[2], g_32[2])},
    }
    print(json.dumps({"kernel": kernel}, indent=1), flush=True)
    del q, k, v, o_fa, o_sd, o_32, g, g_fa, g_sd, g_32, mask
    torch.cuda.empty_cache()

    # --- the real vision tower ---------------------------------------------------------
    m32 = tower(torch.float32)
    upstream = M.Qwen2_5_VLVisionSdpaAttention.forward
    patched = mask_mod._patched_forward(upstream)
    towers = {}
    try:
        for name, thw in SHAPES.items():
            grid = torch.tensor(thw, device=dev)
            n = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
            torch.manual_seed(1)
            px = torch.randn(n, vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2, device=dev)
            px16 = px.bfloat16()
            run = lambda: m16(px16, grid_thw=grid)  # noqa: E731
            r = {"patches": n}
            with torch.no_grad():
                M.Qwen2_5_VLVisionSdpaAttention.forward = upstream
                ref = m32(px, grid_thw=grid)
                set_impl(m16, "sdpa")
                a = run()
                r["sdpa_upstream"] = {"vs_fp32": _cmp(a, ref), "time": _timed(run)}
                M.Qwen2_5_VLVisionSdpaAttention.forward = patched
                p = run()
                r["sdpa_mask_patch"] = {"bit_identical_to_upstream": bool(torch.equal(p, a)), "time": _timed(run)}
                M.Qwen2_5_VLVisionSdpaAttention.forward = upstream
                set_impl(m16, "flash_attention_2")
                f = run()
                r["flash_attention_2"] = {"vs_fp32": _cmp(f, ref), "vs_sdpa_bf16": _cmp(f, a), "time": _timed(run)}
                set_impl(m16, "sdpa")
            towers[name] = r
            print(json.dumps({name: r}, indent=1), flush=True)
    finally:
        M.Qwen2_5_VLVisionSdpaAttention.forward = upstream

    out.update(kernel=kernel, tower=towers)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
