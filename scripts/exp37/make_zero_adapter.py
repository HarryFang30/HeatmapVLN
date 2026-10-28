#!/usr/bin/env python3
"""EXP-37: the zero adapter for the equivalence check (ledger EXP-37 设置 等价检查).

Writes a PEFT LoRA directory with X37's configuration (r 64, alpha 128, dropout 0.05, the trainer's
``LORA_TARGET_PATTERN``, which is the ``target_modules`` regex of X35's adapter_config.json) whose
``lora_B`` tensors are exactly 0, so the adapter-on model is the base bit for bit and a server in
``base-stop-v1`` mode with it must answer every call as B0's server does.  It is built the way
``scripts/train_system2_lora.py`` builds X37's step-0 adapter: ``seed_everything(seed)``, ``load_system2``,
``attach_new_lora`` (PEFT's default init: ``lora_A`` Kaiming-uniform from the CPU generator, ``lora_B``
zeros), so ``lora_A`` is the trainer's initial draw for the same seed as far as loading the model draws
nothing from that generator on either device (not checked against a trainer step-0 file: none is kept).
Nothing is set by hand: ``lora_B`` is checked to be zero, not made so.

Then, unless ``--no-server-check``, the saved directory is loaded onto a freshly loaded base exactly as
the server does (``rpc_model_server._wrap_system2_lora``: plain LoRA, decoder blocks only, every tensor
landed key for key and value for value) and its contract must say ``lora_b_all_zero``.  CPU only; the
model is loaded twice (about 16 GB each in bf16).  A summary goes to ``<output>/zero_adapter.json``
(the server reads only adapter_config.json and adapter_model.safetensors).

    cd $R/exp37/src_x37 && TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=$PWD:$R/rpc/src \\
      $R/envs/ppa/bin/python scripts/exp37/make_zero_adapter.py --model-path $R/InternNav_Model \\
      --reference-config $R/exp35/train/X35/adapter/adapter_config.json --output $R/exp37/zero_adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "heatmapvln-exp37-zero-adapter-v1"
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
# adapter_config.json keys that must equal the reference's (X35's): what the server and PEFT act on.
COMPARED_KEYS = ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type",
                 "use_dora", "use_rslora", "modules_to_save", "fan_in_fan_out", "init_lora_weights",
                 "layers_to_transform", "layers_pattern", "rank_pattern", "alpha_pattern", "exclude_modules")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_saved(directory: Path, reference_config: Path | None, num_layers: int) -> dict[str, Any]:
    """The file as the server will read it: every lora_B exactly 0, lora_A finite and not 0, the config X37's."""
    import torch
    from safetensors import safe_open

    with safe_open(str(directory / WEIGHTS), framework="pt") as handle:
        saved = {key: handle.get_tensor(key) for key in handle.keys()}
    a_keys = sorted(k for k in saved if k.endswith(".lora_A.weight"))
    b_keys = sorted(k for k in saved if k.endswith(".lora_B.weight"))
    from scripts.train_system2_lora import LORA_MODULES_PER_LAYER

    want = num_layers * LORA_MODULES_PER_LAYER
    if len(a_keys) != want or len(b_keys) != want or len(saved) != 2 * want:
        raise RuntimeError(f"{WEIGHTS} holds {len(a_keys)} A / {len(b_keys)} B of {len(saved)} tensors, want {want} each")
    nonzero_b = [k for k in b_keys if bool(saved[k].any())]
    if nonzero_b:
        raise RuntimeError(f"lora_B is not zero: {nonzero_b[:3]}")
    bad_a = [k for k in a_keys if not bool(torch.isfinite(saved[k]).all()) or not bool(saved[k].any())]
    if bad_a:
        raise RuntimeError(f"lora_A is non-finite or all zero: {bad_a[:3]}")
    dtypes = sorted({str(v.dtype) for v in saved.values()})
    config = json.loads((directory / CONFIG).read_text(encoding="utf-8"))
    differing: dict[str, Any] = {}
    if reference_config is not None:
        reference = json.loads(reference_config.read_text(encoding="utf-8"))
        differing = {k: [config.get(k), reference.get(k)] for k in COMPARED_KEYS if config.get(k) != reference.get(k)}
        if differing:
            raise RuntimeError(f"adapter_config.json differs from {reference_config} in {differing}")
    a_digest = hashlib.sha256()
    for key in a_keys:
        a_digest.update(key.encode())
        a_digest.update(saved[key].contiguous().view(torch.uint8).numpy().tobytes())
    return {
        "tensors": len(saved),
        "dtypes": dtypes,
        "lora_b_all_zero": True,
        "lora_a_sha256": a_digest.hexdigest(),
        "config": {k: config.get(k) for k in COMPARED_KEYS},
        "reference_config": str(reference_config) if reference_config else None,
        "reference_config_sha256": sha256_file(reference_config) if reference_config else None,
    }


def server_check(model_path: str, directory: Path) -> dict[str, Any]:
    """Load the directory onto a fresh base with the server's own wrapper and checks."""
    import torch

    from scripts.evaluation.rpc_model_server import _wrap_system2_lora
    from scripts.train_system2_lora import load_system2

    model, _ = load_system2(model_path, torch.device("cpu"))
    wrapped, contract = _wrap_system2_lora(model, str(directory))
    if contract.get("lora_b_all_zero") is not True:
        raise RuntimeError(f"the server's contract does not see lora_B = 0: {contract}")
    del wrapped, model
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="the deployed InternNav_Model directory")
    parser.add_argument("--output", required=True, help="adapter directory to create (must not exist)")
    parser.add_argument("--reference-config", default="",
                        help="an adapter_config.json whose LoRA settings must match (X35's); empty = no comparison")
    parser.add_argument("--seed", type=int, default=42, help="the trainer's --seed (its default)")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-server-check", action="store_true", help="skip loading it through the server's wrapper")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise SystemExit(f"{output} exists; give a new directory")
    reference = Path(args.reference_config) if args.reference_config else None
    if reference is not None and not reference.is_file():
        raise SystemExit(f"no reference config {reference}")

    import torch

    from scripts.train_system2_lora import attach_new_lora, check_lora_targets, load_system2, seed_everything

    started = time.time()
    device = torch.device("cpu")
    # The trainer's order (train_system2_lora._fit): seed, load, attach.
    seed_everything(args.seed, device)
    model, _ = load_system2(args.model_path, device)
    num_layers = int(model.config.num_hidden_layers)
    peft_model = attach_new_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                 gradient_checkpointing=False)
    lora_info = check_lora_targets(peft_model, num_layers)
    part = output.with_name(output.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    peft_model.save_pretrained(str(part), safe_serialization=True)
    del peft_model, model
    saved = check_saved(part, reference, num_layers)
    print(f"[zero-adapter] saved {saved['tensors']} tensors ({saved['dtypes']}), lora_B = 0, "
          f"lora_A sha256 {saved['lora_a_sha256'][:16]}", flush=True)
    contract = None
    if not args.no_server_check:
        contract = server_check(args.model_path, part)
        print(f"[zero-adapter] server wrapper: {contract['lora_layers']} layers, {contract['lora_tensors']} tensors, "
              f"lora_b_all_zero {contract['lora_b_all_zero']}", flush=True)
    summary = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "wall_s": round(time.time() - started, 1),
        "model_path": str(args.model_path),
        "seed": args.seed,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, **lora_info},
        "files": {name: sha256_file(part / name) for name in (WEIGHTS, CONFIG)},
        "check": saved,
        "server_contract": contract,
        "commit": subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT}", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=False).stdout.strip(),
    }
    (part / "zero_adapter.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    part.rename(output)
    print(f"[zero-adapter] {output} adapter_model.safetensors sha256 {summary['files'][WEIGHTS]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
