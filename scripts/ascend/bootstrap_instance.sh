#!/usr/bin/env bash
# Make a freshly created Ascend notebook instance ready to serve, from nothing but
# the container image plus the persistent share.
#
# Everything the deployment needs lives under $PPA_EVAL_ROOT on the SFS Turbo share
# and survives the instance: the repo, the weights, the AMB3R tree and the 6.4 GB
# Python environment (whose absolute prefix is inside the share, so it stays valid).
# CANN comes from the image.  Only two things are lost with the old instance, both
# on the local overlay: the tunnel key's authorization in ~/.ssh/authorized_keys,
# and pip's config.  This script restores the first and checks everything else.
#
# Run it once per new instance:
#   bash scripts/ascend/bootstrap_instance.sh
#
# Then, on the laptop, point the SSH alias at the new address (it changes with every
# instance) and start the servers as usual - see docs/ops/deploy_ascend_910b.md.

set -Eeuo pipefail

ROOT="${PPA_EVAL_ROOT:-$HOME/work/zhr/zhr_1}"
REPO="${PPA_EVAL_REPO:-$ROOT/HeatmapVLN}"
PYTHON="${PPA_EVAL_PYTHON:-$ROOT/envs/ppa/bin/python}"
TUNNEL_KEY_PUB="${PPA_TUNNEL_KEY_PUB:-$ROOT/tunnel/tunnel_key.pub}"
ASCEND_ENV="${PPA_NPU_ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
# The slot ports the tunnel key is allowed to forward, and nothing else.
MODEL_PORT_BASE="${PPA_EVAL_MODEL_PORT_BASE:-52400}"
VO_PORT_BASE="${PPA_EVAL_VO_PORT_BASE:-52500}"
MAX_SLOTS="${PPA_NPU_MAX_SLOTS:-8}"

# The image this environment was built against.  A different CANN or torch means the
# environment on the share may not match the image, which is worth knowing before a
# run rather than after.
EXPECT_CANN="${PPA_NPU_EXPECT_CANN:-8.3.RC1}"
EXPECT_TORCH="${PPA_NPU_EXPECT_TORCH:-2.7.1}"

die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 2; }
ok() { printf '[bootstrap] ok   %s\n' "$*"; }
warn() { printf '[bootstrap] WARN %s\n' "$*" >&2; }

printf '[bootstrap] root=%s\n' "$ROOT"

# --- the machine -------------------------------------------------------------
[[ "$(uname -m)" == "aarch64" ]] || die "expected aarch64, got $(uname -m)"
ok "aarch64"

command -v npu-smi >/dev/null || die "npu-smi not found; is this an Ascend instance?"
chips="$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +910' || true)"
[[ "${chips:-0}" -ge 1 ]] || die "npu-smi reports no 910 chips"
ok "npu-smi sees $chips chip(s)"

[[ -r "$ASCEND_ENV" ]] || die "CANN env script not readable: $ASCEND_ENV"
info="$(dirname "$ASCEND_ENV")/latest/aarch64-linux/ascend_toolkit_install.info"
if [[ -r "$info" ]]; then
  cann="$(awk -F= '/^version=/ {print $2; exit}' "$info" | tr -d ' ')"
  if [[ "$cann" == "$EXPECT_CANN" ]]; then
    ok "CANN $cann"
  else
    warn "CANN is $cann, the environment on the share was built against $EXPECT_CANN"
  fi
else
  warn "could not read $info"
fi

# --- the share ---------------------------------------------------------------
for path in "$ROOT" "$REPO" "$ROOT/amb3r/slam/slam_config.yaml" \
  "$ROOT/InternNav_Model/config.json" "$ROOT/weights/ppa_refine_v2_best.pth" \
  "$ROOT/rpc/src/vla_rpc" "$ROOT/amb3r/checkpoints/DA3NESTED-GIANT-LARGE/model.safetensors"; do
  [[ -e "$path" ]] || die "missing on the share: $path (is the right volume mounted?)"
done
ok "repo, weights, AMB3R tree and RPC tools present on the share"

[[ -x "$PYTHON" ]] || die "missing environment: $PYTHON (rebuild it, see docs/ops/deploy_ascend_910b.md)"

# --- the environment ---------------------------------------------------------
# shellcheck disable=SC1090
source "$ASCEND_ENV"
PYTHONPATH="$ROOT/rpc/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$EXPECT_TORCH" <<'PY' || die "environment check failed"
import json
import sys

expect_torch = sys.argv[1]
problems = []

import numpy
import torch
import torch_npu  # noqa: F401
import transformers

if transformers.__version__ != "4.51.0":
    problems.append(f"transformers {transformers.__version__} != 4.51.0 (runtime_compat gate)")
if not numpy.__version__.startswith("1."):
    problems.append(f"numpy {numpy.__version__} is not 1.x; numpy 2 breaks the numpy-1 extensions here")
if not torch.__version__.startswith(expect_torch):
    problems.append(f"torch {torch.__version__} does not start with {expect_torch}")
try:
    import cv2
except Exception as exc:  # noqa: BLE001
    problems.append(f"cv2 does not import ({type(exc).__name__}: {exc})")
    cv2 = None
if not torch.npu.is_available():
    problems.append("torch.npu.is_available() is False")
elif not torch.npu.is_bf16_supported():
    problems.append("torch.npu.is_bf16_supported() is False; the reference dtype is bf16")

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "diffusers": __import__("diffusers").__version__,
    "numpy": numpy.__version__,
    "cv2": getattr(cv2, "__version__", None),
    "npu_count": torch.npu.device_count() if torch.npu.is_available() else 0,
}, sort_keys=True))
if problems:
    print("ENVIRONMENT: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
PY
ok "environment versions match the certified stack"

# --- the tunnel key ----------------------------------------------------------
# ~/.ssh is on the local overlay, so a new instance starts with no authorization for
# the 4090's tunnel key and the clients cannot reach these servers at all.
if [[ -r "$TUNNEL_KEY_PUB" ]]; then
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  touch "$HOME/.ssh/authorized_keys" && chmod 600 "$HOME/.ssh/authorized_keys"
  key="$(tr -d '\n' < "$TUNNEL_KEY_PUB")"
  comment="${key##* }"
  [[ -n "$comment" ]] || die "$TUNNEL_KEY_PUB has no comment field to identify it by"
  permit=""
  for slot in $(seq 0 $((MAX_SLOTS - 1))); do
    permit+=",permitopen=\"127.0.0.1:$((MODEL_PORT_BASE + slot))\""
    permit+=",permitopen=\"127.0.0.1:$((VO_PORT_BASE + slot))\""
  done
  # Forwarding only, and only to the slot ports: this key can neither open a shell
  # nor reach anything else on the instance.
  line="restrict,port-forwarding${permit} ${key}"
  if grep -qF "$comment" "$HOME/.ssh/authorized_keys"; then
    grep -vF "$comment" "$HOME/.ssh/authorized_keys" > "$HOME/.ssh/authorized_keys.new" || true
    mv "$HOME/.ssh/authorized_keys.new" "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
  fi
  printf '%s\n' "$line" >> "$HOME/.ssh/authorized_keys"
  ok "tunnel key authorized for ports ${MODEL_PORT_BASE}-$((MODEL_PORT_BASE + MAX_SLOTS - 1)) and ${VO_PORT_BASE}-$((VO_PORT_BASE + MAX_SLOTS - 1)), forwarding only"
else
  warn "no tunnel public key at $TUNNEL_KEY_PUB; the client box will not be able to reach these servers"
fi

# --- scratch space -----------------------------------------------------------
# Short paths: CANN's kernel bank binds an AF_UNIX socket under TMPDIR and the limit
# is 108 bytes in total (docs/ops/deploy_ascend_910b.md section 7).
mkdir -p "$ROOT/tmp" "$ROOT/logs" "$ROOT/servers"
ok "scratch directories ready"

cat <<EOF

[bootstrap] done.  Next:
  1. On the laptop, point the SSH alias at this instance's new address and port.
  2. Start the servers here:
       cd $REPO
       PPA_EVAL_ROOT=$ROOT PPA_NPU_DEVICES=0 bash scripts/ascend/run_ppa_servers_npu.sh
  3. In the 4090 container, start the tunnel for each slot, then run the clients with
     PPA_EVAL_EXTERNAL_SERVERS=1.
EOF
