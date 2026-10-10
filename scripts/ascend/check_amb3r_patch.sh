#!/usr/bin/env bash
# Refuse to serve from an AMB3R tree that does not carry scripts/ascend/amb3r_npu.patch.
#
#   bash scripts/ascend/check_amb3r_patch.sh <amb3r-tree> [patch]
#
# AMB3R is third party and is not vendored here, so its two NPU fixes ship as a
# patch that someone has to apply by hand on each instance.  Both of them fail
# SILENTLY when they are missing, which is why this check exists:
#
#   - slam/pipeline.py names 'cuda' in its autocast.  Off CUDA, torch only warns and
#     runs the mapping forward in fp32: different poses from the certified reference,
#     and twice the activation memory on a card that has about 3 GB spare.
#   - thirdparty/depth_anything_3/api.py picks its dtype from
#     torch.cuda.is_bf16_supported(), which is False off CUDA, so DA3 drops to fp16.
#
# Neither leaves a trace in any log.  Before this check, a clean clone and a patched
# tree passed every gate the deployment had: bootstrap_instance.sh only tested that
# files exist, and the launcher's evidence greps do not cover the tree at all.
#
# Read-only.  It never writes to the tree: applying the patch is a decision for the
# operator, and the command to do it is in the refusal message.

set -Eeuo pipefail

TREE="${1:?usage: check_amb3r_patch.sh <amb3r-tree> [patch]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH="${2:-$REPO_ROOT/scripts/ascend/amb3r_npu.patch}"
# df74392^{tree} in the AMB3R checkout: the certified base, identical to the tree
# deployed on the RTX 4090 (it holds the local chunked-SDPA attention and the utils3d
# wheel, and not the NPU patch).  Everything outside the three patched files must still
# match it, or this is not the tree the numbers came from.
CERTIFIED_BASE_TREE="${PPA_NPU_AMB3R_BASE_TREE:-09f1b2faff3b3be71cf2dc775afa26041356d296}"

die() { printf '[amb3r-patch] ERROR: %s\n' "$*" >&2; exit 2; }
warn() { printf '[amb3r-patch] WARN %s\n' "$*" >&2; }
ok() { printf '[amb3r-patch] ok   %s\n' "$*"; }

PIPELINE="$TREE/slam/pipeline.py"
API="$TREE/thirdparty/depth_anything_3/api.py"
ROPE="$TREE/thirdparty/depth_anything_3/model/dinov2/layers/rope.py"

# "Not patched" and "not there at all" need different answers, so check readability
# first: the second one means the share is not mounted or the root is wrong.
[[ -d "$TREE" ]] || die "no AMB3R tree at $TREE (wrong PPA_EVAL_AMB3R_ROOT, or the share is not mounted)"
for file in "$PIPELINE" "$API" "$ROPE"; do
  [[ -r "$file" ]] || die "AMB3R tree incomplete: $file is not readable (wrong PPA_EVAL_AMB3R_ROOT, or the share is not mounted)"
done
[[ -r "$PATCH" ]] || die "missing patch: $PATCH"

apply_hint="git -c safe.directory='$TREE' -C '$TREE' apply '$PATCH'"

# The three lines the patch adds.  Positive checks, because the thing that must be
# true is that the fixed code is present -- not merely that the broken code is gone.
grep -qF "device_type = views_all['images'].device.type" "$PIPELINE" \
  || die "AMB3R tree is not patched: $PIPELINE still names a device instead of following the tensors, so the mapping forward would run in fp32. Apply it with: $apply_hint"
grep -qF "refusing to run in fp16 silently" "$API" \
  || die "AMB3R tree is not patched: $API does not refuse fp16, so DA3 would silently drop to fp16. Apply it with: $apply_hint"
grep -qF "max_position = int(positions.shape[-2]) + 1" "$ROPE" \
  || die "AMB3R tree is not patched: $ROPE still reads positions.max() off the device, and that read hangs this platform (AI_CPU_Timeout / ACL 507017) in the map-init forward. Apply it with: $apply_hint"

# And the broken lines must be gone from these two files.  Fatal, not a warning: a
# half-applied tree -- the fixed line added while the old one is still live above it --
# would otherwise pass, and the old line is the one that runs.  Scoped to these two
# files, where each literal occurs exactly once before the patch and not at all after
# (checked on the 910B tree); the same strings elsewhere in AMB3R (demo.py, sfm/run.py,
# benchmark/*, thirdparty/vggt-omega) are not on the served path and are not looked at.
if grep -qF "with torch.autocast(device_type='cuda', dtype=torch.bfloat16):" "$PIPELINE"; then
  die "$PIPELINE carries the fix AND still has a cuda-named autocast: half-applied, and the cuda-named one is what runs. Re-apply from a clean tree: $apply_hint"
fi
if grep -qF "torch.cuda.is_bf16_supported()" "$API"; then
  die "$API carries the fix AND still probes bf16 through torch.cuda: half-applied, so DA3 can still drop to fp16. Re-apply from a clean tree: $apply_hint"
fi
if grep -qF "max_position = int(positions.max()) + 1" "$ROPE"; then
  die "$ROPE carries the fix AND still has the device read above it: half-applied, and the device read is the one that runs. Re-apply from a clean tree: $apply_hint"
fi

if git -c safe.directory="$TREE" -C "$TREE" rev-parse --git-dir >/dev/null 2>&1; then
  # The hunks apply in reverse exactly when they are present with their context intact.
  git -c safe.directory="$TREE" -C "$TREE" apply --reverse --check "$PATCH" 2>/dev/null \
    || warn "the patch does not reverse-apply cleanly: the tree carries both fixes (checked above) but its context has drifted; re-generate $PATCH against this tree"
  # Nothing else may differ from the base the certified numbers came from.  The three
  # patched files are excluded because the patch is exactly what changes them.
  if git -c safe.directory="$TREE" -C "$TREE" cat-file -e "$CERTIFIED_BASE_TREE^{tree}" 2>/dev/null; then
    git -c safe.directory="$TREE" -C "$TREE" diff --quiet "$CERTIFIED_BASE_TREE" -- . \
      ':!slam/pipeline.py' ':!thirdparty/depth_anything_3/api.py' \
      ':!thirdparty/depth_anything_3/model/dinov2/layers/rope.py' \
      || die "AMB3R tree differs from the certified base ${CERTIFIED_BASE_TREE:0:7} outside the three patched files; see: git -c safe.directory='$TREE' -C '$TREE' diff --stat $CERTIFIED_BASE_TREE -- . ':!slam/pipeline.py' ':!thirdparty/depth_anything_3/api.py' ':!thirdparty/depth_anything_3/model/dinov2/layers/rope.py'"
    # git diff is blind to untracked files, so the comparison above would pass with an
    # extra module sitting in the tree -- and an extra module is exactly how an import
    # gets shadowed.  The weights are legitimately untracked, hence the exclusion.
    untracked="$(git -c safe.directory="$TREE" -C "$TREE" ls-files --others --exclude-standard \
      -- . ':!checkpoints/*' ':!slam/pipeline.py' ':!thirdparty/depth_anything_3/api.py' \
      ':!thirdparty/depth_anything_3/model/dinov2/layers/rope.py' || true)"
    untracked_code="$(printf '%s\n' "$untracked" | grep -E '\.(py|so|pth)$' || true)"
    if [[ -n "$untracked_code" ]]; then
      printf '%s\n' "$untracked_code" >&2
      die "the AMB3R tree carries untracked code (listed above) that the certified base does not have"
    fi
    [[ -z "${untracked//[[:space:]]/}" ]] \
      || warn "untracked non-code files in the tree: $(printf '%s' "$untracked" | tr '\n' ' ')"
    ok "tree matches the certified base ${CERTIFIED_BASE_TREE:0:7} outside the three patched files"
  else
    warn "the certified base tree ${CERTIFIED_BASE_TREE:0:7} is not in this checkout, so only the three patched files were verified"
  fi
else
  warn "$TREE is not a git checkout, so only the three patched files were verified"
fi

ok "AMB3R tree at $TREE carries the NPU patch"
