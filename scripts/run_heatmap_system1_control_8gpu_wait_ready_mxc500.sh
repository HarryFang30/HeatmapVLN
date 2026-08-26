#!/usr/bin/env bash
# Queue-safe entry point for the formal 8-GPU heatmap-control experiment.
#
# The two upstream jobs run independently. This script intentionally has no
# readiness timeout: missing, partially-written, or changing artifacts keep the
# allocated job waiting instead of turning a normal dependency delay into a
# failed cluster submission. Once both publications are stable, the final
# heatmap checksum is captured and the audited 3-epoch train -> full 8-GPU
# R2R val_unseen evaluation launcher replaces this process.

set -u
set -o pipefail
umask 027

readonly REPO_ROOT="/mnt/afs/liwenhao/agent/370910109/HeatmapVLN"

# Pin the producer run, not runs/latest. best.pth is intentionally selected by
# pathname only; its checksum is computed after run completion because epochs
# 5/6 may still replace the current incumbent.
readonly HEATMAP_RUN="/mnt/afs/liwenhao/agent/370910109/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402"
readonly HEATMAP_BEST="${HEATMAP_RUN}/checkpoints/best.pth"
readonly HEATMAP_SUMMARY="${HEATMAP_RUN}/manifest/summary.json"
readonly HEATMAP_METRICS="${HEATMAP_RUN}/logs/metrics.jsonl"

# This is atomically published only after all four DAgger shards and their
# sample indexes pass the finalizer's full validation.
readonly DAGGER_MANIFEST="/mnt/afs/liwenhao/agent/370910109/data/heatmap_system1_training_v1/rollout_control/round_000/full_train_4way_seed17/training_roots.json"

readonly CONTROL_LAUNCHER="${REPO_ROOT}/scripts/run_heatmap_system1_control_8gpu_mxc500.sh"

POLL_SECONDS="${HEATMAP_CONTROL_WAIT_POLL_SECONDS:-300}"
STABLE_SECONDS="${HEATMAP_CONTROL_WAIT_STABLE_SECONDS:-60}"

log() {
  printf '[wait-control][%s] %s\n' "$(date -Is 2>/dev/null || date)" "$*"
}

positive_integer_or_default() {
  local value="$1"
  local fallback="$2"
  if [[ "$value" =~ ^[0-9]+$ ]] && (( 10#$value > 0 )); then
    printf '%s\n' "$((10#$value))"
  else
    printf '%s\n' "$fallback"
  fi
}

POLL_SECONDS="$(positive_integer_or_default "$POLL_SECONDS" 300)"
STABLE_SECONDS="$(positive_integer_or_default "$STABLE_SECONDS" 60)"
readonly POLL_SECONDS STABLE_SECONDS

on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  log "received ${signal_name}; stopping"
  exit "$exit_code"
}

trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

safe_regular_file() {
  [[ -f "$1" && -s "$1" && ! -L "$1" ]]
}

heatmap_run_finished() {
  local last_record

  safe_regular_file "$HEATMAP_BEST" || return 1
  safe_regular_file "$HEATMAP_SUMMARY" || return 1
  safe_regular_file "$HEATMAP_METRICS" || return 1

  # summary.json is written after the training loop; verify it belongs to the
  # exact pinned run and is no longer a partial JSON write.
  [[ "$(tail -c 1 -- "$HEATMAP_SUMMARY" 2>/dev/null)" == "}" ]] || return 1
  LC_ALL=C grep -Fq \
    "\"run_dir\": \"${HEATMAP_RUN}\"" \
    "$HEATMAP_SUMMARY" 2>/dev/null || return 1

  # train.py appends run_complete immediately after summary.json. Requiring a
  # complete final JSONL record closes the tiny summary-write/append interval.
  last_record="$(tail -n 1 -- "$HEATMAP_METRICS" 2>/dev/null)" || return 1
  [[ "$last_record" == \{*\} ]] || return 1
  LC_ALL=C grep -Eq \
    '"record_type"[[:space:]]*:[[:space:]]*"run_complete"' \
    <<< "$last_record" 2>/dev/null
}

dagger_manifest_published() {
  safe_regular_file "$DAGGER_MANIFEST" || return 1
  LC_ALL=C grep -Eq \
    '"schema"[[:space:]]*:[[:space:]]*"heatmapvln-trajectory-dagger-training-roots-v1"' \
    "$DAGGER_MANIFEST" 2>/dev/null || return 1
  LC_ALL=C grep -Eq \
    '"ready"[[:space:]]*:[[:space:]]*true' \
    "$DAGGER_MANIFEST" 2>/dev/null || return 1
  LC_ALL=C grep -Eq \
    '"num_shards"[[:space:]]*:[[:space:]]*4([,}[:space:]]|$)' \
    "$DAGGER_MANIFEST" 2>/dev/null
}

file_signature() {
  local path="$1"
  local checksum_line hash ignored metadata

  safe_regular_file "$path" || return 1
  checksum_line="$(sha256sum -- "$path" 2>/dev/null)" || return 1
  read -r hash ignored <<< "$checksum_line"
  [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || return 1
  metadata="$(stat -c '%s:%Y' -- "$path" 2>/dev/null)" || return 1

  printf '%s:%s\n' "$hash" "$metadata"
}

file_state() {
  if safe_regular_file "$1"; then
    printf 'present'
  else
    printf 'waiting'
  fi
}

log "waiting indefinitely for the exact upstream artifacts"
log "heatmap_best=${HEATMAP_BEST}"
log "heatmap_completion=${HEATMAP_SUMMARY} + final run_complete record"
log "dagger_manifest=${DAGGER_MANIFEST}"
log "poll_seconds=${POLL_SECONDS} stability_seconds=${STABLE_SECONDS}"

while :; do
  log "status: heatmap_best=$(file_state "$HEATMAP_BEST") heatmap_summary=$(file_state "$HEATMAP_SUMMARY") dagger_manifest=$(file_state "$DAGGER_MANIFEST")"

  if heatmap_run_finished &&
     dagger_manifest_published &&
     safe_regular_file "$CONTROL_LAUNCHER"
  then
    heatmap_sig_1="$(file_signature "$HEATMAP_BEST")" || heatmap_sig_1=""
    summary_sig_1="$(file_signature "$HEATMAP_SUMMARY")" || summary_sig_1=""
    metrics_sig_1="$(file_signature "$HEATMAP_METRICS")" || metrics_sig_1=""
    dagger_sig_1="$(file_signature "$DAGGER_MANIFEST")" || dagger_sig_1=""

    if [[ -n "$heatmap_sig_1" &&
          -n "$summary_sig_1" &&
          -n "$metrics_sig_1" &&
          -n "$dagger_sig_1" ]]
    then
      log "candidate artifacts found; checking stability for ${STABLE_SECONDS}s"
      sleep "$STABLE_SECONDS"

      heatmap_sig_2="$(file_signature "$HEATMAP_BEST")" || heatmap_sig_2=""
      summary_sig_2="$(file_signature "$HEATMAP_SUMMARY")" || summary_sig_2=""
      metrics_sig_2="$(file_signature "$HEATMAP_METRICS")" || metrics_sig_2=""
      dagger_sig_2="$(file_signature "$DAGGER_MANIFEST")" || dagger_sig_2=""

      if heatmap_run_finished &&
         dagger_manifest_published &&
         [[ "$heatmap_sig_1" == "$heatmap_sig_2" ]] &&
         [[ "$summary_sig_1" == "$summary_sig_2" ]] &&
         [[ "$metrics_sig_1" == "$metrics_sig_2" ]] &&
         [[ "$dagger_sig_1" == "$dagger_sig_2" ]] &&
         [[ ! "$HEATMAP_BEST" -nt "$HEATMAP_SUMMARY" ]]
      then
        HEATMAP_SHA256="${heatmap_sig_2%%:*}"
        if [[ "$HEATMAP_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
          break
        fi
        log "final heatmap SHA-256 was malformed; continuing to wait"
      else
        log "candidate changed or was not fully published; continuing to wait"
      fi
    fi
  fi

  sleep "$POLL_SECONDS"
done

log "both upstream tasks are ready and stable"
log "final heatmap SHA-256=${HEATMAP_SHA256}"
log "starting 8-GPU control training followed by epoch-3 full 8-GPU evaluation"

cd "$REPO_ROOT" || exit 1

export HEATMAP_CONTROL_CKPT="$HEATMAP_BEST"
export HEATMAP_CONTROL_CKPT_SHA256="$HEATMAP_SHA256"
export DAGGER_TRAINING_ROOTS_MANIFEST="$DAGGER_MANIFEST"

export GPU_DEVICES="0,1,2,3,4,5,6,7"
export EVAL_GPU_DEVICES="$GPU_DEVICES"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29641"
export EVAL_RPC_PORT_BASE="51400"
export EVAL_DISPLAY_BASE="280"

export HEATMAP_CONTROL_EPOCH_SIZE="72000"
export HEATMAP_CONTROL_DRY_RUN="0"
export HEATMAP_CONTROL_AUTO_RESUME="1"
export HEATMAP_CONTROL_AUTO_EVAL="1"
export EVAL_CONTROL_MODE="on"
export EVAL_X11_MODE="bundle"

unset HEATMAP_CONTROL_RESUME
unset EVAL_OUTPUT_ROOT
unset EVAL_PREFLIGHT_ONLY
unset EVAL_SMOKE_ONLY
unset EVAL_SKIP_SMOKE
unset EVAL_REUSE_XVFB
unset LOG_FILE

exec /usr/bin/bash "$CONTROL_LAUNCHER"
