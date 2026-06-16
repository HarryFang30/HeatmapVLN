#!/usr/bin/env bash
# Parallel-capacity experiment for Stage2 pano adapter on 8×MXC500.
#
# This wrapper keeps the frozen student/System1 config unchanged and only uses
# a larger PanoLatentSpaceAdapter hidden_dim=1024.  It expects the dense native
# teacher sidecar to already exist, so it does not launch teacher collection.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export STAGE2_ADAPTER_STUDENT_CONFIG="${STAGE2_ADAPTER_STUDENT_CONFIG:-configs/train_pano_adapter_stage2_8gpu.yaml}"
export STAGE2_ADAPTER_CONFIG="${STAGE2_ADAPTER_CONFIG:-configs/adapter_pano_stage2_h1024.yaml}"
export STAGE2_ADAPTER_OUT_DIR="${STAGE2_ADAPTER_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage2_adapter_h1024}"
export LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/stage2_adapter_h1024_8gpu_mxc500.log}"

# This experiment should reuse the dense sidecar produced by the default
# launcher.  If it is missing, fail fast instead of spending another long
# collection pass.
export STAGE2_TEACHER_COLLECT_ENABLE="${STAGE2_TEACHER_COLLECT_ENABLE:-0}"

# Use a different default port so this can run beside the baseline job on a
# separate allocation without colliding with its torchrun rendezvous.
export MASTER_PORT_STAGE2_ADAPTER="${MASTER_PORT_STAGE2_ADAPTER:-29619}"

exec bash "${SCRIPT_DIR}/run_stage2_pano_adapter_8gpu_mxc500_launcher.sh"
