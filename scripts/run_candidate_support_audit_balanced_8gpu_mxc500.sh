#!/usr/bin/env bash
set -Eeuo pipefail

FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
REPO="$FJL_ROOT/HeatmapVLN"

export AUDIT_DATASET_SPLIT=train
export AUDIT_DEPLOYMENT_ARM=native
export AUDIT_COHORTS_DIR="$FJL_ROOT/data/candidate_support_audit_cohorts_v2/train_balanced_512_seed20260810"
export AUDIT_COHORT_EPISODES_PER_SHARD=64
export AUDIT_MAX_EPISODES_PER_SHARD=0
export AUDIT_MAX_GB_PER_SHARD="${AUDIT_MAX_GB_PER_SHARD:-10}"
export AUDIT_MAX_GB_TOTAL="${AUDIT_MAX_GB_TOTAL:-80}"
export AUDIT_ROOT="$FJL_ROOT/data/candidate_support_audit_v2/train_balanced_512_native_seed42"
export AUDIT_OUTPUT_ROOT="$FJL_ROOT/model/candidate_support_audit_v2/train_balanced_512_native_seed42"

exec bash "$REPO/scripts/run_candidate_support_audit_8gpu_mxc500.sh"
