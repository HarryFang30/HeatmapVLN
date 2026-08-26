#!/usr/bin/env bash
set -euo pipefail

FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
REPO_ROOT=${ENDPOINT_PROBE_REPO_ROOT:-${FJL_ROOT}/HeatmapVLN}
PYTHON_BIN=${ENDPOINT_PROBE_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}

SOURCE_AUDIT_ROOT=${ENDPOINT_PROBE_SOURCE_AUDIT_ROOT:-${FJL_ROOT}/data/candidate_support_audit_v2/train_balanced_512_native_seed42}
CONTINUATION_ROOT=${ENDPOINT_PROBE_CONTINUATION_ROOT:-${FJL_ROOT}/data/candidate_continuation_v1/train_balanced_1024_native_pi0_seed42}
TARGETS_ROOT=${ENDPOINT_PROBE_TARGETS_ROOT:-${FJL_ROOT}/data/candidate_continuation_targets_v1/train_balanced_1024_seed20260810}
WORKER_ROOT=${ENDPOINT_PROBE_WORKER_ROOT:-${FJL_ROOT}/model/candidate_continuation_v1/train_balanced_1024_native_pi0_seed42/workers}
OUTPUT_DIR=${ENDPOINT_PROBE_OUTPUT_DIR:-${FJL_ROOT}/model/endpoint_advantage_identifiability_v1/train_balanced_1024_native_pi0_seed42}

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${ENDPOINT_PROBE_GPU_DEVICE:-0}
export OMP_NUM_THREADS=${ENDPOINT_PROBE_OMP_NUM_THREADS:-4}

# A plain SSH/dev shell does not inherit the MetaX runtime variables that the
# web scheduler normally injects.  PyTorch imports the MetaX Triton backend
# when AdamW is constructed, so keep this identical to the repository's
# established MXC500 training launchers.
export MACA_HOME=${MACA_HOME:-/opt/maca-3.3.0}
export MACA_PATH=${MACA_PATH:-${MACA_HOME}}
export MACA_DIR=${MACA_DIR:-${MACA_PATH}}
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

extra_args=()
if [[ ${ENDPOINT_PROBE_SKIP_INTEGRITY_CHECK:-0} == 1 ]]; then
  extra_args+=(--skip-integrity-check)
fi
if [[ ${ENDPOINT_PROBE_PREFLIGHT_ONLY:-0} == 1 ]]; then
  extra_args+=(--preflight-only)
fi
if [[ ${ENDPOINT_PROBE_STATIC_ONLY:-0} == 1 ]]; then
  extra_args+=(--static-only)
fi
if [[ ${ENDPOINT_PROBE_MAX_STATES:-0} != 0 ]]; then
  extra_args+=(--max-states "${ENDPOINT_PROBE_MAX_STATES}")
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

exec "${PYTHON_BIN}" -u \
  "${REPO_ROOT}/scripts/evaluation/probe_endpoint_advantage_identifiability.py" \
  --source-audit-root "${SOURCE_AUDIT_ROOT}" \
  --continuation-root "${CONTINUATION_ROOT}" \
  --targets-root "${TARGETS_ROOT}" \
  --worker-root "${WORKER_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-shards "${ENDPOINT_PROBE_EXPECTED_SHARDS:-8}" \
  --folds "${ENDPOINT_PROBE_FOLDS:-5}" \
  --fold-seed "${ENDPOINT_PROBE_FOLD_SEED:-20260812}" \
  --model-seeds "${ENDPOINT_PROBE_MODEL_SEEDS:-17,42,73}" \
  --variants "${ENDPOINT_PROBE_VARIANTS:-candidate_only,candidate_system2,candidate_system2_heatmap_metadata,candidate_system2_heatmap_geometry,candidate_system2_heatmap_tokens}" \
  --hidden-width "${ENDPOINT_PROBE_HIDDEN_WIDTH:-64}" \
  --dropout "${ENDPOINT_PROBE_DROPOUT:-0.1}" \
  --batch-size "${ENDPOINT_PROBE_BATCH_SIZE:-32}" \
  --epochs "${ENDPOINT_PROBE_EPOCHS:-40}" \
  --patience "${ENDPOINT_PROBE_PATIENCE:-6}" \
  --learning-rate "${ENDPOINT_PROBE_LEARNING_RATE:-3e-4}" \
  --weight-decay "${ENDPOINT_PROBE_WEIGHT_DECAY:-1e-4}" \
  --max-validation-destroy-state-rate "${ENDPOINT_PROBE_MAX_DESTROY_STATE_RATE:-0.02}" \
  --bootstrap-replicates "${ENDPOINT_PROBE_BOOTSTRAP_REPLICATES:-1000}" \
  --device "${ENDPOINT_PROBE_DEVICE:-cuda}" \
  "${extra_args[@]}"
