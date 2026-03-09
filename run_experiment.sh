#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/openbmb/soar-toolkit:latest}"
MODEL_INPUT=""
DESCRIPTION="manual experiment"
PORT="${PORT:-30000}"
RESULTS_TSV="${RESULTS_TSV:-${REPO_DIR}/results.tsv}"
SERVER_ARGS="${SERVER_ARGS:-}"
ENABLE_GPTQ="${ENABLE_GPTQ:-0}"
GPTQ_GROUP_SIZE="${GPTQ_GROUP_SIZE:-128}"
GPTQ_BITS="${GPTQ_BITS:-4}"
SPEED_S1="${REPO_DIR}/eval-official/perf_public_set.jsonl"
SPEED_S8="${REPO_DIR}/eval-official/perf_public_set.jsonl"
SPEED_SMAX="${REPO_DIR}/eval-official/perf_public_set.jsonl"

usage() {
  cat <<'EOF'
Usage:
  bash run_experiment.sh --model-input <host_model_dir> [options]

Options:
  --description <text>          Experiment description (required for traceability)
  --docker-image <image>        Docker image (default: ghcr.io/openbmb/soar-toolkit:latest)
  --port <port>                 SGLang server port (default: 30000)
  --results-tsv <path>          Path to results.tsv
  --server-args <text>          Exported as SGLANG_SERVER_ARGS inside container
  --enable-gptq                 Enable W4A16 GPTQ quantization (sets ENABLE_GPTQ=1)
  --gptq-group-size <int>       GPTQ group size (default: 128)
  --gptq-bits <int>             GPTQ bits (default: 4)
  --speed-s1 <jsonl>            Dataset for S1 benchmark
  --speed-s8 <jsonl>            Dataset for S8 benchmark
  --speed-smax <jsonl>          Dataset for Smax benchmark

Env vars:
  DOCKER_IMAGE, PORT, RESULTS_TSV, ENABLE_GPTQ, GPTQ_GROUP_SIZE, GPTQ_BITS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-input)
      MODEL_INPUT="$2"; shift 2 ;;
    --description)
      DESCRIPTION="$2"; shift 2 ;;
    --docker-image)
      DOCKER_IMAGE="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --results-tsv)
      RESULTS_TSV="$2"; shift 2 ;;
    --server-args)
      SERVER_ARGS="$2"; shift 2 ;;
    --enable-gptq)
      ENABLE_GPTQ="1"; shift ;;
    --gptq-group-size)
      GPTQ_GROUP_SIZE="$2"; shift 2 ;;
    --gptq-bits)
      GPTQ_BITS="$2"; shift 2 ;;
    --speed-s1)
      SPEED_S1="$2"; shift 2 ;;
    --speed-s8)
      SPEED_S8="$2"; shift 2 ;;
    --speed-smax)
      SPEED_SMAX="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${MODEL_INPUT}" ]]; then
  echo "--model-input is required" >&2
  usage
  exit 2
fi

if [[ ! -d "${MODEL_INPUT}" ]]; then
  echo "Model input dir not found: ${MODEL_INPUT}" >&2
  exit 2
fi

for p in "${SPEED_S1}" "${SPEED_S8}" "${SPEED_SMAX}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Dataset file not found: ${p}" >&2
    exit 2
  fi
done

COMMIT="nogit"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT="$(git rev-parse --short HEAD)"
fi

TS="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${REPO_DIR}/experiments/${TS}"
mkdir -p "${EXP_DIR}/model-output"

SERVER_LOG="${EXP_DIR}/server.log"
BENCH_LOG="${EXP_DIR}/bench.log"
EVAL_LOG="${EXP_DIR}/eval.log"
RESULT_JSON="${EXP_DIR}/result.json"

to_container_path() {
  local host_path="$1"
  case "$host_path" in
    "${REPO_DIR}"/*)
      printf "/workspace/sala-game/%s" "${host_path#${REPO_DIR}/}"
      ;;
    *)
      printf "%s" "$host_path"
      ;;
  esac
}

S1_IN_CONTAINER="$(to_container_path "${SPEED_S1}")"
S8_IN_CONTAINER="$(to_container_path "${SPEED_S8}")"
SMAX_IN_CONTAINER="$(to_container_path "${SPEED_SMAX}")"

EXTRA_MOUNTS=()
for host_path in "${SPEED_S1}" "${SPEED_S8}" "${SPEED_SMAX}"; do
  case "$host_path" in
    "${REPO_DIR}"/*) ;;
    *)
      parent="$(cd "$(dirname "${host_path}")" && pwd)"
      mount_arg="${parent}:${parent}:ro"
      seen=0
      for m in "${EXTRA_MOUNTS[@]:-}"; do
        [[ "$m" == "$mount_arg" ]] && seen=1 && break
      done
      [[ $seen -eq 0 ]] && EXTRA_MOUNTS+=("-v" "$mount_arg")
      ;;
  esac
done

CONTAINER_CMD="
set -euo pipefail
cd /workspace/sala-game
source ./prepare_env.sh
bash ./prepare_model.sh --input /workspace/model-input --output /workspace/model-output
eval python3 -m sglang.launch_server --model-path /workspace/model-output --port ${PORT} \${SGLANG_SERVER_ARGS} > /workspace/exp/server.log 2>&1 &
SERVER_PID=\$!
cleanup() { kill \${SERVER_PID} >/dev/null 2>&1 || true; }
trap cleanup EXIT

for i in \$(seq 1 180); do
  if curl -sS --fail http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then
    echo "[run_experiment] server ready at iteration \${i}"
    break
  fi
  if [ \$i -eq 60 ]; then echo "[run_experiment] still waiting for server at 120s..."; fi
  if [ \$i -eq 120 ]; then echo "[run_experiment] still waiting for server at 240s..."; fi
  sleep 2
done

if ! curl -sS --fail http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then
  echo "[run_experiment] ERROR: server did not become healthy within 360s" >&2
  exit 1
fi

SPEED_DATA_S1='${S1_IN_CONTAINER}' \\
SPEED_DATA_S8='${S8_IN_CONTAINER}' \\
SPEED_DATA_SMAX='${SMAX_IN_CONTAINER}' \\
bash ./eval-official/bench_serving.sh http://127.0.0.1:${PORT} > /workspace/exp/bench.log 2>&1

python3 ./eval-official/eval_model.py \\
  --model_path /workspace/model-output \\
  --api_base http://127.0.0.1:${PORT} \\
  --data_path /workspace/sala-game/eval-official/perf_public_set.jsonl \\
  --concurrency 8 > /workspace/exp/eval.log 2>&1
"

DOCKER_ARGS=(
  run --rm --gpus all --network host
  -e "SGLANG_SERVER_ARGS=${SERVER_ARGS}"
  -e "ENABLE_GPTQ=${ENABLE_GPTQ}"
  -e "GPTQ_GROUP_SIZE=${GPTQ_GROUP_SIZE}"
  -e "GPTQ_BITS=${GPTQ_BITS}"
  -v "${REPO_DIR}:/workspace/sala-game"
  -v "${MODEL_INPUT}:/workspace/model-input:ro"
  -v "${EXP_DIR}/model-output:/workspace/model-output"
  -v "${EXP_DIR}:/workspace/exp"
)
if [[ ${#EXTRA_MOUNTS[@]:-0} -gt 0 ]]; then
  DOCKER_ARGS+=("${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"}")
fi
DOCKER_ARGS+=(--entrypoint bash "${DOCKER_IMAGE}" -c "${CONTAINER_CMD}")

echo "[run_experiment] commit=${COMMIT}"
echo "[run_experiment] exp_dir=${EXP_DIR}"
echo "[run_experiment] image=${DOCKER_IMAGE}"

set +e
docker "${DOCKER_ARGS[@]}"
DOCKER_RC=$?
set -e

SUMMARY_JSON="$(python3 - <<'PY' "${REPO_DIR}"
import glob
import os
import sys

repo_dir = sys.argv[1]
# eval_model.py runs with cwd=/workspace/sala-game inside container,
# which maps to REPO_DIR on the host; so outputs land at REPO_DIR/outputs/
paths = glob.glob(os.path.join(repo_dir, "outputs", "*", "summary.json"))
paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(paths[0] if paths else "")
PY
)"

if [[ ${DOCKER_RC} -ne 0 ]]; then
  echo "[run_experiment] docker failed rc=${DOCKER_RC}, writing crash row"
  python3 "${REPO_DIR}/parse_results.py" \
    --results-tsv "${RESULTS_TSV}" \
    --commit "${COMMIT}" \
    --description "${DESCRIPTION}" \
    --s1 0 --s8 0 --smax 0 --accuracy 0 \
    --force-status crash \
    --result-json "${RESULT_JSON}"
  exit ${DOCKER_RC}
fi

if [[ -z "${SUMMARY_JSON}" ]]; then
  echo "[run_experiment] summary.json not found, marking crash"
  python3 "${REPO_DIR}/parse_results.py" \
    --results-tsv "${RESULTS_TSV}" \
    --commit "${COMMIT}" \
    --description "${DESCRIPTION}" \
    --bench-log "${BENCH_LOG}" \
    --accuracy 0 \
    --force-status crash \
    --result-json "${RESULT_JSON}"
  exit 1
fi

python3 "${REPO_DIR}/parse_results.py" \
  --results-tsv "${RESULTS_TSV}" \
  --commit "${COMMIT}" \
  --description "${DESCRIPTION}" \
  --bench-log "${BENCH_LOG}" \
  --summary-json "${SUMMARY_JSON}" \
  --result-json "${RESULT_JSON}"

# Archive summary.json into EXP_DIR for self-contained experiment record
cp -f "${SUMMARY_JSON}" "${EXP_DIR}/summary.json" 2>/dev/null || true

echo "[run_experiment] done: ${RESULT_JSON}"
