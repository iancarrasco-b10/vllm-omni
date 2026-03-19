#!/bin/bash
# Qwen3-TTS Base Voice Clone Benchmark Runner
#
# Benchmarks vllm-omni serving for the Base (voice clone) task type.
# Produces JSON results and optional comparison plots.
#
# Usage:
#   # Full benchmark with default settings:
#   bash run_benchmark_base.sh
#
#   # x-vector only mode (faster, no in-context learning):
#   bash run_benchmark_base.sh --xvec-only
#
#   # Custom reference audio:
#   REF_AUDIO="https://example.com/my_voice.wav" \
#   REF_TEXT="Transcript of the reference audio." \
#   bash run_benchmark_base.sh
#
#   # Custom settings:
#   GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark_base.sh
#
#   # Use 0.6B model:
#   MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base bash run_benchmark_base.sh
#
# Environment variables:
#   GPU_DEVICE       - GPU index to use (default: 0)
#   NUM_PROMPTS      - Number of prompts per concurrency level (default: 50)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   MODEL            - Model name (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
#   PORT             - Server port (default: 8000)
#   GPU_MEM_TALKER   - gpu_memory_utilization for talker stage (default: 0.3)
#   GPU_MEM_CODE2WAV - gpu_memory_utilization for code2wav stage (default: 0.2)
#   STAGE_CONFIG     - Path to stage config YAML (default: configs/qwen3_tts_bs1.yaml)
#   REF_AUDIO        - Reference audio URL for voice cloning
#   REF_TEXT         - Transcript of reference audio

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
GPU_DEVICE="${GPU_DEVICE:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
PORT="${PORT:-8000}"
GPU_MEM_TALKER="${GPU_MEM_TALKER:-0.3}"
GPU_MEM_CODE2WAV="${GPU_MEM_CODE2WAV:-0.2}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/configs/qwen3_tts_bs1.yaml}"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

REF_AUDIO="${REF_AUDIO:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav}"
REF_TEXT="${REF_TEXT:-Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.}"

# Parse args
XVEC_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --xvec-only) XVEC_ONLY=true ;;
    esac
done

mkdir -p "${RESULT_DIR}"

CONFIG_LABEL="base_icl"
if [ "${XVEC_ONLY}" = true ]; then
    CONFIG_LABEL="base_xvec_only"
fi

echo "============================================================"
echo " Qwen3-TTS Base Voice Clone Benchmark"
echo "============================================================"
echo " GPU:          ${GPU_DEVICE}"
echo " Model:        ${MODEL}"
echo " Prompts:      ${NUM_PROMPTS}"
echo " Concurrency:  ${CONCURRENCY}"
echo " Port:         ${PORT}"
echo " Stage config: ${STAGE_CONFIG}"
echo " Mode:         ${CONFIG_LABEL}"
echo " Results:      ${RESULT_DIR}"
echo "============================================================"

prepare_config() {
    local config_template="$1"
    local config_name="$2"
    local output_path="${RESULT_DIR}/${config_name}_stage_config.yaml"

    sed \
        -e "s/devices: \"0\"/devices: \"${GPU_DEVICE}\"/g" \
        -e "s/gpu_memory_utilization: 0.3/gpu_memory_utilization: ${GPU_MEM_TALKER}/g" \
        -e "s/gpu_memory_utilization: 0.2/gpu_memory_utilization: ${GPU_MEM_CODE2WAV}/g" \
        "${config_template}" > "${output_path}"

    echo "${output_path}"
}

start_server() {
    local stage_config="$1"
    local config_name="$2"
    local log_file="${RESULT_DIR}/server_${config_name}_${TIMESTAMP}.log"

    echo ""
    echo "Starting server with config: ${config_name}"
    echo "  Stage config: ${stage_config}"
    echo "  Log file: ${log_file}"

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
        --omni \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --stage-configs-path "${stage_config}" \
        --stage-init-timeout 120 \
        --trust-remote-code \
        --disable-log-stats \
        > "${log_file}" 2>&1 &

    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    echo "  Waiting for server to be ready..."
    local max_wait=300
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server is ready! (waited ${waited}s)"
            return 0
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "  ERROR: Server process died. Check log: ${log_file}"
            tail -20 "${log_file}"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "  ERROR: Server did not start within ${max_wait}s. Check log: ${log_file}"
    kill ${SERVER_PID} 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "  Stopping server (PID: ${SERVER_PID})..."
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        local pids
        pids=$(lsof -ti:${PORT} 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            echo "  Cleaning up remaining processes on port ${PORT}..."
            echo "${pids}" | xargs kill -9 2>/dev/null || true
        fi
        echo "  Server stopped."
        SERVER_PID=""
    fi
}

trap 'stop_server' EXIT

echo ""
echo "============================================================"
echo " Benchmarking: ${CONFIG_LABEL}"
echo "============================================================"

stage_config=$(prepare_config "${SCRIPT_DIR}/${STAGE_CONFIG}" "${CONFIG_LABEL}")
start_server "${stage_config}" "${CONFIG_LABEL}"

# Build concurrency args
conc_args=""
for c in ${CONCURRENCY}; do
    conc_args="${conc_args} ${c}"
done

# Build optional flags
EXTRA_ARGS=""
if [ "${XVEC_ONLY}" = true ]; then
    EXTRA_ARGS="--x-vector-only"
fi

cd "${PROJECT_ROOT}"
python "${SCRIPT_DIR}/vllm_omni/bench_tts_base_serve.py" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency ${conc_args} \
    --num-warmups "${NUM_WARMUPS}" \
    --ref-audio "${REF_AUDIO}" \
    --ref-text "${REF_TEXT}" \
    --config-name "${CONFIG_LABEL}" \
    --result-dir "${RESULT_DIR}" \
    ${EXTRA_ARGS}

stop_server
sleep 5

# Plot results if plot script exists
RESULT_FILE=$(ls -t "${RESULT_DIR}"/bench_${CONFIG_LABEL}_*.json 2>/dev/null | head -1)
if [ -n "${RESULT_FILE}" ] && [ -f "${SCRIPT_DIR}/plot_results.py" ]; then
    echo ""
    echo "============================================================"
    echo " Generating plots..."
    echo "============================================================"

    python "${SCRIPT_DIR}/plot_results.py" \
        --results "${RESULT_FILE}" \
        --labels "${CONFIG_LABEL}" \
        --output "${RESULT_DIR}/qwen3_tts_base_benchmark_${TIMESTAMP}.png"
fi

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}"
echo "============================================================"
