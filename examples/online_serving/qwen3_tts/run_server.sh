#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh Base                      # Base 0.6B 12Hz (defaults)
#   ./run_server.sh Base 1.7B                 # Base 1.7B 12Hz
#   ./run_server.sh Base 0.6B 25Hz            # Base 0.6B 25Hz
#   ./run_server.sh CustomVoice 1.7B 25Hz     # CustomVoice 1.7B 25Hz
#   ./run_server.sh Qwen/Qwen3-TTS-12Hz-0.6B-Base   # Direct HF model path
#   ./run_server.sh /local/path/to/model             # Local model path

set -e

ARG1="${1:-CustomVoice}"

# If the first arg contains a slash, treat it as a direct model path.
if [[ "$ARG1" == */* ]]; then
    MODEL="$ARG1"
else
    TASK_TYPE="$ARG1"
    SIZE="${2:-}"
    HZ="${3:-12Hz}"

    case "$TASK_TYPE" in
        CustomVoice)
            SIZE="${SIZE:-1.7B}"
            MODEL="Qwen/Qwen3-TTS-${HZ}-${SIZE}-CustomVoice"
            ;;
        VoiceDesign)
            SIZE="${SIZE:-1.7B}"
            MODEL="Qwen/Qwen3-TTS-${HZ}-${SIZE}-VoiceDesign"
            ;;
        Base)
            SIZE="${SIZE:-0.6B}"
            MODEL="Qwen/Qwen3-TTS-${HZ}-${SIZE}-Base"
            ;;
        *)
            echo "Unknown task type: $TASK_TYPE"
            echo "Supported: CustomVoice, VoiceDesign, Base"
            exit 1
            ;;
    esac
fi

echo "Starting Qwen3-TTS server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --chat-template "{% for m in messages %}{{ m['content'] }}{% endfor %}" \
    --omni