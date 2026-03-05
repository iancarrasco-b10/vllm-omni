#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh Base                      # Base 0.6B 12Hz (defaults)
#   ./run_server.sh Base 1.7B                 # Base 1.7B 12Hz
#   ./run_server.sh Base 0.6B 25Hz            # Base 0.6B 25Hz
#   ./run_server.sh CustomVoice 1.7B 25Hz     # CustomVoice 1.7B 25Hz

set -e

TASK_TYPE="${1:-CustomVoice}"
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

echo "Starting Qwen3-TTS server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --chat-template "{% for m in messages %}{{ m['content'] }}{% endfor %}" \
    --omni
