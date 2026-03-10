#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                           # Default: CustomVoice model
#   ./run_server.sh CustomVoice               # CustomVoice model
#   ./run_server.sh VoiceDesign               # VoiceDesign model
#   ./run_server.sh Base                      # Base (voice clone) model
#   ./run_server.sh /path/to/local/checkpoint # Local checkpoint directory
#   ./run_server.sh org/model-name            # Hugging Face model id

set -e

MODEL_INPUT="${1:-CustomVoice}"
MODEL_SOURCE="preset"

case "$MODEL_INPUT" in
    CustomVoice)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;
    VoiceDesign)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ;;
    Base)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;
    *)
        if [ -d "$MODEL_INPUT" ]; then
            MODEL="$MODEL_INPUT"
            MODEL_SOURCE="local"
        elif [[ "$MODEL_INPUT" == */* ]]; then
            MODEL="$MODEL_INPUT"
            MODEL_SOURCE="hf"
        else
            echo "Unknown model argument: $MODEL_INPUT"
            echo "Supported presets: CustomVoice, VoiceDesign, Base"
            echo "Or pass a local checkpoint directory path."
            echo "Or pass a Hugging Face model id (e.g. org/model-name)."
            exit 1
        fi
        ;;
esac

if [ "$MODEL_SOURCE" = "local" ]; then
    echo "Starting Qwen3-TTS server with local checkpoint: $MODEL"
elif [ "$MODEL_SOURCE" = "hf" ]; then
    echo "Starting Qwen3-TTS server with Hugging Face model: $MODEL"
else
    echo "Starting Qwen3-TTS server with model: $MODEL"
fi

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --chat-template "{% for m in messages %}{{ m['content'] }}{% endfor %}" \
    --omni
