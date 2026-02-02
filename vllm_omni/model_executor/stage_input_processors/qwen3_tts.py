"""Stage input processor for Qwen3-TTS: Talker → SpeechTokenizer transition."""

from typing import Any

import torch


def talker2speech_tokenizer_async_chunk(
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Async-chunk payload extractor for Qwen3-TTS Talker → SpeechTokenizer.

    Stage-0 emits per-step codec codes; they are sent via connector and consumed by Stage-1 as `prompt_token_ids`.
    Returns: `code_predictor_codes` (List[int]) / `codec_streaming` (bool) / `finished` (torch.bool).
    """
    if not isinstance(pooling_output, dict):
        return None

    # `codec_streaming` is the cross-stage streaming toggle (not the official `non_streaming_mode`).
    # It can be overridden per request.
    info = getattr(request, "additional_information_cpu", None)
    if info is None:
        info = getattr(request, "additional_information", None)
    # vLLM may pass additional information as a list for batched requests; Qwen3-TTS typically uses batch=1.
    if isinstance(info, list) and info and isinstance(info[0], dict):
        info = info[0]
    if not isinstance(info, dict):
        info = {}

    def _first(x: object, default: object) -> object:
        if isinstance(x, list):
            return x[0] if x else default
        return x if x is not None else default

    # In async_chunk, Stage-1 consumes only newly scheduled tokens per step; Stage-0 must stream frame-aligned windows.
    # Stage-1 trims left-context each step.
    codec_streaming_val = _first(info.get("codec_streaming"), True)
    codec_streaming = bool(codec_streaming_val) if isinstance(codec_streaming_val, bool) else True
    # Do not override from `pooling_output`: this is a pipeline contract.
    # Mis-overrides can break Stage-1 trim/paste rules.

    # The stop-token step is not a decodable frame; only notify Stage-1 via `finished`.
    finished = False
    try:
        finished = bool(request.is_finished())
    except Exception:
        finished = False

    if finished:
        return {
            "code_predictor_codes": [],
            "codec_streaming": codec_streaming,
            "finished": torch.tensor(True, dtype=torch.bool),
        }

    # Talker AR stage exposes per-step codes as `audio_codes` (shape [T, Q]).
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        # Nothing to send for this step.
        return None

    # `audio_codes` may include prefill/placeholder frames (shape [T,Q]); take only the last frame and skip if all-zero.
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        try:
            if frame.numel() == 0 or not bool(frame.any().item()):
                return None
        except Exception:
            # If `.any()` is unreliable, prefer sending the last frame and let Stage-1 fail-fast on misalignment.
            pass
    elif audio_codes.ndim == 1:
        frame = audio_codes
    else:
        raise ValueError(f"Invalid audio_codes shape for Qwen3-TTS async_chunk: {tuple(audio_codes.shape)}")

    frame = frame.to(torch.long).reshape(-1)
    codec_codes = frame.cpu().tolist()

    return {
        "code_predictor_codes": codec_codes,
        "codec_streaming": codec_streaming,
        "finished": torch.tensor(bool(finished), dtype=torch.bool),
    }
