"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch


def _get_request_info(request: Any) -> dict[str, Any]:
    info = getattr(request, "additional_information_cpu", None)
    if info is None:
        info = getattr(request, "additional_information", None)
    if isinstance(info, list) and info and isinstance(info[0], dict):
        info = info[0]
    return info if isinstance(info, dict) else {}


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Qwen3-TTS async_chunk: {tuple(audio_codes.shape)}")


def talker2code2wav_async_chunk(
    connector: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    if not isinstance(pooling_output, dict):
        return None

    info = _get_request_info(request)
    request_id = request.external_req_id

    codec_streaming_raw = info.get("codec_streaming", True)
    if isinstance(codec_streaming_raw, list):
        codec_streaming_raw = codec_streaming_raw[0] if codec_streaming_raw else True
    codec_streaming = codec_streaming_raw if isinstance(codec_streaming_raw, bool) else True

    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )

    finished = bool(request.is_finished())

    appended_frame = False
    if not finished:
        frame = _extract_last_frame(pooling_output)
        if frame is None:
            return None
        codec_codes = frame.cpu().tolist()
        connector.code_prompt_token_ids[request_id].append(codec_codes)
        appended_frame = True

    length = len(connector.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size

    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size

    if finished and (not appended_frame) and chunk_length == 0:
        return {
            "code_predictor_codes": [],
            "codec_streaming": codec_streaming,
            "codec_context_codes": [],
            "codec_context_frames": 0,
            "codec_total_frames": 0,
            "codec_chunk_frames": 0,
            "codec_num_code_groups": 0,
            "codec_layout": "codebook_major",
            "finished": torch.tensor(True, dtype=torch.bool),
        }

    if length <= 0:
        return {
            "code_predictor_codes": [],
            "codec_streaming": codec_streaming,
            "codec_context_codes": [],
            "codec_context_frames": 0,
            "codec_total_frames": 0,
            "codec_chunk_frames": 0,
            "codec_num_code_groups": 0,
            "codec_layout": "codebook_major",
            "finished": torch.tensor(bool(finished), dtype=torch.bool),
        }

    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = connector.code_prompt_token_ids[request_id][-end_index:]

    if ctx_frames > 0:
        ctx_part = window_frames[:ctx_frames]
        codec_context_codes = torch.tensor(ctx_part).transpose(0, 1).reshape(-1).tolist()
    else:
        codec_context_codes = []

    chunk_part = window_frames[ctx_frames:]
    code_predictor_codes = torch.tensor(chunk_part).transpose(0, 1).reshape(-1).tolist()

    num_code_groups = int(
        len(connector.code_prompt_token_ids[request_id][-1])
        if connector.code_prompt_token_ids[request_id]
        else 0
    )

    return {
        "code_predictor_codes": code_predictor_codes,
        "codec_streaming": codec_streaming,
        "codec_context_codes": codec_context_codes,
        "codec_context_frames": int(ctx_frames),
        "codec_total_frames": int(end_index),
        "codec_chunk_frames": int(context_length),
        "codec_num_code_groups": num_code_groups,
        "codec_layout": "codebook_major",
        "finished": torch.tensor(bool(finished), dtype=torch.bool),
    }
