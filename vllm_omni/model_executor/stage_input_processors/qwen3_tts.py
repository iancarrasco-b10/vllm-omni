"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)

logger = init_logger(__name__)


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        # audio_codes shape: [num_frames, Q] where Q=num_quantizers (16)
        audio_codes = output.multimodal_output["audio_codes"].to(torch.long)
        # Filter invalid frames: zero-padded (EOS) and frames containing
        # out-of-range values (e.g. stop_token_id=2150 exceeds codebook_size=2048).
        _CODEBOOK_SIZE = 2048
        valid_mask = audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
        audio_codes = audio_codes[valid_mask]
        ref_code = output.multimodal_output.get("ref_code")
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            ref_code = ref_code.to(torch.long).cpu().contiguous()
            ref_code_len = int(ref_code.shape[0])
            audio_codes = torch.cat([ref_code.to(audio_codes.device), audio_codes], dim=0)
        else:
            ref_code_len = 0
        # Code2Wav expects codebook-major flat: [Q*num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        additional_information = {"left_context_size": [ref_code_len]} if ref_code_len > 0 else None
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )
    return code2wav_inputs


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
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            # frame is already on CPU (batch-copied in sample_tokens); .tolist() is cheap.
            codec_codes = frame.tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
        ref_code = pooling_output.get("ref_code")
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0 and request_payload.get(request_id) is None:
            # ref_code is already on CPU from _update_intermediate_buffer.
            request_payload[request_id] = ref_code.to(torch.long).contiguous()
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    # Per-request override takes priority over dynamic IC.
    per_request_override = False
    initial_chunk_size = 0
    additional_information = getattr(request, "additional_information", None)

    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
            per_request_override = True

    # Dynamic IC: cache per request so boundaries stay stable for its lifetime.
    if not per_request_override:
        _ic_cache = getattr(transfer_manager, "_cached_ic", None)
        if _ic_cache is None:
            _ic_cache = {}
            transfer_manager._cached_ic = _ic_cache
        if request_id not in _ic_cache:
            max_ic = max_ic_for_chunk_size(chunk_size)
            active = sum(1 for v in transfer_manager.code_prompt_token_ids.values() if len(v) > 0)
            # ic_capacity decouples IC sizing from max_batch_size (scheduler_max_num_seqs),
            # which can be much larger than typical concurrency, collapsing IC to tiny values.
            ic_capacity = int(cfg.get("ic_capacity", 0))
            if ic_capacity <= 0:
                ic_capacity = min(getattr(transfer_manager, "scheduler_max_num_seqs", 1), 16)
            _ic_cache[request_id] = compute_dynamic_initial_chunk_size(active, ic_capacity, max_ic)
        initial_chunk_size = _ic_cache[request_id]

    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )

    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping to codec_chunk_frames.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": True,
            }
        return None

    in_initial_phase = initial_chunk_size > 0 and initial_chunk_size < chunk_size and length < chunk_size

    if in_initial_phase:
        # IC phase: emit every initial_chunk_size frames with growing left context.
        if not finished and length % initial_chunk_size != 0:
            return None
        context_length = (
            length % initial_chunk_size if (finished and length % initial_chunk_size != 0) else initial_chunk_size
        )
    else:
        # Normal phase: offset so the first normal emit picks up after IC phase.
        # IC is stateless (may change with load); any mismatch is absorbed by left_context.
        initial_coverage = (
            ((chunk_size - 1) // initial_chunk_size) * initial_chunk_size if 0 < initial_chunk_size < chunk_size else 0
        )
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Build ref_code context.  To avoid re-serializing ref_code (which can be
    # 120+ frames × 16 quantizers) through SHM on every chunk, we send it only
    # once on the first chunk as a separate field.  The receiving side caches
    # it and prepends locally for Code2Wav decoding.
    ref_code = request_payload.get(request_id)
    ref_code_payload: list[int] | None = None
    ref_code_num_frames = 0
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        _ref_list_cache = getattr(transfer_manager, "_ref_code_list_cache", None)
        if _ref_list_cache is None:
            _ref_list_cache = {}
            transfer_manager._ref_code_list_cache = _ref_list_cache
        ref_frames = _ref_list_cache.get(request_id)
        if ref_frames is None:
            ref_frames = ref_code.tolist()
            _ref_list_cache[request_id] = ref_frames
        ref_code_num_frames = len(ref_frames)
        left_context_size += ref_code_num_frames

        _sent = getattr(transfer_manager, "_ref_code_sent", None)
        if _sent is None:
            _sent = set()
            transfer_manager._ref_code_sent = _sent
        if request_id not in _sent:
            # First chunk: include ref_code as a flat codebook-major list.
            num_q = len(ref_frames[0])
            ref_code_payload = [ref_frames[f][q_] for q_ in range(num_q) for f in range(ref_code_num_frames)]
            _sent.add(request_id)

    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)
    code_predictor_codes = [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)]

    result: dict[str, Any] = {
        "code_predictor_codes": code_predictor_codes,
        "left_context_size": left_context_size,
        "finished": finished,
    }
    if ref_code_payload is not None:
        result["ref_code_data"] = ref_code_payload
        result["ref_code_num_frames"] = ref_code_num_frames
    elif ref_code_num_frames > 0:
        result["ref_code_num_frames"] = ref_code_num_frames
    return result
