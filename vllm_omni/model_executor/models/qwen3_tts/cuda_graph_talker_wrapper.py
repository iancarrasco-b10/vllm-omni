# Copyright 2026 The Alibaba Qwen team & contributors.
# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph wrapper for Qwen3-TTS Talker MTP (multi-token prediction).

Captures the entire MTP pipeline (code predictor forward, embedding
summation, text step addition) as a CUDA graph for each supported batch
size.  Graph replay eliminates per-step kernel launch overhead which is
the dominant cost during autoregressive decode.

Based on PR #1925 by @evezhier, extended with multi-batch-size support.
"""

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

logger = init_logger(__name__)

_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]


class TalkerMTPCudaGraphWrapper:
    """CUDA Graph wrapper for talker_mtp with multi-batch support.

    Captures the MTP pipeline for each batch size in ``capture_batch_sizes``.
    At runtime, the smallest captured size >= actual batch is selected and
    inputs are zero-padded to that size.
    """

    def __init__(
        self,
        talker_model,
        talker_config,
        device="cuda",
        enabled=True,
        temperature=0.9,
        top_k=50,
        num_warmup_steps=3,
        capture_batch_sizes: list[int] | None = None,
    ):
        self.device = device
        self.device_index = torch.device(device).index or 0
        self.enabled = enabled

        self.talker = talker_model
        self.code_predictor = talker_model.code_predictor
        self.num_code_groups = talker_config.num_code_groups
        self.hidden_size = talker_config.hidden_size
        self.vocab_size = talker_model.code_predictor.config.vocab_size
        self.temperature = temperature
        self.top_k = top_k
        self.num_warmup_steps = num_warmup_steps
        self.capture_batch_sizes = sorted(capture_batch_sizes or _CAPTURE_BATCH_SIZES)

        self.graphs: dict[int, CUDAGraph] = {}
        self.input_ids_bufs: dict[int, torch.Tensor] = {}
        self.last_id_hidden_bufs: dict[int, torch.Tensor] = {}
        self.past_hidden_bufs: dict[int, torch.Tensor] = {}
        self.text_step_bufs: dict[int, torch.Tensor] = {}
        self.audio_codes_bufs: dict[int, torch.Tensor] = {}
        self.inputs_embeds_out_bufs: dict[int, torch.Tensor] = {}

        self.warmed_up = False

    def _get_padded_bs(self, actual_bs: int) -> int | None:
        for bs in self.capture_batch_sizes:
            if actual_bs <= bs:
                return bs
        return None

    def _alloc_buffers(self, bs: int):
        d, H, Q = self.device, self.hidden_size, self.num_code_groups
        self.input_ids_bufs[bs] = torch.zeros(bs, 1, dtype=torch.long, device=d)
        self.last_id_hidden_bufs[bs] = torch.zeros(bs, 1, H, dtype=torch.bfloat16, device=d)
        self.past_hidden_bufs[bs] = torch.zeros(bs, 1, H, dtype=torch.bfloat16, device=d)
        self.text_step_bufs[bs] = torch.zeros(bs, 1, H, dtype=torch.bfloat16, device=d)
        self.audio_codes_bufs[bs] = torch.zeros(bs, Q, dtype=torch.long, device=d)
        self.inputs_embeds_out_bufs[bs] = torch.zeros(bs, H, dtype=torch.bfloat16, device=d)

    @torch.inference_mode()
    def _mtp_forward(self, bs: int):
        """Run the full MTP pipeline for batch size ``bs``; captured by graph."""
        audio_codes = self.code_predictor.forward(
            layer0_code=self.input_ids_bufs[bs],
            layer0_embed=self.last_id_hidden_bufs[bs],
            last_talker_hidden=self.past_hidden_bufs[bs],
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        self.audio_codes_bufs[bs].copy_(audio_codes)

        layer0 = self.audio_codes_bufs[bs][:, :1]
        invalid0 = (layer0 < 0) | (layer0 >= int(self.vocab_size))
        self.audio_codes_bufs[bs].masked_fill_(invalid0.expand_as(self.audio_codes_bufs[bs]), 0)
        residual_ids = self.audio_codes_bufs[bs][:, 1:]

        embeds = [self.last_id_hidden_bufs[bs]]
        for i in range(self.num_code_groups - 1):
            emb = self.code_predictor.get_input_embeddings()[i](residual_ids[:, i : i + 1])
            embeds.append(emb)

        summed = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        result = (summed + self.text_step_bufs[bs]).reshape(bs, -1)
        self.inputs_embeds_out_bufs[bs].copy_(result)

    def _capture_for_bs(self, bs: int):
        self._alloc_buffers(bs)
        for _ in range(self.num_warmup_steps):
            self._mtp_forward(bs)
        torch.cuda.synchronize(self.device)

        with torch.cuda.device(self.device_index):
            graph = CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._mtp_forward(bs)
                s.synchronize()
                with torch.cuda.graph(graph):
                    self._mtp_forward(bs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graphs[bs] = graph

    def warmup(self, device: torch.device):
        if not self.enabled:
            logger.info("TalkerMTPCudaGraphWrapper: disabled, skipping capture")
            return
        if device.type != "cuda":
            logger.info("CUDA Graph warmup skipped: device %s is not CUDA", device)
            return
        if self.warmed_up:
            return
        self.device = device
        self.device_index = device.index or 0

        logger.info(
            "TalkerMTPCudaGraphWrapper: capturing graphs for batch sizes %s",
            self.capture_batch_sizes,
        )
        for bs in self.capture_batch_sizes:
            try:
                self._capture_for_bs(bs)
                logger.info("  Captured Talker MTP graph for bs=%d", bs)
            except Exception:
                logger.warning("  Failed to capture Talker MTP graph for bs=%d", bs, exc_info=True)

        self.warmed_up = True
        logger.info(
            "TalkerMTPCudaGraphWrapper: captured %d graphs (bs=%s)",
            len(self.graphs),
            sorted(self.graphs.keys()),
        )

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: torch.Tensor,
        last_id_hidden: torch.Tensor,
        past_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actual_bs = input_ids.shape[0]
        padded_bs = self._get_padded_bs(actual_bs)

        if padded_bs is None or padded_bs not in self.graphs:
            return self.talker._talker_mtp_eager(input_ids, last_id_hidden, past_hidden, text_step)

        self.input_ids_bufs[padded_bs].zero_()
        self.last_id_hidden_bufs[padded_bs].zero_()
        self.past_hidden_bufs[padded_bs].zero_()
        self.text_step_bufs[padded_bs].zero_()

        self.input_ids_bufs[padded_bs][:actual_bs].copy_(input_ids.reshape(actual_bs, 1))
        self.last_id_hidden_bufs[padded_bs][:actual_bs].copy_(last_id_hidden.reshape(actual_bs, 1, -1))
        self.past_hidden_bufs[padded_bs][:actual_bs].copy_(past_hidden.reshape(actual_bs, 1, -1))
        self.text_step_bufs[padded_bs][:actual_bs].copy_(text_step.reshape(actual_bs, 1, -1))

        self.graphs[padded_bs].replay()

        return (
            self.inputs_embeds_out_bufs[padded_bs][:actual_bs].clone(),
            self.audio_codes_bufs[padded_bs][:actual_bs].clone(),
        )
