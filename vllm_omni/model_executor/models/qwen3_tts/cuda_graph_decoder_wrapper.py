# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

Captures the decoder forward pass at fine-grained exact sizes (T=1..50)
for streaming and coarser bucket sizes for larger inputs.  Graph replay
eliminates kernel-launch overhead during inference.

Inspired by https://github.com/tsdocode/nano-qwen3tts-vllm which captures
one graph per exact decode length for zero-padding-overhead streaming.
"""

import time

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_STATS_LOG_INTERVAL = 20  # log a summary every N decode calls

_DEFAULT_EXACT_SIZES = list(range(1, 51))
_DEFAULT_BUCKET_SIZES = [64, 100, 150, 200, 250, 300, 400, 500]


class CUDAGraphDecoderWrapper:
    """CUDA Graph wrapper with fine-grained exact-match and bucket-padded decode.

    For streaming TTS the decoder is called repeatedly with small, predictable
    input sizes (e.g. 10, 20, 30, 35 frames).  Capturing a graph for each
    exact size lets us replay without any zero-padding overhead.  Larger inputs
    fall back to the smallest captured bucket that fits, with padding.

    All graphs share a single CUDA memory pool to minimise VRAM usage.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        exact_sizes: list[int] | None = None,
        bucket_sizes: list[int] | None = None,
        num_quantizers: int = 16,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self.exact_sizes = sorted(exact_sizes or _DEFAULT_EXACT_SIZES)
        self.bucket_sizes = sorted(bucket_sizes or _DEFAULT_BUCKET_SIZES)
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

        self._graph_pool: tuple | None = None
        self._warmed_up = False

        # Profiling counters
        self._exact_hits = 0
        self._bucket_hits = 0
        self._eager_fallbacks = 0
        self._total_decode_ms = 0.0
        self._call_count = 0

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        if device.type != "cuda":
            logger.info(
                "CUDA Graph warmup skipped: device %s is not CUDA", device,
            )
            return
        if not self.enabled:
            logger.info("CUDA Graph is disabled, skipping warmup")
            return
        if self._warmed_up:
            logger.warning("CUDA Graph already warmed up, skipping")
            return

        self.decoder.eval()

        all_sizes = sorted(set(self.exact_sizes + self.bucket_sizes))
        max_size = max(all_sizes)

        logger.info(
            "CUDA Graph warmup: %d exact sizes (1..%d), %d bucket sizes %s",
            len(self.exact_sizes),
            max(self.exact_sizes) if self.exact_sizes else 0,
            len(self.bucket_sizes),
            self.bucket_sizes,
        )

        # Global warmup at the largest size to stabilise memory allocations.
        with torch.no_grad():
            _ = self.decoder(
                torch.randint(
                    0, 100, (1, self.num_quantizers, max_size),
                    dtype=dtype, device=device,
                )
            )
        torch.cuda.synchronize(device)

        # Capture in reverse order so smaller graphs reuse the pool
        # allocated by larger ones.
        for size in reversed(all_sizes):
            try:
                self._capture_graph(size, device, dtype)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDA Graph for size=%d", size,
                    exc_info=True,
                )

        self._warmed_up = True
        logger.info(
            "CUDA Graph warmup complete. Captured %d graphs.",
            len(self.graphs),
        )

    def _capture_graph(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        static_input = torch.zeros(
            1, self.num_quantizers, size, dtype=dtype, device=device,
        )

        with torch.no_grad():
            _ = self.decoder(static_input)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self._graph_pool):
                static_output = self.decoder(static_input)

        if self._graph_pool is None:
            self._graph_pool = graph.pool()

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode a single code tensor via CUDA graph replay when possible."""
        if not self.enabled or not self._warmed_up or codes.shape[0] != 1:
            self._record("eager", codes.shape[-1])
            return self.decoder(codes)

        actual_size = codes.shape[-1]
        total_upsample = self.decoder.total_upsample
        t0 = time.perf_counter()

        # Fast path: exact-match graph (no padding needed).
        if actual_size in self.graphs:
            self.static_inputs[actual_size].copy_(codes)
            self.graphs[actual_size].replay()
            result = self.static_outputs[actual_size].clone()
            self._record("exact", actual_size, t0)
            return result

        # Fallback: smallest bucket graph that fits.
        bucket = self._find_bucket(actual_size)
        if bucket is not None:
            self.static_inputs[bucket].zero_()
            self.static_inputs[bucket][:, :, :actual_size] = codes
            self.graphs[bucket].replay()
            actual_len = actual_size * total_upsample
            result = self.static_outputs[bucket][..., :actual_len].clone()
            self._record("bucket", actual_size, t0, bucket)
            return result

        self._record("eager", actual_size, t0)
        return self.decoder(codes)

    def _record(
        self,
        path: str,
        size: int,
        t0: float | None = None,
        bucket: int | None = None,
    ):
        if path == "exact":
            self._exact_hits += 1
        elif path == "bucket":
            self._bucket_hits += 1
        else:
            self._eager_fallbacks += 1

        if t0 is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_decode_ms += elapsed_ms
        else:
            elapsed_ms = None

        self._call_count += 1

        if self._call_count <= 5 or self._call_count % _STATS_LOG_INTERVAL == 0:
            avg_ms = (
                self._total_decode_ms / max(self._exact_hits + self._bucket_hits, 1)
            )
            extra = ""
            if path == "bucket" and bucket is not None:
                extra = f" (padded to {bucket})"
            elif path == "exact":
                extra = " (exact match)"
            logger.info(
                "[CUDAGraph Decoder] #%d  path=%s  T=%d%s  "
                "elapsed=%.2fms  | totals: exact=%d bucket=%d eager=%d  "
                "avg=%.2fms",
                self._call_count,
                path,
                size,
                extra,
                elapsed_ms if elapsed_ms is not None else 0.0,
                self._exact_hits,
                self._bucket_hits,
                self._eager_fallbacks,
                avg_ms,
            )

    def _find_bucket(self, actual_size: int) -> int | None:
        for bsize in self.bucket_sizes:
            if actual_size <= bsize and bsize in self.graphs:
                return bsize
        return None

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = (
                left_context_size
                if start_index - left_context_size > 0
                else start_index
            )

            codes_chunk = codes[..., start_index - context_size: end_index]
            wav_chunk = self.decode(codes_chunk)

            wavs.append(wav_chunk[..., context_size * total_upsample:])
            start_index = end_index

        return torch.cat(wavs, dim=-1)
