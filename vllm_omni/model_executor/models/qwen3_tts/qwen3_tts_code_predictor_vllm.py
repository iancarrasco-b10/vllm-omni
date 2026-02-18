from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec, KVCacheTensor
from vllm.v1.worker.gpu import attn_utils

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

logger = init_logger(__name__)


def _build_rope_cache(head_dim: int, max_seq: int, theta: float, device: torch.device, dtype: torch.float32):
    """Pre-compute RoPE cos/sin tables: [max_seq, head_dim/2]."""
    dim_half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, dtype=torch.float32, device=device) / dim_half))
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)  # [max_seq, dim_half]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    """Apply RoPE to x: [batch, heads, seq, head_dim]. positions: [batch, seq] or [seq]."""
    dim_half = x.shape[-1] // 2
    # cos/sin are [max_seq, dim_half]; gather by position
    if positions.dim() == 1:
        c = cos[positions]  # [seq, dim_half]
        s = sin[positions]
        c = c.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim_half]
        s = s.unsqueeze(0).unsqueeze(0)
    else:
        c = cos[positions].unsqueeze(1)  # [batch, 1, seq, dim_half]
        s = sin[positions].unsqueeze(1)
    x1 = x[..., :dim_half]
    x2 = x[..., dim_half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Simple RMS normalization."""
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + eps)
    return (x_normed * weight).to(x.dtype)


class _LocalPredictorKVCache:
    """Minimal local KV cache + attention metadata for running
    code_predictor inside one worker (independent of engine KV)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        max_seq_len: int,
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device

        # Collect attention layers registered in this vllm_config.
        kv_cache_spec_by_layer = attn_utils.get_kv_cache_spec(vllm_config)
        if not kv_cache_spec_by_layer:
            raise RuntimeError("Local predictor KVCache requires vLLM Attention layers to be registered.")

        # We only need enough blocks for a tiny per-frame sequence (<= max_seq_len).
        any_spec = next(iter(kv_cache_spec_by_layer.values()))
        block_size = int(any_spec.block_size)
        blocks_per_seq = (int(max_seq_len) + block_size - 1) // block_size
        num_blocks = max(1, int(max_batch_size) * int(blocks_per_seq))

        # Allocate per-layer KV caches (small, independent).
        kv_cache_tensors: list[KVCacheTensor] = []
        for layer_name, spec in kv_cache_spec_by_layer.items():
            kv_cache_tensors.append(KVCacheTensor(size=int(spec.page_size_bytes) * num_blocks, shared_by=[layer_name]))

        merged_spec: KVCacheSpec = KVCacheSpec.merge(list(kv_cache_spec_by_layer.values()))
        self.kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=list(kv_cache_spec_by_layer.keys()), kv_cache_spec=merged_spec)
            ],
        )

        # Init backend + bind KV cache tensors to attention modules.
        self.attn_backends, self.attn_metadata_builders = attn_utils.init_attn_backend(
            self.kv_cache_config, vllm_config, device
        )
        self.runner_kv_caches: list[torch.Tensor] = []
        attn_utils.init_kv_cache(
            self.runner_kv_caches,
            vllm_config.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            device,
        )

        # Precompute a fixed block table mapping for the maximum batch.
        self.block_size = block_size
        self.blocks_per_seq = blocks_per_seq
        self.max_batch_size = int(max_batch_size)

        bt = torch.full((self.max_batch_size, self.blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(self.max_batch_size):
            for j in range(self.blocks_per_seq):
                bt[i, j] = i * self.blocks_per_seq + j
        self._block_table = bt

    def build_attn_metadata(
        self,
        *,
        num_reqs: int,
        query_lens: torch.Tensor,  # (num_reqs,) int32 on cpu
        seq_lens: torch.Tensor,  # (num_reqs,) int32 on cpu
    ) -> tuple[dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
        """Build attention metadata, positions, and slot_mapping dict.

        Returns:
            (attn_metadata, positions, slot_mappings_by_layer)
            - attn_metadata: per-layer attention metadata for attn backends.
            - positions: (num_tokens,) position IDs on device.
            - slot_mappings_by_layer: {layer_name: slot_mapping_tensor} for
              set_forward_context so that unified_kv_cache_update can write
              the KV cache correctly.
        """
        num_reqs = int(num_reqs)
        if num_reqs <= 0:
            return {}, torch.empty((0,), dtype=torch.int64, device=self.device), {}
        if num_reqs > self.max_batch_size:
            raise ValueError(f"num_reqs={num_reqs} exceeds local predictor max_batch_size={self.max_batch_size}")

        query_lens_i32 = query_lens.to(dtype=torch.int32, device="cpu")
        seq_lens_i32 = seq_lens.to(dtype=torch.int32, device="cpu")

        # query_start_loc: prefix sums of query_lens.
        qsl = torch.zeros((num_reqs + 1,), dtype=torch.int32, device="cpu")
        qsl[1:] = torch.cumsum(query_lens_i32, dim=0)
        num_tokens = int(qsl[-1].item())
        if num_tokens <= 0:
            return {}, torch.empty((0,), dtype=torch.int64, device=self.device), {}

        # positions: for each request i, emit positions [seq_len-query_len .. seq_len-1]
        pos_list: list[torch.Tensor] = []
        for i in range(num_reqs):
            ql = int(query_lens_i32[i].item())
            sl = int(seq_lens_i32[i].item())
            start = sl - ql
            pos_list.append(torch.arange(start, sl, dtype=torch.int64))
        positions_cpu = torch.cat(pos_list, dim=0)

        # slot_mapping: map each query token to a physical slot in the paged KV cache.
        # We allocate per-request contiguous blocks; slot = base + position.
        slot_mapping = torch.empty((num_tokens,), dtype=torch.int64, device="cpu")
        cursor = 0
        for i in range(num_reqs):
            ql = int(query_lens_i32[i].item())
            sl = int(seq_lens_i32[i].item())
            start = sl - ql
            for p in range(start, sl):
                block_idx = p // self.block_size
                offset = p % self.block_size
                block_id = int(self._block_table[i, block_idx].item())
                slot_mapping[cursor] = block_id * self.block_size + offset
                cursor += 1

        max_seq_len = int(seq_lens_i32[:num_reqs].max().item())
        query_start_loc_gpu = qsl.to(device=self.device)
        seq_lens_gpu = seq_lens_i32.to(device=self.device)
        block_table = self._block_table[:num_reqs].contiguous()
        slot_mapping_gpu = slot_mapping.to(device=self.device)

        attn_metadata = attn_utils.build_attn_metadata(
            self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=qsl,
            seq_lens=seq_lens_gpu,
            max_seq_len=max_seq_len,
            block_tables=[block_table],
            slot_mappings=[slot_mapping_gpu],
            kv_cache_config=self.kv_cache_config,
        )

        # Build slot_mappings_by_layer for set_forward_context.
        # Fix for vllm 0.15.0
        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for kv_cache_group in self.kv_cache_config.kv_cache_groups:
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping_gpu

        return attn_metadata, positions_cpu.to(device=self.device), slot_mappings_by_layer


class Qwen3TTSTalkerCodePredictorModelVLLM(nn.Module):
    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        talker_hidden_size: int | None = None,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config, cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Official code_predictor uses one embedding table per residual group.
        # Some Qwen3-TTS checkpoints store codec embeddings in the talker hidden
        # space, even when `code_predictor_config.hidden_size` is smaller.
        # We keep the embedding dim aligned with the checkpoint and project down
        # via `small_to_mtp_projection` in the wrapper module.
        emb_dim = int(talker_hidden_size) if talker_hidden_size is not None else int(config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(self, positions: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Token-major: [num_tokens, hidden]
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Match vLLM Qwen2/Qwen3 packing conventions: q_proj/k_proj/v_proj -> qkv_proj,
        # gate_proj/up_proj -> gate_up_proj.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped.endswith("scale"):
                    mapped = maybe_remap_kv_scale_name(mapped, params_dict)
                    if mapped is None:
                        continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                mapped = maybe_remap_kv_scale_name(name, params_dict)
                if mapped is None:
                    continue
                if name.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
        return loaded_params


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(nn.Module):
    """vLLM-native code_predictor used by the AR talker (residual codebooks)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = config
        self.talker_config = talker_config

        # Keep module/weight names aligned with official checkpoint (talker.code_predictor.model.*).
        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.model",
        )

        # One head per residual group.
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        self._kv_cache: _LocalPredictorKVCache | None = None
        self._fast_ready = False

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    # ---- Fast SDPA path (bypasses vLLM attention infrastructure) ----

    def _init_fast_path(self, device: torch.device) -> None:
        """One-time setup: pre-compute RoPE and allocate KV cache tensors."""
        if self._fast_ready:
            return
        cfg = self.config
        num_layers = cfg.num_hidden_layers
        num_q_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_q_heads)
        max_seq = cfg.num_code_groups + 2  # prefill(2) + decode(Q-1)
        rope_theta = getattr(cfg, "rope_theta", 10000.0)
        eps = cfg.rms_norm_eps

        self._fast_num_layers = num_layers
        self._fast_num_q_heads = num_q_heads
        self._fast_num_kv_heads = num_kv_heads
        self._fast_head_dim = head_dim
        self._fast_q_size = num_q_heads * head_dim
        self._fast_kv_size = num_kv_heads * head_dim
        self._fast_gqa_groups = num_q_heads // num_kv_heads
        self._fast_eps = eps
        self._fast_max_seq = max_seq
        self._fast_hidden = cfg.hidden_size
        self._fast_intermediate = cfg.intermediate_size

        self._fast_rope_cos, self._fast_rope_sin = _build_rope_cache(
            head_dim, max_seq, rope_theta, device, torch.bfloat16
        )

        # KV cache: [num_layers, 2(k,v), max_batch=1, max_seq, kv_heads, head_dim]
        self._fast_k_cache = torch.zeros(
            num_layers, 1, max_seq, num_kv_heads, head_dim,
            dtype=torch.bfloat16, device=device,
        )
        self._fast_v_cache = torch.zeros(
            num_layers, 1, max_seq, num_kv_heads, head_dim,
            dtype=torch.bfloat16, device=device,
        )

        # Pre-compute causal mask for SDPA (avoids recomputation per call)
        self._fast_causal_mask = torch.tril(
            torch.ones(max_seq, max_seq, dtype=torch.bool, device=device)
        )

        # CUDA graph infrastructure for decode steps
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._decode_graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self._graph_warmup_done = False

        self._fast_ready = True
        logger.info(
            "[FastCodePredictor] Initialized: layers=%d q_heads=%d kv_heads=%d "
            "head_dim=%d hidden=%d max_seq=%d",
            num_layers, num_q_heads, num_kv_heads, head_dim, cfg.hidden_size, max_seq,
        )

    @torch.inference_mode()
    def _fast_layer_forward(
        self, layer_idx: int, hidden: torch.Tensor, positions: torch.Tensor,
        seq_len: int, tok_count: int,
    ) -> torch.Tensor:
        """Run one transformer layer using SDPA. hidden: [1, tok_count, H]."""
        layer = self.model.layers[layer_idx]
        eps = self._fast_eps
        num_q = self._fast_num_q_heads
        num_kv = self._fast_num_kv_heads
        hd = self._fast_head_dim
        q_size = self._fast_q_size
        kv_size = self._fast_kv_size
        inter = self._fast_intermediate

        # Pre-attention norm
        residual = hidden
        hidden = _rms_norm(hidden, layer.input_layernorm.weight, eps)

        # QKV projection (fused [q_size + 2*kv_size, hidden])
        qkv = F.linear(hidden, layer.self_attn.qkv_proj.weight)
        q = qkv[..., :q_size].reshape(1, tok_count, num_q, hd).transpose(1, 2)
        k = qkv[..., q_size:q_size+kv_size].reshape(1, tok_count, num_kv, hd).transpose(1, 2)
        v = qkv[..., q_size+kv_size:].reshape(1, tok_count, num_kv, hd).transpose(1, 2)

        # QK norms (per-head RMS norm)
        q = _rms_norm(q, layer.self_attn.q_norm.weight, eps)
        k = _rms_norm(k, layer.self_attn.k_norm.weight, eps)

        # RoPE
        q = _apply_rope(q, self._fast_rope_cos, self._fast_rope_sin, positions)
        k = _apply_rope(k, self._fast_rope_cos, self._fast_rope_sin, positions)

        # Store new K, V into cache
        start = seq_len - tok_count
        self._fast_k_cache[layer_idx, 0, start:seq_len] = k[0].transpose(0, 1)
        self._fast_v_cache[layer_idx, 0, start:seq_len] = v[0].transpose(0, 1)

        # Attention over full cache [0:seq_len] with native GQA support
        k_full = self._fast_k_cache[layer_idx, :1, :seq_len].transpose(1, 2)
        v_full = self._fast_v_cache[layer_idx, :1, :seq_len].transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=(tok_count > 1), enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(1, tok_count, q_size)

        # Output projection
        attn_out = F.linear(attn_out, layer.self_attn.o_proj.weight)
        hidden = residual + attn_out

        # Post-attention norm + MLP
        residual = hidden
        hidden = _rms_norm(hidden, layer.post_attention_layernorm.weight, eps)

        gate_up = F.linear(hidden, layer.mlp.gate_up_proj.weight)
        gate = gate_up[..., :inter]
        up = gate_up[..., inter:]
        hidden = F.silu(gate) * up
        hidden = F.linear(hidden, layer.mlp.down_proj.weight)
        hidden = residual + hidden

        return hidden

    @torch.inference_mode()
    def _fast_model_forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor, seq_len: int, tok_count: int) -> torch.Tensor:
        """Run all layers + final norm. inputs_embeds: [1, tok_count, H]."""
        hidden = inputs_embeds
        for i in range(self._fast_num_layers):
            hidden = self._fast_layer_forward(i, hidden, positions, seq_len, tok_count)
        hidden = _rms_norm(hidden, self.model.norm.weight, self._fast_eps)
        return hidden

    def _capture_decode_graphs(self, device: torch.device) -> None:
        """Capture a CUDA graph per decode step (step 1..Q-2). Each graph does:
        projection → 5 layers → norm → lm_head[step].
        """
        if self._graph_warmup_done:
            return
        num_groups = int(self.config.num_code_groups)
        h_pred = self._fast_hidden
        vocab = self.config.vocab_size

        logger.info("[FastCodePredictor] Capturing %d CUDA graphs...", num_groups - 2)

        # Warmup: run each step eagerly a few times
        for _ in range(3):
            self._fast_k_cache.zero_()
            self._fast_v_cache.zero_()
            # Prefill warmup
            dummy_pf = torch.zeros(1, 2, h_pred, dtype=torch.bfloat16, device=device)
            self._fast_model_forward(dummy_pf, self._prefill_pos, seq_len=2, tok_count=2)
            # Decode warmup
            for step in range(1, num_groups - 1):
                dummy_in = torch.zeros(1, 1, h_pred, dtype=torch.bfloat16, device=device)
                self._pos_buf[0] = 1 + step
                self._fast_model_forward(dummy_in, self._pos_buf, seq_len=2 + step, tok_count=1)

        # Now capture a graph per decode step
        for step in range(1, num_groups - 1):
            # Reset KV cache and run prefill to populate positions [0:2]
            self._fast_k_cache.zero_()
            self._fast_v_cache.zero_()
            dummy_pf = torch.zeros(1, 2, h_pred, dtype=torch.bfloat16, device=device)
            self._fast_model_forward(dummy_pf, self._prefill_pos, seq_len=2, tok_count=2)
            # Fill preceding decode positions in KV cache
            for prev in range(1, step):
                dummy_in = torch.zeros(1, 1, h_pred, dtype=torch.bfloat16, device=device)
                self._pos_buf[0] = 1 + prev
                self._fast_model_forward(dummy_in, self._pos_buf, seq_len=2 + prev, tok_count=1)

            seq_len_for_step = 2 + step
            static_in = torch.zeros(1, 1, h_pred, dtype=torch.bfloat16, device=device)
            self._pos_buf[0] = seq_len_for_step - 1

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._graph_pool):
                hidden = self._fast_model_forward(static_in, self._pos_buf, seq_len=seq_len_for_step, tok_count=1)
                static_out = self.lm_head[step](hidden[:, -1:, :]).squeeze(1)

            self._decode_graphs[step] = (g, static_in, static_out)

        self._graph_warmup_done = True
        logger.info("[FastCodePredictor] Captured %d CUDA graphs", len(self._decode_graphs))

    @torch.inference_mode()
    def fast_forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Fast SDPA-based prediction of residual codebooks 1..Q-1 with CUDA graph replay."""
        device = layer0_code.device
        self._init_fast_path(device)

        num_groups = int(self.config.num_code_groups)
        max_steps = num_groups - 1

        if not hasattr(self, '_pos_buf'):
            self._pos_buf = torch.zeros(1, dtype=torch.long, device=device)
            self._prefill_pos = torch.arange(2, dtype=torch.long, device=device)
            self._codes_buf = torch.zeros(1, num_groups, dtype=torch.long, device=device)

        # Try to capture graphs on first real call
        self._capture_decode_graphs(device)
        use_graphs = self._graph_warmup_done and len(self._decode_graphs) > 0

        # Zero KV cache
        self._fast_k_cache.zero_()
        self._fast_v_cache.zero_()

        # Prefill
        prefill_input = torch.cat([last_talker_hidden, layer0_embed], dim=1).to(torch.bfloat16)
        prefill_input = self.small_to_mtp_projection(prefill_input)
        hidden = self._fast_model_forward(prefill_input, self._prefill_pos, seq_len=2, tok_count=2)
        logits = self.lm_head[0](hidden[:, -1:, :]).squeeze(1)

        self._codes_buf[0, 0] = layer0_code.reshape(-1)[0]
        seq_len = 2
        embeddings = self.model.get_input_embeddings()
        lm_heads = self.lm_head
        projection = self.small_to_mtp_projection
        inv_temp = 1.0 / temperature if do_sample and temperature > 0 else 0.0

        for step in range(1, num_groups):
            if inv_temp > 0:
                scaled = logits * inv_temp
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            self._codes_buf[0, step] = next_id.reshape(-1)[0]

            if step < max_steps:
                # Embed + project (outside graph since depends on sampled token)
                tok_embed = embeddings[step - 1](next_id.long())
                tok_embed = projection(tok_embed.to(torch.bfloat16))  # [1, 1, H_pred]

                if use_graphs and step in self._decode_graphs:
                    g, static_in, static_out = self._decode_graphs[step]
                    static_in.copy_(tok_embed)
                    g.replay()
                    logits = static_out
                else:
                    self._pos_buf[0] = seq_len
                    hidden = self._fast_model_forward(tok_embed, self._pos_buf, seq_len=seq_len + 1, tok_count=1)
                    logits = lm_heads[step](hidden[:, -1:, :]).squeeze(1)
                seq_len += 1

        return self._codes_buf.clone()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Ensure all vLLM custom layers consult the predictor vllm_config
        # (esp. for Attention static_forward_context).
        with set_current_vllm_config(self._vllm_config):
            loaded: set[str] = set()
            model_weights: list[tuple[str, torch.Tensor]] = []
            other_weights: list[tuple[str, torch.Tensor]] = []
            for name, w in weights:
                if name.startswith("model."):
                    model_weights.append((name[len("model.") :], w))
                else:
                    other_weights.append((name, w))

            loaded_model = self.model.load_weights(model_weights)
            loaded |= {f"model.{n}" for n in loaded_model}

            params = dict(self.named_parameters(remove_duplicate=False))
            for name, w in other_weights:
                if name not in params:
                    continue
                default_weight_loader(params[name], w)
                loaded.add(name)
            return loaded

    def _maybe_init_kv_cache(self, device: torch.device) -> None:
        if self._kv_cache is not None:
            return
        max_seq_len = int(getattr(self.config, "num_code_groups", 16) or 16)
        # Upper bound on batch size: vLLM scheduler max_num_seqs (fallback 8).
        max_batch = int(getattr(self._vllm_config.scheduler_config, "max_num_seqs", 8) or 8)
        max_batch = max(1, max_batch)
        self._kv_cache = _LocalPredictorKVCache(
            vllm_config=self._vllm_config,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch,
            device=device,
        )

    @torch.inference_mode()
    def reset_cache(self) -> None:
        # We reuse a fixed kv cache buffer and overwrite starting at slot 0.
        # No action required here (seq_lens controls what is read).
        return

    @torch.inference_mode()
    def prefill_logits(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Prefill with 2 tokens: [past_hidden, layer0_embed]. Returns logits for residual group 0."""
        self._maybe_init_kv_cache(inputs_embeds.device)
        assert self._kv_cache is not None

        bsz = int(inputs_embeds.shape[0])
        qlen = 2
        # Flatten to token-major.
        hs = inputs_embeds.to(dtype=torch.bfloat16).reshape(bsz * qlen, -1)
        hs = self.small_to_mtp_projection(hs)

        query_lens = torch.full((bsz,), qlen, dtype=torch.int32)
        seq_lens = query_lens.clone()
        attn_metadata, positions, slot_mappings = self._kv_cache.build_attn_metadata(
            num_reqs=bsz, query_lens=query_lens, seq_lens=seq_lens
        )

        with (
            set_current_vllm_config(self._vllm_config),
            set_forward_context(
                attn_metadata,
                self._vllm_config,
                num_tokens=int(hs.shape[0]),
                slot_mapping=slot_mappings,
            ),
        ):
            out = self.model(positions=positions, inputs_embeds=hs)

        # Gather last token per request.
        last_idx = torch.arange(qlen - 1, bsz * qlen, step=qlen, device=out.device, dtype=torch.long)
        last_h = out.index_select(0, last_idx)
        logits = self.lm_head[0](last_h)
        return logits

    @torch.inference_mode()
    def decode_logits(self, input_ids: torch.Tensor, *, generation_step: int, past_seq_len: int) -> torch.Tensor:
        """Decode one new token for residual group `generation_step` (1..Q-1)."""
        self._maybe_init_kv_cache(input_ids.device)
        assert self._kv_cache is not None
        bsz = int(input_ids.shape[0])
        if generation_step <= 0:
            raise ValueError("generation_step must be >= 1 for decode_logits")

        embed_idx = generation_step - 1
        hs = self.model.get_input_embeddings()[embed_idx](input_ids.to(dtype=torch.long).reshape(bsz, 1))
        hs = self.small_to_mtp_projection(hs.reshape(bsz, -1))

        query_lens = torch.ones((bsz,), dtype=torch.int32)
        seq_lens = torch.full((bsz,), int(past_seq_len) + 1, dtype=torch.int32)
        attn_metadata, positions, slot_mappings = self._kv_cache.build_attn_metadata(
            num_reqs=bsz, query_lens=query_lens, seq_lens=seq_lens
        )

        with (
            set_current_vllm_config(self._vllm_config),
            set_forward_context(
                attn_metadata,
                self._vllm_config,
                num_tokens=int(hs.shape[0]),
                slot_mapping=slot_mappings,
            ),
        ):
            out = self.model(positions=positions, inputs_embeds=hs)

        logits = self.lm_head[generation_step](out)
        return logits

    _fwd_call_count = 0
    _fwd_total_ms = 0.0

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Dispatches to fast SDPA path (bsz=1) or legacy vLLM-attention path."""
        import time as _time

        bsz = int(layer0_code.shape[0])
        if bsz == 1:
            _t0 = _time.perf_counter()
            result = self.fast_forward(
                layer0_code, layer0_embed, last_talker_hidden,
                do_sample, temperature, top_k, top_p,
            )
            _t1 = _time.perf_counter()
            _ms = (_t1 - _t0) * 1000.0

            cls = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM
            cls._fwd_call_count += 1
            cls._fwd_total_ms += _ms
            if cls._fwd_call_count % 5 == 1:
                logger.info(
                    "[FastCodePredictor] #%d  total=%.1fms  avg=%.1fms",
                    cls._fwd_call_count, _ms,
                    cls._fwd_total_ms / cls._fwd_call_count,
                )
            return result

        # Fallback to legacy vLLM-attention path for bsz > 1
        return self._legacy_forward(
            layer0_code, layer0_embed, last_talker_hidden,
            do_sample, temperature, top_k, top_p,
        )

    @torch.inference_mode()
    def _legacy_forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Legacy vLLM-attention-based prediction path."""
        bsz = int(layer0_code.shape[0])
        num_groups = int(self.config.num_code_groups)
        max_steps = num_groups - 1

        self.reset_cache()

        prefill_input = torch.cat([last_talker_hidden, layer0_embed], dim=1)
        logits = self.prefill_logits(prefill_input)

        all_codes = [layer0_code.reshape(bsz, 1)]
        past_seq_len = 2

        for step in range(1, num_groups):
            if do_sample and temperature > 0:
                scaled = logits / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)
            all_codes.append(next_ids)

            if step < max_steps:
                logits = self.decode_logits(
                    next_ids.reshape(bsz),
                    generation_step=step,
                    past_seq_len=past_seq_len,
                )
                past_seq_len += 1

        return torch.cat(all_codes, dim=1)
