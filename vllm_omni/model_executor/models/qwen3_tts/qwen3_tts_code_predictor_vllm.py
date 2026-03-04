from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.transformers_utils.config import set_default_rope_theta

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


KVCache = tuple[torch.Tensor, torch.Tensor]


class CodePredictorAttention(nn.Module):
    """Standalone attention using SDPA + dense KV buffers.

    Reuses QKVParallelLinear, RowParallelLinear, RMSNorm (QK-norm), and RoPE
    from vLLM but replaces the paged-attention backend with
    ``F.scaled_dot_product_attention``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, qlen] position ids.
            hidden_states: [B, qlen, hidden_size].
            kv_cache: (k_cache, v_cache) each [B, num_kv_heads, max_seq_len, head_dim].
            seq_len: total sequence length *after* this forward (past + current query).

        Returns:
            output: [B, qlen, hidden_size].
        """
        bsz, qlen, _ = hidden_states.shape
        k_cache, v_cache = kv_cache

        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * qlen, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(positions.reshape(-1), q, k)

        q = q.view(bsz, qlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, qlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, qlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        start_pos = seq_len - qlen
        k_cache[:bsz, :, start_pos:seq_len, :] = k
        v_cache[:bsz, :, start_pos:seq_len, :] = v

        k_full = k_cache[:bsz, :, :seq_len, :]
        v_full = v_cache[:bsz, :, :seq_len, :]

        attn_out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            scale=self.scaling,
            is_causal=(qlen == seq_len),
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz * qlen, -1)
        output, _ = self.o_proj(attn_out)
        return output.view(bsz, qlen, -1)


class CodePredictorDecoderLayer(nn.Module):
    """Standalone decoder layer for the code predictor.

    Same architecture as ``Qwen3DecoderLayer`` (attention + MLP with
    pre-norm residuals) but uses ``CodePredictorAttention`` instead of
    vLLM's ``Attention`` backend. Weight names are identical so existing
    checkpoints load without changes.
    """

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)

        self.self_attn = CodePredictorAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_cache: KVCache,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            seq_len=seq_len,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3TTSTalkerCodePredictorModelVLLM(nn.Module):
    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        talker_hidden_size: int | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
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

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_caches: list[KVCache],
        seq_len: int,
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, qlen] position ids.
            inputs_embeds: [B, qlen, hidden_size].
            kv_caches: list of (k_cache, v_cache) per layer.
            seq_len: total sequence length after this forward.
        """
        hidden_states = inputs_embeds
        residual = None
        for layer, kv_cache in zip(self.layers, kv_caches):
            hidden_states, residual = layer(positions, hidden_states, residual, kv_cache, seq_len)
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

        # Dense KV cache state (allocated lazily).
        self._kv_caches: list[KVCache] | None = None
        self._max_seq_len = int(getattr(config, "num_code_groups", 16) or 16)
        self._num_layers = int(config.num_hidden_layers)
        self._num_kv_heads = int(config.num_key_value_heads)
        self._head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
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

        max_fast_batch = int(getattr(self._vllm_config.scheduler_config, "max_num_seqs", 1) or 1)
        max_fast_batch = max(1, min(max_fast_batch, 32))

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
        self._fast_max_batch = max_fast_batch

        self._fast_rope_cos, self._fast_rope_sin = _build_rope_cache(
            head_dim, max_seq, rope_theta, device, torch.bfloat16
        )

        # KV cache: [num_layers, max_batch, max_seq, kv_heads, head_dim]
        self._fast_k_cache = torch.zeros(
            num_layers, max_fast_batch, max_seq, num_kv_heads, head_dim,
            dtype=torch.bfloat16, device=device,
        )
        self._fast_v_cache = torch.zeros(
            num_layers, max_fast_batch, max_seq, num_kv_heads, head_dim,
            dtype=torch.bfloat16, device=device,
        )

        # Pre-compute causal mask for SDPA (avoids recomputation per call)
        self._fast_causal_mask = torch.tril(
            torch.ones(max_seq, max_seq, dtype=torch.bool, device=device)
        )

        # CUDA graph infrastructure keyed by (decode_step, batch_size)
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._decode_graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self._graph_warmup_done = False

        # Whole-loop graph: captures entire AR loop as single CUDA graph
        self._whole_loop_graphs: dict[int, dict] = {}
        self._whole_loop_captured = False
        self._whole_loop_top_k = 50

        # Pre-allocate per-step position tensors for whole-loop graph
        self._step_positions = torch.arange(max_seq, dtype=torch.long, device=device)

        self._fast_ready = True
        logger.info(
            "[FastCodePredictor] Initialized: layers=%d q_heads=%d kv_heads=%d "
            "head_dim=%d hidden=%d max_seq=%d max_batch=%d",
            num_layers, num_q_heads, num_kv_heads, head_dim, cfg.hidden_size, max_seq, max_fast_batch,
        )

    @torch.inference_mode()
    def _fast_layer_forward(
        self, layer_idx: int, hidden: torch.Tensor, positions: torch.Tensor,
        seq_len: int, tok_count: int, bsz: int,
    ) -> torch.Tensor:
        """Run one transformer layer using SDPA. hidden: [bsz, tok_count, H]."""
        layer = self.model.layers[layer_idx]
        eps = self._fast_eps
        num_q = self._fast_num_q_heads
        num_kv = self._fast_num_kv_heads
        hd = self._fast_head_dim
        q_size = self._fast_q_size
        kv_size = self._fast_kv_size
        inter = self._fast_intermediate

        residual = hidden
        hidden = _rms_norm(hidden, layer.input_layernorm.weight, eps)

        qkv = F.linear(hidden, layer.self_attn.qkv_proj.weight)
        q = qkv[..., :q_size].reshape(bsz, tok_count, num_q, hd).transpose(1, 2)
        k = qkv[..., q_size:q_size+kv_size].reshape(bsz, tok_count, num_kv, hd).transpose(1, 2)
        v = qkv[..., q_size+kv_size:].reshape(bsz, tok_count, num_kv, hd).transpose(1, 2)

        q = _rms_norm(q, layer.self_attn.q_norm.weight, eps)
        k = _rms_norm(k, layer.self_attn.k_norm.weight, eps)

        q = _apply_rope(q, self._fast_rope_cos, self._fast_rope_sin, positions)
        k = _apply_rope(k, self._fast_rope_cos, self._fast_rope_sin, positions)

        start = seq_len - tok_count
        self._fast_k_cache[layer_idx, :bsz, start:seq_len] = k.transpose(1, 2)
        self._fast_v_cache[layer_idx, :bsz, start:seq_len] = v.transpose(1, 2)

        k_full = self._fast_k_cache[layer_idx, :bsz, :seq_len].transpose(1, 2)
        v_full = self._fast_v_cache[layer_idx, :bsz, :seq_len].transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=(tok_count > 1), enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, tok_count, q_size)

        attn_out = F.linear(attn_out, layer.self_attn.o_proj.weight)
        hidden = residual + attn_out

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
    def _fast_model_forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor, seq_len: int, tok_count: int, bsz: int) -> torch.Tensor:
        """Run all layers + final norm. inputs_embeds: [bsz, tok_count, H]."""
        hidden = inputs_embeds
        for i in range(self._fast_num_layers):
            hidden = self._fast_layer_forward(i, hidden, positions, seq_len, tok_count, bsz)
        hidden = _rms_norm(hidden, self.model.norm.weight, self._fast_eps)
        return hidden

    def _capture_decode_graphs(self, device: torch.device) -> None:
        """Capture CUDA graphs for all (decode_step, batch_size) combinations.

        Keys are (step, bsz) tuples. Each graph captures the model forward
        + lm_head projection for one decode step at a specific batch size.
        """
        if self._graph_warmup_done:
            return
        num_groups = int(self.config.num_code_groups)
        h_pred = self._fast_hidden
        max_fast_batch = self._fast_max_batch
        num_decode_steps = num_groups - 2
        total_graphs = num_decode_steps * max_fast_batch

        logger.info(
            "[FastCodePredictor] Capturing %d CUDA graphs (%d steps × %d batch sizes)...",
            total_graphs, num_decode_steps, max_fast_batch,
        )

        # Warmup: run each batch size through all steps eagerly
        for bsz in range(1, max_fast_batch + 1):
            for _ in range(3):
                self._fast_k_cache[:, :bsz].zero_()
                self._fast_v_cache[:, :bsz].zero_()
                dummy_pf = torch.zeros(bsz, 2, h_pred, dtype=torch.bfloat16, device=device)
                self._fast_model_forward(dummy_pf, self._prefill_pos, seq_len=2, tok_count=2, bsz=bsz)
                for step in range(1, num_groups - 1):
                    dummy_in = torch.zeros(bsz, 1, h_pred, dtype=torch.bfloat16, device=device)
                    self._pos_buf[0] = 1 + step
                    self._fast_model_forward(dummy_in, self._pos_buf, seq_len=2 + step, tok_count=1, bsz=bsz)

        # Capture one graph per (step, bsz) pair
        for bsz in range(1, max_fast_batch + 1):
            for step in range(1, num_groups - 1):
                self._fast_k_cache[:, :bsz].zero_()
                self._fast_v_cache[:, :bsz].zero_()
                dummy_pf = torch.zeros(bsz, 2, h_pred, dtype=torch.bfloat16, device=device)
                self._fast_model_forward(dummy_pf, self._prefill_pos, seq_len=2, tok_count=2, bsz=bsz)
                for prev in range(1, step):
                    dummy_in = torch.zeros(bsz, 1, h_pred, dtype=torch.bfloat16, device=device)
                    self._pos_buf[0] = 1 + prev
                    self._fast_model_forward(dummy_in, self._pos_buf, seq_len=2 + prev, tok_count=1, bsz=bsz)

                seq_len_for_step = 2 + step
                static_in = torch.zeros(bsz, 1, h_pred, dtype=torch.bfloat16, device=device)
                self._pos_buf[0] = seq_len_for_step - 1

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._graph_pool):
                    hidden = self._fast_model_forward(static_in, self._pos_buf, seq_len=seq_len_for_step, tok_count=1, bsz=bsz)
                    static_out = self.lm_head[step](hidden[:, -1:, :]).squeeze(1)

                self._decode_graphs[(step, bsz)] = (g, static_in, static_out)

        self._graph_warmup_done = True
        logger.info("[FastCodePredictor] Captured %d CUDA graphs", len(self._decode_graphs))

    def _whole_loop_body(
        self,
        prefill_input: torch.Tensor,
        layer0_code: torch.Tensor,
        gumbel_noise: torch.Tensor,
        inv_temp: torch.Tensor,
        bsz: int,
        top_k: int,
    ) -> torch.Tensor:
        """Execute the full AR loop — ONLY for graph warmup / capture.

        NOT a general-purpose forward; zeros the KV cache at the start so each
        call is self-contained (required for CUDA graph capture).  The runtime
        eager fallback uses the incremental per-step path in fast_forward().

        All operations are CUDA-graph-safe: no data-dependent control flow,
        no torch.multinomial.  Sampling uses the Gumbel-max trick.
        """
        device = prefill_input.device
        num_groups = int(self.config.num_code_groups)
        embeddings = self.model.get_input_embeddings()
        projection = self.small_to_mtp_projection
        positions = self._step_positions

        self._fast_k_cache[:, :bsz].zero_()
        self._fast_v_cache[:, :bsz].zero_()

        hidden = self._fast_model_forward(
            prefill_input, positions[:2], seq_len=2, tok_count=2, bsz=bsz,
        )
        logits = self.lm_head[0](hidden[:, -1:, :]).squeeze(1)

        codes_buf = torch.zeros(bsz, num_groups, dtype=torch.long, device=device)
        codes_buf[:, 0] = layer0_code

        for step in range(1, num_groups):
            scaled = logits.float() * inv_temp
            topk_vals, _ = scaled.topk(top_k, dim=-1)
            scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
            next_ids = (scaled + gumbel_noise[step - 1]).argmax(dim=-1, keepdim=True)
            codes_buf[:, step] = next_ids.reshape(bsz)

            if step < num_groups - 1:
                tok_embed = embeddings[step - 1](next_ids.long())
                tok_embed = projection(tok_embed.to(torch.bfloat16))
                pos = positions[step + 1 : step + 2]
                hidden = self._fast_model_forward(
                    tok_embed, pos, seq_len=step + 2, tok_count=1, bsz=bsz,
                )
                logits = self.lm_head[step](hidden[:, -1:, :]).squeeze(1)

        return codes_buf

    def _capture_whole_loop_graphs(self, device: torch.device) -> None:
        """Capture one CUDA graph per batch size containing the full AR loop.

        On failure (OOM, unsupported op, etc.), logs a warning and leaves
        ``_whole_loop_captured`` False so fast_forward() falls through to the
        incremental per-step path without any regression.
        """
        if self._whole_loop_captured:
            return

        num_groups = int(self.config.num_code_groups)
        h_pred = self._fast_hidden
        vocab = int(self.config.vocab_size)
        max_bsz = self._fast_max_batch
        top_k = self._whole_loop_top_k

        logger.info(
            "[WholeLoopGraph] Capturing %d graphs (bsz 1..%d, "
            "%d AR steps, top_k=%d, vocab=%d)...",
            max_bsz, max_bsz, num_groups - 1, top_k, vocab,
        )
        t0 = __import__("time").perf_counter()

        try:
            for bsz in range(1, max_bsz + 1):
                static_prefill = torch.zeros(
                    bsz, 2, h_pred, dtype=torch.bfloat16, device=device,
                )
                static_layer0_code = torch.zeros(bsz, dtype=torch.long, device=device)
                static_gumbel = torch.zeros(
                    num_groups - 1, bsz, vocab, dtype=torch.float32, device=device,
                )
                static_inv_temp = torch.ones(1, dtype=torch.float32, device=device)

                for _ in range(3):
                    self._whole_loop_body(
                        static_prefill, static_layer0_code,
                        static_gumbel, static_inv_temp, bsz, top_k,
                    )

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._graph_pool):
                    static_codes = self._whole_loop_body(
                        static_prefill, static_layer0_code,
                        static_gumbel, static_inv_temp, bsz, top_k,
                    )

                self._whole_loop_graphs[bsz] = {
                    "graph": g,
                    "prefill": static_prefill,
                    "layer0_code": static_layer0_code,
                    "gumbel": static_gumbel,
                    "inv_temp": static_inv_temp,
                    "codes": static_codes,
                }
        except Exception as e:
            logger.warning(
                "[WholeLoopGraph] Capture failed at bsz=%d: %s  "
                "— falling back to incremental per-step path",
                bsz, e,
            )
            self._whole_loop_graphs.clear()
            self._whole_loop_captured = True  # don't retry
            return

        elapsed = (__import__("time").perf_counter() - t0) * 1000
        self._whole_loop_captured = True
        logger.info(
            "[WholeLoopGraph] Captured %d graphs in %.0f ms", max_bsz, elapsed,
        )

    @torch.inference_mode()
    def _whole_loop_forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor | None:
        """Try to run via whole-loop CUDA graph. Returns None if not available."""
        bsz = int(layer0_code.shape[0])
        if not self._whole_loop_captured or bsz not in self._whole_loop_graphs:
            return None
        if top_k != self._whole_loop_top_k:
            return None

        entry = self._whole_loop_graphs[bsz]
        device = layer0_code.device

        prefill_input = torch.cat(
            [last_talker_hidden, layer0_embed], dim=1,
        ).to(torch.bfloat16)
        prefill_input = self.small_to_mtp_projection(prefill_input)

        entry["prefill"].copy_(prefill_input)
        entry["layer0_code"].copy_(layer0_code.reshape(bsz))

        if do_sample and temperature > 0:
            inv_temp = 1.0 / temperature
            uniform = torch.rand_like(entry["gumbel"])
            uniform.clamp_(1e-10, 1.0 - 1e-7)
            entry["gumbel"].copy_(-torch.log(-torch.log(uniform)))
            entry["inv_temp"].fill_(inv_temp)
        else:
            entry["gumbel"].zero_()
            entry["inv_temp"].fill_(1.0)

        entry["graph"].replay()
        return entry["codes"].clone()

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
        """Fast SDPA-based prediction of residual codebooks 1..Q-1.

        Tries (in order):
        1. Whole-loop CUDA graph  — single replay for the entire AR loop
        2. Incremental per-step   — one decode step at a time with growing
           KV cache; uses per-step CUDA graphs when available, otherwise
           eager PyTorch.  This is the same path as before the whole-loop
           optimisation, so there is no fallback regression.
        """
        device = layer0_code.device
        bsz = int(layer0_code.shape[0])
        self._init_fast_path(device)

        # --- path 1: whole-loop graph (single replay) ---
        self._capture_whole_loop_graphs(device)
        result = self._whole_loop_forward(
            layer0_code, layer0_embed, last_talker_hidden,
            do_sample, temperature, top_k,
        )
        if result is not None:
            return result

        # --- path 2: incremental per-step decode (unchanged from pre-PR) ---
        # Each step writes one KV entry and reads the full accumulated cache,
        # exactly as before. Per-step CUDA graphs accelerate individual
        # model_forward calls; pure eager is the final fallback.
        return self._incremental_per_step_forward(
            layer0_code, layer0_embed, last_talker_hidden,
            do_sample, temperature, top_k, top_p,
        )

    @torch.inference_mode()
    def _incremental_per_step_forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Incremental KV-cached decode, one step at a time.

        Identical to the pre-whole-loop-graph code path: prefill populates
        2 KV entries, then each decode step appends 1 entry and attends over
        the full history.  Per-step CUDA graphs are used when captured.
        """
        device = layer0_code.device
        bsz = int(layer0_code.shape[0])
        num_groups = int(self.config.num_code_groups)
        max_steps = num_groups - 1

        if not hasattr(self, '_pos_buf'):
            self._pos_buf = torch.zeros(1, dtype=torch.long, device=device)
            self._prefill_pos = torch.arange(2, dtype=torch.long, device=device)

        self._capture_decode_graphs(device)
        use_graphs = self._graph_warmup_done and bsz <= self._fast_max_batch

        self._fast_k_cache[:, :bsz].zero_()
        self._fast_v_cache[:, :bsz].zero_()

        prefill_input = torch.cat([last_talker_hidden, layer0_embed], dim=1).to(torch.bfloat16)
        prefill_input = self.small_to_mtp_projection(prefill_input)
        hidden = self._fast_model_forward(prefill_input, self._prefill_pos, seq_len=2, tok_count=2, bsz=bsz)
        logits = self.lm_head[0](hidden[:, -1:, :]).squeeze(1)

        codes_buf = torch.zeros(bsz, num_groups, dtype=torch.long, device=device)
        codes_buf[:, 0] = layer0_code.reshape(bsz)
        seq_len = 2
        embeddings = self.model.get_input_embeddings()
        lm_heads = self.lm_head
        projection = self.small_to_mtp_projection
        inv_temp = 1.0 / temperature if do_sample and temperature > 0 else 0.0

        for step in range(1, num_groups):
            if inv_temp > 0:
                scaled = logits.float() * inv_temp
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                probs = probs.clamp(min=0.0)
                row_sums = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(row_sums > 0, probs / row_sums, torch.ones_like(probs) / probs.shape[-1])
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)
            codes_buf[:, step] = next_ids.reshape(bsz)

            if step < max_steps:
                tok_embed = embeddings[step - 1](next_ids.long())
                tok_embed = projection(tok_embed.to(torch.bfloat16))

                graph_key = (step, bsz)
                if use_graphs and graph_key in self._decode_graphs:
                    g, static_in, static_out = self._decode_graphs[graph_key]
                    static_in.copy_(tok_embed)
                    self._pos_buf[0] = seq_len
                    g.replay()
                    logits = static_out
                else:
                    self._pos_buf[0] = seq_len
                    hidden = self._fast_model_forward(tok_embed, self._pos_buf, seq_len=seq_len + 1, tok_count=1, bsz=bsz)
                    logits = lm_heads[step](hidden[:, -1:, :]).squeeze(1)
                seq_len += 1

        return codes_buf

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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

    def _allocate_kv_caches(self, batch_size: int, device: torch.device) -> list[KVCache]:
        """Allocate dense KV cache tensors for all layers."""
        caches: list[KVCache] = []
        for _ in range(self._num_layers):
            k = torch.zeros(
                batch_size,
                self._num_kv_heads,
                self._max_seq_len,
                self._head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            v = torch.zeros(
                batch_size,
                self._num_kv_heads,
                self._max_seq_len,
                self._head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            caches.append((k, v))
        return caches

    @torch.inference_mode()
    def reset_cache(self) -> None:
        if self._kv_caches is not None:
            for k, v in self._kv_caches:
                k.zero_()
                v.zero_()

    @torch.inference_mode()
    def prefill_logits(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Prefill with 2 tokens: [past_hidden, layer0_embed]. Returns logits for residual group 0."""
        bsz = int(inputs_embeds.shape[0])
        qlen = 2
        device = inputs_embeds.device

        if self._kv_caches is None or self._kv_caches[0][0].shape[0] < bsz:
            self._kv_caches = self._allocate_kv_caches(bsz, device)

        hs = inputs_embeds.to(dtype=torch.bfloat16)  # [B, 2, H]
        hs = self.small_to_mtp_projection(hs.reshape(bsz * qlen, -1)).view(bsz, qlen, -1)

        positions = torch.arange(qlen, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

        out = self.model(positions=positions, inputs_embeds=hs, kv_caches=self._kv_caches, seq_len=qlen)

        last_h = out[:, -1, :]  # [B, hidden]
        logits = self.lm_head[0](last_h)
        return logits

    @torch.inference_mode()
    def decode_logits(self, input_ids: torch.Tensor, *, generation_step: int, past_seq_len: int) -> torch.Tensor:
        """Decode one new token for residual group `generation_step` (1..Q-1)."""
        assert self._kv_caches is not None
        bsz = int(input_ids.shape[0])
        if generation_step <= 0:
            raise ValueError("generation_step must be >= 1 for decode_logits")

        embed_idx = generation_step - 1
        hs = self.model.get_input_embeddings()[embed_idx](input_ids.to(dtype=torch.long).reshape(bsz, 1))
        hs = self.small_to_mtp_projection(hs.reshape(bsz, -1)).view(bsz, 1, -1)

        seq_len = past_seq_len + 1
        positions = torch.full((bsz, 1), past_seq_len, dtype=torch.long, device=input_ids.device)

        out = self.model(positions=positions, inputs_embeds=hs, kv_caches=self._kv_caches, seq_len=seq_len)

        logits = self.lm_head[generation_step](out[:, 0, :])
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
        """Dispatches to fast SDPA path (any bsz) or legacy fallback."""
        import time as _time

        bsz = int(layer0_code.shape[0])
        self._init_fast_path(layer0_code.device)

        if bsz > self._fast_max_batch:
            return self._legacy_forward(
                layer0_code, layer0_embed, last_talker_hidden,
                do_sample, temperature, top_k, top_p,
            )

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
                "[FastCodePredictor] #%d  bsz=%d  total=%.1fms  avg=%.1fms",
                cls._fwd_call_count, bsz, _ms,
                cls._fwd_total_ms / cls._fwd_call_count,
            )
        return result

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
                scaled = logits.float() / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                probs = probs.clamp(min=0.0)
                row_sums = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(row_sums > 0, probs / row_sums, torch.ones_like(probs) / probs.shape[-1])
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
