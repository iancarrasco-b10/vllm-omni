# Qwen3-TTS Code Predictor Performance Optimizations

## Problem

The Qwen3-TTS Talker (Stage-0) was generating codec tokens at only **~10 tok/s**, resulting in a warm TTFA (Time To First Audio) of **~1,091 ms**. For a 1.7B parameter AR model that should be capable of 100+ tok/s, this was unacceptably slow.

## Root Cause Analysis

Profiling revealed the per-step execution breakdown for each Talker decode step:

| Component | Time | Notes |
|---|---|---|
| Main Talker model forward (1.7B) | **2.1 ms** | CUDA graphs active, fast |
| **Code predictor (residual codebooks)** | **86 ms** | 15-step AR loop, no graphs |
| hidden\_states → CPU copy | 0.0 ms | Negligible |
| **Total per step** | **~88 ms** | → ~11 tok/s |

The code predictor consumes **97% of per-step time**. It predicts residual codebooks 1..Q-1 (15 sequential AR decode steps for `num_code_groups=16`) using a small 5-layer, 1024-hidden transformer.

Further profiling inside the code predictor showed:

| Component | Time | Per Step |
|---|---|---|
| Decode (14 steps) | 76 ms | 5.4 ms/step |
| Prefill (2 tokens) | 5.6 ms | — |
| Sampling (`torch.multinomial`) | 3.6 ms | 0.2 ms/step |

Each 5.4 ms decode step ran through **vLLM's full paged-attention infrastructure**: `build_attn_metadata()` with Python loops, `.item()` calls, CPU tensor allocations, `set_forward_context` context managers, and `set_current_vllm_config` — massive overhead for a model that processes 1 token through 5 layers with max sequence length 17.

## Solution

Two-stage optimization applied to `qwen3_tts_code_predictor_vllm.py`:

### Stage 1: SDPA Fast Path (86 ms → 52 ms)

Replaced vLLM's paged-attention infrastructure with direct PyTorch operations:

- **`F.scaled_dot_product_attention`** with `enable_gqa=True` for native GQA support (8 KV heads, 16 Q heads) — eliminates expensive `repeat_interleave` tensor expansion
- **Pre-allocated tensor KV cache** `[num_layers, 1, max_seq, kv_heads, head_dim]` — no paged memory, no block tables, no slot mapping
- **Pre-computed RoPE cos/sin tables** via `_build_rope_cache()` — single table lookup instead of recomputation
- **Direct `F.linear()` calls** on vLLM's fused weight tensors (`qkv_proj`, `gate_up_proj`) — bypasses vLLM's custom linear layers and their overhead
- **Inline `_rms_norm()`** — simple RMS normalization without vLLM's `RMSNorm` module overhead

The fast path reuses the same model weights loaded by vLLM but executes them through a lightweight forward path (`_fast_layer_forward` → `_fast_model_forward`).

### Stage 2: CUDA Graph Capture (52 ms → 6.5 ms)

Captured one CUDA graph per decode step (14 graphs total for `num_code_groups=16`):

- Each graph bakes in the fixed `seq_len`, position index, and `lm_head[step]` for that step
- **Embedding lookup + sampling run outside the graph** (data-dependent operations)
- **Model forward + logits computation run inside the graph** (fixed-shape operations)
- KV cache tensor memory locations are shared across all graphs — writes/reads at fixed offsets are captured correctly
- Graphs are captured lazily on first inference call via `_capture_decode_graphs()`
- Warmup runs 3 eager iterations per step before capture to stabilize CUDA caches

## Results

| Metric | Before | After | Improvement |
|---|---|---|---|
| Code predictor per step | 86 ms | 6.5 ms | **13x faster** |
| Stage-0 TPS | 10 tok/s | 48 tok/s | **4.8x faster** |
| TTFA (warm) | 1,091 ms | 287 ms | **3.8x faster** |
| Total generation time (short text) | 3,524 ms | 614 ms | **5.7x faster** |

TPS is consistent at ~48 tok/s across different input lengths (tested up to 83 output tokens).

## Files Modified

### `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py`

Primary optimization target. Added:

- **Helper functions** at module level:
  - `_build_rope_cache()` — pre-computes RoPE cos/sin tables
  - `_apply_rope()` — applies rotary embeddings to Q/K tensors
  - `_rms_norm()` — lightweight RMS normalization

- **Fast path methods** on `Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM`:
  - `_init_fast_path()` — one-time setup: RoPE tables, KV cache tensors, CUDA graph pool
  - `_fast_layer_forward()` — single transformer layer via SDPA (replaces vLLM `Qwen3DecoderLayer`)
  - `_fast_model_forward()` — all layers + final norm
  - `_capture_decode_graphs()` — captures one CUDA graph per decode step
  - `fast_forward()` — main entry point: prefill (eager) + decode (graph replay) + sampling (eager)

- **Dispatch in `forward()`**: routes `bsz=1` to `fast_forward()`, falls back to `_legacy_forward()` for batched inputs

### `vllm_omni/worker/gpu_model_runner.py`

- Added timing instrumentation to `_talker_mtp_forward()` for monitoring code predictor latency per step

### `vllm_omni/worker/gpu_ar_model_runner.py`

- Added timing instrumentation for the main model forward and hidden\_states CPU copy to isolate bottleneck contributions

### `vllm_omni/entrypoints/omni_stage.py`

- Added per-request TPS logging for both synchronous and asynchronous (`async_chunk`) execution paths in Stage-0

## Architecture

```
Per Talker Decode Step:
┌──────────────────────────────────────────────────────┐
│ Main Talker Forward (1.7B, CUDA graph)     ~2.1 ms   │
├──────────────────────────────────────────────────────┤
│ Code Predictor (fast path)                 ~6.5 ms   │
│  ├─ Prefill [hidden, layer0_embed] (eager) ~1.0 ms   │
│  ├─ Decode steps 1..14 (CUDA graph replay) ~4.5 ms   │
│  │   └─ Per step: static_in.copy_ → graph.replay()  │
│  └─ Sampling (torch.multinomial, eager)    ~1.0 ms   │
├──────────────────────────────────────────────────────┤
│ Total per step                             ~8.6 ms   │
│ → Effective throughput                    ~48 tok/s   │
└──────────────────────────────────────────────────────┘
```

## Remaining Opportunities

- **~15 ms MTP wrapper overhead** between the raw 6.5 ms code predictor and the 15.8 ms observed at `_talker_mtp_forward` — includes embedding lookup, `small_to_mtp_projection`, sampling, Python loop, and CPU copy of `code_predictor_codes`
- ~~**Batch code predictor** for `bsz > 1`~~ — **Done**: CUDA graphs now captured for all `(step, bsz)` pairs up to `max_num_seqs`
- **Overlap code predictor with main Talker forward** using separate CUDA streams
- **Graph-safe sampling** to capture the entire AR loop (embed + forward + sample) as a single graph
