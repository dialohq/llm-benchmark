# gpt-oss run results on 1×H100 80GB (2026-04-21)

Single-H100 test rig (driver 580.126.09, CUDA 13.0). All three engines
booted and served `openai/gpt-oss-20b`; numbers below are **single-stream
decode tok/s** from `/v1/chat/completions` with `max_tokens=256`,
`temperature=0`, 83-token prompt.

| Engine | Version | Full recipe applied | Decode tok/s |
|--------|---------|---------------------|--------------|
| **vLLM** | 0.19.1 | FA3 + async-scheduling + torch.compile + CUDA graphs + EAGLE3 spec decode + reasoning/tool-call parsers + stream-interval 1 | **368.4** |
| **SGLang** | 0.5.10.post1 | FA3 + chunked prefill + piecewise CUDA graphs | **316.8** |
| **TRT-LLM** | 1.2.1 | PyTorch backend (same OpenAI Triton MoE kernel as vLLM/SGLang on Hopper) | **179.3** |

vLLM EAGLE3 acceptance at `num_speculative_tokens=3`:
- 282 accepted / 247 drafts ⇒ **1.14 extra tokens per draft**
- per-position acceptance: pos0=61 %, pos1=34 %, pos2=19 %

## What worked, what didn't

### vLLM — full recipe went in clean
`RedHatAI/gpt-oss-20b-speculator.eagle3` is the draft that actually works
with vLLM 0.19.1. The nvidia drafts (`nvidia/gpt-oss-120b-Eagle3-v2`) are
**TRT-LLM-only** — I hit that trap first and it parsed but drew no
speedup. The lmsys drafts (`lmsys/EAGLE3-gpt-oss-*`) are gated.

### 120b on one H100: doesn't fit with EAGLE3
MXFP4 weights are ~63 GB, the RedHatAI EAGLE3 draft loads as BF16 so the
main + draft footprint climbs to ~78 GB before activations/KV/CUDA graphs
can claim any room. OOM at "profile run" stage every time. **EAGLE3 on
120b needs TP≥2** on Hopper; demoted to gpt-oss-20b for the single-GPU
comparison.

### SGLang — EAGLE3 doesn't work for 20b
Three public drafts on HF (`nebius/EAGLE3-gpt-oss-20b`,
`zhuyksir/EAGLE3-gpt-oss-20b-bf16`, `lukeysong/gpt-oss-20b-moe-eagle3`),
all of them **truncated-vocab** (32 k or 64 k out of gpt-oss's 201 k).
SGLang's `vocab_parallel_embedding` asserts vocab-size parity and refuses
to load them. The `lmsys/EAGLE3-gpt-oss-20b-bf16` weights that would
match are **gated** (HF 401 without a token). Without a compatible draft
I ran the SGLang base recipe only.

### SGLang — `--enable-torch-compile` crashed
With `SGLANG_ENABLE_SPEC_V2=1` + `--enable-torch-compile`, CUDA graph
capture failed with
`AttributeError: type object 'weakref.ProxyType' has no attribute
'__torch_dispatch__'` (torch dynamo bug on this combo). SGLang's own
error message told me to drop `--enable-torch-compile` — did that and
startup completed.

### TRT-LLM — the NVML segfault and the fix
First launch crashed mid-init in `nvmlSystemGetConfComputeSettings`:

```
Model init total -- 12.69s
!!!!!!! Segfault encountered !!!!!!!
  File "<unknown>", line 0, in nvmlSystemGetConfComputeSettings
  File "<unknown>", line 0, in ffi_call
```

**Root cause.** `tensorrt-llm` 1.2.1 pins `nvidia-ml-py>=13`. Without a
pin pip/uv resolves to the newest wheel on PyPI (at the time of this run,
`13.595.45` — the r595 driver branch). `nvidia-ml-py` wheels are
versioned `13.<driver>.<patch>` and each one ships a Python `ctypes`
struct whose layout matches the NVML library for that driver branch. The
host here runs driver **r580.126.09**, so the r595 Python struct is
larger than the `libnvidia-ml.so` output buffer. When TRT-LLM calls
`nvmlSystemGetConfComputeSettings(c_nvmlSystemConfComputeSettings_v1_t())`,
NVML writes into the ctypes buffer using the r580 layout, the excess
bytes land in unmapped memory, and the worker segfaults. The crash is
inside the C call — TRT-LLM's own `try/except` wrapping around it is
useless.

**Fix.** Pin `nvidia-ml-py` to the wheel matching the host driver branch:

```
nvidia-ml-py==13.580.126   # driver r580.x
```

This repo exposes it as a **second extra axis** in
`trt-llm/pyproject.toml`. You pick one GPU extra *and* one driver extra:

```
uv sync --extra h100 --extra driver-r580
uv sync --extra b200 --extra driver-r590
```

Available driver extras: `driver-r580`, `driver-r590`, `driver-r595`.
Extras in the same group are declared mutually exclusive via
`[tool.uv] conflicts`.

On this host the fix produced a clean engine init in ~7:14 (import +
MPI spawn + weight load + autotuner warmup) and `trtllm-serve` served
`/v1/chat/completions` normally.

**Single-stream number.** **179.3 tok/s** on gpt-oss-20b, about half of
vLLM. This matches the description: on Hopper the TRTLLM-Gen MoE path
(the fast one) isn't available, so trt-llm falls back to the OpenAI
Triton kernel via its PyTorch backend and pays an extra IPC/MPI hop
that vLLM and SGLang don't have. The TRTLLM-Gen advantage only kicks
in on Blackwell (sm_100+).

### Other TRT-LLM notes
- Cold start is **~7 min** on this host: import is ~4 min, the MPI
  worker imports TRT-LLM a second time (~1 min more), autotuner warmup
  is another ~1.5 min. Stays fast afterwards because PDL
  (Programmatic Dependent Launch) kicks in.
- `trtllm-serve --backend pytorch` is the right flag — no TRT engine-
  build step, same OpenAI Triton MoE kernel as vLLM/SGLang on Hopper.
- `--host 127.0.0.1 --port 9000 --max_batch_size 4 --max_seq_len 4096`
  for a 1×H100 batch-1 smoke test.

## Notes I was told about in advance that mattered

- `--attention-backend fa3` is mandatory on Hopper for gpt-oss attention
  sinks. FlashInfer + XFormers both refuse to load. vLLM auto-selects
  `FLASH_ATTN` backend correctly.
- `VLLM_USE_V1=1` is ignored by 0.19.1 (V1 engine is already default,
  warns "Unknown vLLM environment variable"). Harmless.
- `--reasoning-parser openai_gptoss` and `--tool-call-parser openai` are
  free and should be on by default for gpt-oss.
- `--stream-interval 1` matters when you care about true per-token
  streaming; default value of 20 batches 20 tokens per SSE message and
  looks laggy to clients.
- SGLang 0.5.10 *automatically* disables piecewise CUDA graph when
  `SGLANG_ENABLE_SPEC_V2=1` is on. Without spec v2 the piecewise path
  captured 50 bucket sizes in ~38 s.
