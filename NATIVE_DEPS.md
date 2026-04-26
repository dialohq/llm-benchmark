# Native / system dependencies

Everything in this repo is installed with `uv` into project-local venvs using
`uv` -managed Python (3.12). The list below is what the **host OS** must
provide before any `uv sync` runs — none of it is installable from PyPI.

Checked on: 2026-04-21 (CUDA 13.0 driver, NVIDIA-SMI 580.126.09).

## Common to all three engines (vLLM, SGLang, TensorRT-LLM)

- NVIDIA driver ≥ **580.x** (required by the CUDA-13 runtime the torch/
  trt-llm / sgl-kernel wheels are compiled against). CUDA-12.9 wheels will
  also run on this driver — the driver is backward-compatible.
- A CUDA toolkit is **not** required at the host level for running the
  prebuilt wheels; the Python packages bundle their own CUDA runtime. A
  system toolkit is only needed if you plan to compile kernels from source
  (e.g. flash-attn from source, DeepGEMM rebuild, custom ops).
- NCCL is pulled in transitively via `nvidia-nccl-cu12` / `nvidia-nccl-cu13`
  wheels; do **not** also have a system NCCL on `LD_LIBRARY_PATH`, or you
  will get symbol clashes at import. `unset LD_LIBRARY_PATH` if in doubt.
- `hf_transfer` is bundled in every venv for 10x faster model pulls. Set
  `HF_HUB_ENABLE_HF_TRANSFER=1` at launch.

## SGLang-specific

Install with `apt-get`:

```
libnuma-dev libnuma1 libopenmpi-dev libczmq4 libczmq-dev
```

- GDRCopy 2.5.1 — multi-node NCCL acceleration. Build from source:
  `git clone https://github.com/NVIDIA/gdrcopy && cd gdrcopy && make lib_install`.
  Skip if you are running single-node only.
- DeepEP / Mooncake 0.3.9 — only needed for prefill/decode disaggregation.

## TensorRT-LLM-specific

Install with `apt-get`:

```
libopenmpi-dev libzmq3-dev
```

- `mpi4py` has no usable wheel on Python 3.12, it is built against the
  system libopenmpi-dev headers at install time. If this fails, the whole
  `tensorrt-llm` install will fail.
- NGC PyTorch containers ship a conflicting cu128 torch; install TRT-LLM
  into a **fresh** `uv` venv only — never on top of the NGC container's
  Python. The cu130 torch pinned by TRT-LLM 1.2+ is not compatible with
  the NGC 25.03 image's torch stack.
- `libzmq3-dev` is only mandatory for disaggregated serving. Install it
  anyway — it is trivially small.

### nvidia-ml-py must match the *host driver* branch (`driver-r*` extras)

`tensorrt_llm._utils.confidential_compute_enabled()` calls
`nvmlSystemGetConfComputeSettings(c_nvmlSystemConfComputeSettings_v1_t())`
at engine init. That struct's layout changed between NVML versions, and
`nvidia-ml-py` ships per-driver-branch wheels named `13.<driver>.<patch>`.
If the pinned binding targets a newer driver branch than the host's
`libnvidia-ml.so`, the ctypes call reads past the struct → **silent
segfault during `Model init`** (crash is in C, not catchable by
try/except).

`trt-llm/pyproject.toml` exposes this as a second extra axis. Pick
exactly one GPU extra **and** exactly one driver extra:

```
uv sync --extra h100 --extra driver-r580     # this host (580.126.09)
uv sync --extra b200 --extra driver-r590     # etc.
```

| Driver reported by `nvidia-smi` | extra to use   | pin resolved              |
|---------------------------------|----------------|---------------------------|
| 580.x                           | `driver-r580`  | `nvidia-ml-py==13.580.126` |
| 590.x                           | `driver-r590`  | `nvidia-ml-py==13.590.48`  |
| 595.x                           | `driver-r595`  | `nvidia-ml-py==13.595.45`  |

The driver extras are mutually exclusive (`[tool.uv] conflicts`) so uv
will reject two at once. If your host runs r570/r575, downgrade
`tensorrt-llm` to 1.1.x first (it pins `nvidia-ml-py>=12,<13`) — this
pyproject targets 1.2.1 and intentionally excludes the <r580 branches.

Verify the NVML binding matches the driver before launching
`trtllm-serve`:

```
python -c "
import pynvml; pynvml.nvmlInit()
cc = pynvml.c_nvmlSystemConfComputeSettings_v1_t()
rc = pynvml.nvmlSystemGetConfComputeSettings(cc)
print('NVML ok, rc =', rc)
pynvml.nvmlShutdown()
"
```

A clean exit (rc=0 or rc=25 "not supported" on non-CC hardware) means
the ABI matches. A segfault means the driver extra is wrong.

## vLLM-specific

No extra system packages beyond the driver. Everything is wheel-delivered.

## Per-GPU-variant CUDA pairing

| Variant | SM    | Torch wheel channel | Notes                                              |
|---------|-------|---------------------|----------------------------------------------------|
| h100    | sm_90 | cu129               | Most stable wheel coverage                         |
| h200    | sm_90 | cu129               | Same wheels as H100; more HBM (141 GB)             |
| b200    | sm_100| cu130               | CUDA ≥ 12.8 required at runtime; cu130 preferred   |

All three engines use `uv`'s `[tool.uv.sources]` to route `torch` et al. to
the correct index based on the extra you activate (`--extra h100`, etc.).
No manual `--extra-index-url` is required at install time.

## Why separate venvs per engine

The three engines pin **different** `flashinfer-python` versions:

- vLLM 0.19.1 → `flashinfer-python==0.6.6`
- SGLang 0.5.10.post1 → `flashinfer-python==0.6.7.post3`
- TensorRT-LLM 1.2.1 → `flashinfer-python==0.6.4`

These are mutually exclusive; a shared venv would force-downgrade one
engine's cubin cache. Each engine gets its own project tree.
