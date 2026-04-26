# LLM inference engine benches — vLLM / SGLang / TensorRT-LLM

Three isolated `uv` projects, one per inference engine. Each project has three
GPU-variant extras (`h100`, `h200`, `b200`) and one venv per extra:

```
/workspace
├── NATIVE_DEPS.md         <-- system packages, drivers, GDRCopy, etc.
├── vllm/
│   ├── pyproject.toml
│   ├── .venv-h100/        <-- torch 2.10.0+cu129, vllm 0.19.1
│   ├── .venv-h200/        <-- same as h100
│   └── .venv-b200/        <-- torch 2.10.0+cu130, vllm 0.19.1
├── sglang/
│   ├── pyproject.toml
│   ├── .venv-h100/        <-- torch 2.9.1+cu129, sglang 0.5.10.post1
│   ├── .venv-h200/        <-- same as h100
│   └── .venv-b200/        <-- torch 2.9.1+cu130, sglang 0.5.10.post1
└── trt-llm/
    ├── pyproject.toml
    ├── .venv-h100/        <-- torch 2.9.1+cu130, tensorrt-llm 1.2.1
    ├── .venv-h200/        <-- same as h100
    └── .venv-b200/        <-- same as h100
```

All Python interpreters are **uv-managed CPython 3.12.13** (no system python).
Installed via `uv python install 3.12`.

## Re-creating a venv

```bash
# prerequisite: uv >= 0.11 and a uv-managed python 3.12
uv python install 3.12

cd <engine>/
rm -rf .venv-<variant>
uv venv .venv-<variant> --python /.uv/python_install/cpython-3.12-linux-x86_64-gnu/bin/python3.12
UV_PYTHON_PREFERENCE=only-managed \
UV_PROJECT_ENVIRONMENT=.venv-<variant> \
  uv sync --extra <variant> --index-strategy unsafe-best-match --prerelease allow
```

**TRT-LLM also needs a driver extra** (see `NATIVE_DEPS.md` §
"nvidia-ml-py must match the host driver branch"):

```bash
cd trt-llm/
UV_PYTHON_PREFERENCE=only-managed UV_PROJECT_ENVIRONMENT=.venv-h100 \
  uv sync --extra h100 --extra driver-r580 \
          --index-strategy unsafe-best-match --prerelease allow
```

Pick the `driver-r{NNN}` extra that matches
`nvidia-smi --query-gpu=driver_version --format=csv,noheader`. Mismatch
will segfault `trtllm-serve` during model init.

`--index-strategy unsafe-best-match` is **mandatory** — without it `uv` will
take torch from the default PyPI index (the CPU-only wheel) instead of the
cu129/cu130 index declared in `pyproject.toml`.

`--prerelease allow` is needed by sglang (uses `flash-attn-4>=4.0.0b4`) and
handy if you ever bump trt-llm to a 1.3.0rc.

## Pinned versions (as of 2026-04-21)

| Engine | Version | Torch | flashinfer | Other |
|--------|---------|-------|------------|-------|
| vLLM | 0.19.1 | 2.10.0 (cu129/cu130) | 0.6.6 (+cubin) | xgrammar 0.1.33, triton 3.6.0 |
| SGLang | 0.5.10.post1 | 2.9.1 (cu129/cu130) | 0.6.7.post3 (+cubin) | sglang-kernel 0.4.1, cuda-python 12.9, transformers 5.3.0 |
| TensorRT-LLM | 1.2.1 | 2.9.1 (cu130) | 0.6.4 | tensorrt 10.14.1, nvidia-nccl-cu13, transformers 4.57.3 |

The three engines pin **different** `flashinfer-python` versions, so they
cannot share a venv. Each engine always lives in its own venv tree.

## Activating a venv

```bash
source /workspace/vllm/.venv-h100/bin/activate
# or for sglang etc.
```

## Running

```bash
# vllm
source /workspace/vllm/.venv-h100/bin/activate
HF_HUB_ENABLE_HF_TRANSFER=1 vllm serve openai/gpt-oss-120b --tensor-parallel-size 8 --async-scheduling ...

# sglang
source /workspace/sglang/.venv-h100/bin/activate
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 8 --attention-backend fa3 ...

# trt-llm
source /workspace/trt-llm/.venv-h100/bin/activate
trtllm-serve openai/gpt-oss-120b --tp_size 8 ...
```

See `NATIVE_DEPS.md` for the `apt-get` packages that must be present at the
host level (libopenmpi-dev, libnuma-dev, libzmq3-dev, GDRCopy) before any of
the above will import cleanly.
