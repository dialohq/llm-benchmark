{
  description = "LLM serving benchmark — devshells for vLLM, SGLang, TensorRT-LLM";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  # Pull prebuilt unfree cuda* derivations (cuda_nvcc, cuda_cudart,
  # cuda_cccl) from the cuda-maintainers binary cache instead of building
  # them locally on first eval. With these substituters the cold path of
  # `nix develop .#<engine>` drops from minutes to seconds.
  nixConfig = {
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # cudaPackages.* (cuda_nvcc, cuda_cudart, cuda_cccl) are unfree
        # under the CUDA EULA and won't evaluate without this. cudaSupport
        # stays false on purpose — we use prebuilt wheels for everything
        # GPU-related, we only need nixpkgs to give us nvcc and the
        # toolchain headers, not to recompile e.g. opencv with CUDA.
        config = {
          allowUnfree = true;
          cudaSupport = false;
        };
      };
      lib = pkgs.lib;

      # NVIDIA userspace driver libs come from the host kernel module, not Nix.
      # libcuda.so + libnvidia-ml.so are dlopen'd at runtime by the engines and
      # by /usr/bin/nvidia-smi; without this on LD_LIBRARY_PATH they are not
      # found. Verified absent of libnccl* here, so this path is safe to add
      # alongside wheel-bundled NCCL (see NATIVE_DEPS.md §"Common").
      driverLib = "/usr/lib/x86_64-linux-gnu";

      # Tools every engine shell needs: uv (lockfile-driven installer), Python
      # 3.12 (pyproject pin), and a compiler toolchain because TRT-LLM's
      # mpi4py and any flash-attn rebuild are sdist-only on Py 3.12.
      commonTools = with pkgs; [
        uv
        python312
        git
        gcc
        gnumake
        cmake
        pkg-config
        which
        binutils
        # Some Python deps ship sdist-only on 3.13 and build via maturin /
        # setuptools-rust (e.g. outlines-core, sglang's transitive dep).
        rustc
        cargo
        # ninja: flashinfer / vllm / sglang JIT-build kernels via ninja.
        # Without it the first launch falls back to make and is much slower.
        ninja
        # ccache wraps gcc/nvcc so a re-JIT of the same kernel (across
        # engine restarts, across vllm vs sglang sharing a triton kernel)
        # is a hash hit instead of a fresh compile. ~10x speedup on warm
        # launches.
        ccache
        # Operator / smoke-test tools — curl drives smoke.sh, jq parses
        # the response, htop/nvtop are nice on a 700W TDP host.
        curl
        jq
        htop
      ];

      # Libraries that prebuilt wheels link against at runtime. libstdc++ is
      # the load-bearing one — torch/vllm/sglang wheels are compiled against
      # a recent libstdc++ that the host glibc can't supply on its own.
      # The rest are common-but-quietly-needed: libffi (cffi/pyzmq), openssl
      # (urllib3 cert bundle on some wheels), libtinfo (curses-using TUIs
      # like vllm's progress bar), xz (transformers occasionally), nccl
      # is intentionally omitted because the engines bundle their own.
      commonRuntimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        glib
        libffi
        openssl
        ncurses
        xz
        # tensorrt_llm's C++ extension dlopens libpython3.12.so without an
        # rpath into the Python it was built against. Adding python312
        # here puts /nix/store/.../python3.../lib on LD_LIBRARY_PATH so
        # the dlopen resolves; harmless for vllm/sglang.
        python312
      ];

      # SGLang loads `deep_gemm` from sglang.layers.quantization, and
      # deep_gemm.__init__ JIT-initialises with `_find_cuda_home()` which
      # asserts that $CUDA_HOME or `which nvcc` resolves. Torch's bundled
      # nvidia/* wheels ship runtime libs but not nvcc — so we provide the
      # CUDA 12.9 toolchain from nixpkgs and point CUDA_HOME at a merged
      # directory below. cu129 here matches the torch=2.9.1+cu129 wheel.
      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-12.9-merged";
        paths = with pkgs.cudaPackages; [
          cuda_nvcc       # nvcc compiler driver
          cuda_cudart     # libcudart + headers (already in torch wheel, but
                          # nvcc looks for them under CUDA_HOME)
          cuda_cccl       # CUB / Thrust headers — deep_gemm includes them
        ];
      };

      # Per-engine system deps. These mirror the apt-get lines in NATIVE_DEPS.md
      # but pulled from nixpkgs:
      #   libnuma-dev / libnuma1   → numactl
      #   libopenmpi-dev           → openmpi (multi-output: bin+dev+lib)
      #   libczmq4 / libczmq-dev   → czmq    (4.2.1 — same SONAME as libczmq4)
      #   libzmq3-dev              → zeromq  (4.3.x; libzmq.so.5)
      sglangSysDeps = with pkgs; [ numactl openmpi czmq zeromq cudaToolkit ];
      trtllmSysDeps = with pkgs; [ openmpi zeromq cudaToolkit ];
      vllmSysDeps   = [ cudaToolkit ];

      mkEngineShell = { name, projectDir, sysDeps }:
        pkgs.mkShellNoCC {
          inherit name;
          packages = commonTools ++ sysDeps;

          shellHook = ''
            # Locate the engine subdir by walking up from $PWD until we find
            # the flake.nix at the repo root, then appending the project dir.
            # `nix develop` keeps the user's cwd; ${toString ./.} is the
            # /nix/store copy of the flake, which is read-only and wrong for
            # `uv sync`. This walks back to the writable checkout.
            __root="$PWD"
            while [ "$__root" != "/" ] && [ ! -f "$__root/flake.nix" ]; do
              __root="$(dirname "$__root")"
            done
            if [ -f "$__root/flake.nix" ]; then
              export PROJECT_DIR="$__root/${projectDir}"
            else
              export PROJECT_DIR="$PWD/${projectDir}"
            fi
            unset __root

            # hf_transfer is in every venv; flip the flag so HfApi uses it.
            export HF_HUB_ENABLE_HF_TRANSFER=1

            # Pin the venv to the project tree so `uv sync` from any subdir
            # writes there (matches the existing vllm/.venv).
            export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv"

            # Pin to CPython 3.12. The lockfiles were resolved against 3.12
            # and several transitive deps (pyyaml 6.0.1, vllm-flash-attn,
            # some flashinfer cubins) ship no 3.13 wheels — sdist builds
            # blow up on 3.13's wheel-tag assertion. uv will download a
            # managed 3.12 if it isn't already present.
            export UV_PYTHON=3.12

            # The TensorRT-LLM wheel from pypi.nvidia.com is multi-GB and
            # routinely takes longer than uv's default 30s HTTP timeout to
            # transfer. Bump to 10 minutes so cold syncs don't fail mid-
            # download. Harmless for the other engines.
            export UV_HTTP_TIMEOUT=600

            # Persistent JIT compile caches. Each engine's first launch
            # triggers Triton kernel codegen + Inductor + flashinfer ninja
            # builds; without explicit dirs these caches live under /tmp
            # and get wiped on container restart. Shared root keeps the
            # caches warm across vllm/sglang switches when the kernels
            # match (they often do — same triton, same flashinfer pin).
            : "''${LLM_CACHE_ROOT:=$HOME/.cache/llm-benchmark}"
            export TRITON_CACHE_DIR="$LLM_CACHE_ROOT/triton"
            export TORCHINDUCTOR_CACHE_DIR="$LLM_CACHE_ROOT/inductor"
            export FLASHINFER_WORKSPACE_BASE="$LLM_CACHE_ROOT/flashinfer"
            export VLLM_CACHE_ROOT="$LLM_CACHE_ROOT/vllm"
            export CCACHE_DIR="$LLM_CACHE_ROOT/ccache"
            mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
                     "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" \
                     "$CCACHE_DIR"

            # Wrap gcc/g++/nvcc with ccache so re-JIT of identical sources
            # (very common — the same flashinfer kernel signature recurs
            # across launches) hits cache instead of recompiling. Triton
            # has its own object cache; ccache covers the C/CXX/CUDA path
            # used by torch.utils.cpp_extension and flashinfer's ninja
            # rules. NVCC_CCACHE_ENABLE makes nvcc's preprocessor stable
            # so the hash is deterministic.
            if command -v ccache >/dev/null 2>&1; then
              export CC="ccache gcc"
              export CXX="ccache g++"
              export NVCC_PREPEND_FLAGS="''${NVCC_PREPEND_FLAGS:-} -ccbin gcc"
              ccache --max-size=20G >/dev/null 2>&1 || true
            fi

            # mpi4py's build reads $MPICC. openmpi already puts it on PATH,
            # but exporting helps uv's isolated build env keep it.
            if command -v mpicc >/dev/null 2>&1; then
              export MPICC="$(command -v mpicc)"
            fi

            # deep_gemm + any flash-attn rebuild look here. Point at the
            # merged toolchain (nvcc + cudart + cccl). NVCC_PREPEND_FLAGS
            # adds the CCCL include dir so deep_gemm's #include <cuda/...>
            # resolves without the user having to set it.
            if command -v nvcc >/dev/null 2>&1; then
              export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
              export CUDA_PATH="$CUDA_HOME"
            fi

            # Host driver libs (libcuda, libnvidia-ml) + Nix runtime libs
            # (libstdc++, zlib, glib) needed by prebuilt torch / engine wheels.
            export LD_LIBRARY_PATH="${lib.makeLibraryPath (commonRuntimeLibs ++ sysDeps)}:${driverLib}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Build-time linker also needs to see libcuda.so. flashinfer
            # JIT-compiles kernels (sglang's tinygemm2, vllm's piecewise
            # CUDA graphs) and links with `-lcuda`; without LIBRARY_PATH
            # pointing at the host driver dir, ld.bfd errors with
            # "cannot find -lcuda" and the build fails before generation
            # starts. Headers come from cudaToolkit, the stub library
            # comes from the host driver package.
            export LIBRARY_PATH="${driverLib}''${LIBRARY_PATH:+:$LIBRARY_PATH}"

            # Hint a sync command tailored to this host. Driver detection
            # via nvidia-smi falls back to "driver-r580" (the project's
            # default-supported branch) so the message is still useful in
            # CI / docker images without a driver loaded.
            __drv=$(LD_LIBRARY_PATH="${driverLib}:$LD_LIBRARY_PATH" \
                    /usr/bin/nvidia-smi --query-gpu=driver_version \
                    --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
            __drv="''${__drv:-580}"
            echo "▶ ${name} devshell ready. Project: $PROJECT_DIR"
            ${if name == "trt-llm" then ''
              # Hard-fail shell entry on unsupported drivers. The
              # tensorrt-llm 1.2.x cu13 stack references CUDA Driver
              # API symbols (cuKernelGetName, added in CUDA 12.4) that
              # don't exist in libcuda.so on r5xx (≤r575) hosts. A
              # successful `uv sync` followed by an ImportError at
              # `import tensorrt_llm` is much more confusing than
              # refusing to enter the shell at all.
              if [ -z "''${__drv:-}" ] || [ "$__drv" -lt 580 ] 2>/dev/null; then
                echo "  ✘ trt-llm devshell requires NVIDIA driver r580+. " >&2
                echo "    Detected: r''${__drv:-<none>}." >&2
                echo "    cu13 torch wheels need CUDA Driver API symbols" >&2
                echo "    (e.g. cuKernelGetName) that pre-r550 libcuda.so" >&2
                echo "    does not export — 'import tensorrt_llm' would" >&2
                echo "    crash with 'undefined symbol'. Use the vllm or" >&2
                echo "    sglang devshell on this host instead." >&2
                exit 1
              fi
              echo "  Sync with:  cd \"$PROJECT_DIR\" && uv sync --extra driver-r''${__drv}"
            '' else ''
              echo "  Sync with:  cd \"$PROJECT_DIR\" && uv sync --extra h100"
            ''}
            unset __drv
          '';
        };
    in {
      devShells.${system} = {
        vllm    = mkEngineShell { name = "vllm";    projectDir = "vllm";    sysDeps = vllmSysDeps;   };
        sglang  = mkEngineShell { name = "sglang";  projectDir = "sglang";  sysDeps = sglangSysDeps; };
        trt-llm = mkEngineShell { name = "trt-llm"; projectDir = "trt-llm"; sysDeps = trtllmSysDeps; };

        default = self.devShells.${system}.vllm;
      };
    };
}
