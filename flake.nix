{
  description = "LLM serving benchmark — devshells for vLLM, SGLang, TensorRT-LLM";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  # Pull prebuilt unfree cuda* derivations from the cuda-maintainers cache.
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
        # cudaPackages.* is unfree under CUDA EULA. cudaSupport stays off:
        # we use prebuilt wheels for everything GPU-related, we only need
        # nixpkgs to give us nvcc + headers.
        config = { allowUnfree = true; cudaSupport = false; };
      };
      lib = pkgs.lib;

      # NVIDIA userspace driver libs come from the host kernel module, not
      # Nix. /usr/lib/x86_64-linux-gnu has libcuda.so + libnvidia-ml.so on
      # this container; verified absent of libnccl* (engines bundle their
      # own — a host NCCL on LD_LIBRARY_PATH would clash at import).
      driverLib = "/usr/lib/x86_64-linux-gnu";

      # cuda_nvcc (compiler), cuda_cudart (runtime headers nvcc looks for
      # under CUDA_HOME), cuda_cccl (CUB/Thrust headers deep_gemm pulls in).
      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-12.9-merged";
        paths = with pkgs.cudaPackages; [ cuda_nvcc cuda_cudart cuda_cccl ];
      };

      common = with pkgs; [
        uv python312 git gcc gnumake cmake pkg-config which binutils
        # ninja: flashinfer / vllm / sglang JIT-build kernels with it.
        ninja
        # rustc/cargo: outlines-core (sglang transitive dep) builds via
        # setuptools-rust on Py 3.13+ wheels-missing paths.
        rustc cargo
        # operator + smoke-test tools.
        curl jq htop
      ];

      # Runtime libs prebuilt wheels dlopen. libstdc++ is load-bearing —
      # torch/vllm/sglang wheels are linked against a recent libstdc++.
      # python312 here puts /nix/store/.../python3-*/lib on LD_LIBRARY_PATH
      # so tensorrt_llm's C++ extension can dlopen libpython3.12.so.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib zlib glib libffi openssl ncurses xz python312
      ];

      # /usr/lib/x86_64-linux-gnu must come last (lowest priority) — Nix
      # libs win, host driver fills in libcuda/libnvidia-ml only.
      ldPath = sysDeps:
        lib.makeLibraryPath (runtimeLibs ++ sysDeps) + ":" + driverLib;

      # Persistent JIT caches (flashinfer ninja, Triton, Inductor) — the
      # only thing we can't express as static mkShell env attrs because
      # $HOME isn't expanded at flake eval. Tiny shellHook handles it.
      cacheHook = ''
        : "''${LLM_CACHE_ROOT:=$HOME/.cache/llm-benchmark}"
        export TRITON_CACHE_DIR="$LLM_CACHE_ROOT/triton"
        export TORCHINDUCTOR_CACHE_DIR="$LLM_CACHE_ROOT/inductor"
        export FLASHINFER_WORKSPACE_BASE="$LLM_CACHE_ROOT/flashinfer"
        export VLLM_CACHE_ROOT="$LLM_CACHE_ROOT/vllm"
        mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
                 "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT"
      '';
    in {
      devShells.${system} = {
        vllm = pkgs.mkShellNoCC {
          name = "vllm";
          packages = common ++ [ cudaToolkit ];

          HF_HUB_ENABLE_HF_TRANSFER = "1";
          UV_PYTHON = "3.12";
          # tensorrt-llm wheel from pypi.nvidia.com is multi-GB;
          # harmless for the lighter engines.
          UV_HTTP_TIMEOUT = "600";
          CUDA_HOME = "${cudaToolkit}";
          CUDA_PATH = "${cudaToolkit}";
          # Skip Triton's libcuda discovery via /sbin/ldconfig — on
          # this nix-ld container ldconfig is a nushell wrapper built
          # against an older glibc and crashes under nixpkgs-unstable.
          TRITON_LIBCUDA_PATH = driverLib;
          # Build-time linker needs libcuda.so for flashinfer's
          # `-lcuda` (sglang tinygemm2, vllm piecewise CUDA graphs).
          LIBRARY_PATH = driverLib;
          LD_LIBRARY_PATH = ldPath [ ];

          shellHook = cacheHook;
        };

        sglang = pkgs.mkShellNoCC {
          name = "sglang";
          # apt-get → nixpkgs translation:
          #   libnuma-dev → numactl, libopenmpi-dev → openmpi,
          #   libczmq-dev → czmq (4.2.1 = libczmq.so.4),
          #   libzmq3-dev → zeromq.
          packages = common ++ (with pkgs; [
            cudaToolkit numactl openmpi czmq zeromq
          ]);

          HF_HUB_ENABLE_HF_TRANSFER = "1";
          UV_PYTHON = "3.12";
          UV_HTTP_TIMEOUT = "600";
          CUDA_HOME = "${cudaToolkit}";
          CUDA_PATH = "${cudaToolkit}";
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          TRITON_LIBCUDA_PATH = driverLib;
          LIBRARY_PATH = driverLib;
          LD_LIBRARY_PATH = ldPath (with pkgs; [
            numactl openmpi czmq zeromq
          ]);

          shellHook = cacheHook;
        };

        trt-llm = pkgs.mkShellNoCC {
          name = "trt-llm";
          packages = common ++ (with pkgs; [ cudaToolkit openmpi zeromq ]);

          HF_HUB_ENABLE_HF_TRANSFER = "1";
          UV_PYTHON = "3.12";
          UV_HTTP_TIMEOUT = "600";
          CUDA_HOME = "${cudaToolkit}";
          CUDA_PATH = "${cudaToolkit}";
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          TRITON_LIBCUDA_PATH = driverLib;
          LIBRARY_PATH = driverLib;
          LD_LIBRARY_PATH = ldPath (with pkgs; [ openmpi zeromq ]);

          shellHook = cacheHook + ''
            # tensorrt-llm 1.2 cu13 stack references CUDA Driver API
            # symbols (cuKernelGetName, added in CUDA 12.4) that aren't
            # in libcuda.so on r5xx (≤r575). Refuse shell entry rather
            # than producing an unrunnable venv.
            drv=$(LD_LIBRARY_PATH="${driverLib}" /usr/bin/nvidia-smi \
              --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
              | head -1 | cut -d. -f1)
            if [ -z "''${drv:-}" ] || [ "$drv" -lt 580 ] 2>/dev/null; then
              echo "✘ trt-llm devshell requires NVIDIA driver r580+; got r''${drv:-<none>}" >&2
              echo "  Use vllm or sglang on this host instead." >&2
              exit 1
            fi
          '';
        };

        default = self.devShells.${system}.vllm;
      };
    };
}
