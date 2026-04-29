{
  description = "LLM serving benchmark — devshells for vLLM, SGLang, TensorRT-LLM";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

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
        config = { allowUnfree = true; cudaSupport = false; };
      };
      lib = pkgs.lib;

      # Host driver libs (libcuda.so + libnvidia-ml.so) come from the
      # kernel module, not Nix. /usr/lib/x86_64-linux-gnu has them on
      # this container; libnccl* is verified absent (engines bundle
      # their own NCCL via wheels).
      driverLib = "/usr/lib/x86_64-linux-gnu";

      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-12.9-merged";
        paths = with pkgs.cudaPackages; [ cuda_nvcc cuda_cudart cuda_cccl ];
      };

      # Note: no python here. UV_PYTHON_PREFERENCE=only-managed makes uv
      # download python-build-standalone, whose interpreter's PT_INTERP
      # is /lib64/ld-linux-x86-64.so.2 → on this nix-ld container that's
      # nix-ld, which honors NIX_LD_LIBRARY_PATH. nixpkgs' own python's
      # PT_INTERP is the Nix glibc loader and ignores it, so mixing the
      # two is what previously forced LD_LIBRARY_PATH into the picture.
      common = with pkgs; [
        uv git gcc gnumake cmake pkg-config which binutils
        ninja rustc cargo curl jq htop
      ];

      # Runtime libs the manylinux wheels dlopen.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib zlib glib libffi openssl ncurses xz
      ];

      ldPath = sysDeps:
        lib.makeLibraryPath (runtimeLibs ++ sysDeps) + ":" + driverLib;

      cacheHook = ''
        : "''${LLM_CACHE_ROOT:=$HOME/.cache/llm-benchmark}"
        export TRITON_CACHE_DIR="$LLM_CACHE_ROOT/triton"
        export TORCHINDUCTOR_CACHE_DIR="$LLM_CACHE_ROOT/inductor"
        export FLASHINFER_WORKSPACE_BASE="$LLM_CACHE_ROOT/flashinfer"
        export VLLM_CACHE_ROOT="$LLM_CACHE_ROOT/vllm"
        mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
                 "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT"
      '';

      # Thin attribute-merging helper around mkShellNoCC. Sets the env
      # vars every engine shares; the per-engine call overrides whatever
      # it needs. Not a derivation factory — each shell stays explicit.
      mkShell = attrs: pkgs.mkShellNoCC ({
        HF_HUB_ENABLE_HF_TRANSFER = "1";
        UV_PYTHON = "3.12";
        # Avoid pkgs.python312 so the venv's python is uv-managed
        # (nix-ld-loaded), then NIX_LD_LIBRARY_PATH alone is enough.
        UV_PYTHON_PREFERENCE = "only-managed";
        # Multi-GB tensorrt-llm wheel from pypi.nvidia.com.
        UV_HTTP_TIMEOUT = "600";
        CUDA_HOME = "${cudaToolkit}";
        CUDA_PATH = "${cudaToolkit}";
        # Skip Triton's libcuda discovery via /sbin/ldconfig — the
        # ldconfig wrapper here is a nushell script that crashes under
        # nixpkgs-unstable's glibc.
        TRITON_LIBCUDA_PATH = driverLib;
        # Build-time linker for flashinfer's `-lcuda`.
        LIBRARY_PATH = driverLib;
        shellHook = cacheHook;
      } // attrs);
    in {
      devShells.${system} = {
        vllm = mkShell {
          name = "vllm";
          packages = common ++ [ cudaToolkit ];
          NIX_LD_LIBRARY_PATH = ldPath [ ];
        };

        sglang = mkShell {
          name = "sglang";
          # apt-get → nixpkgs translation:
          #   libnuma-dev → numactl, libopenmpi-dev → openmpi,
          #   libczmq-dev → czmq (4.2.1 = libczmq.so.4),
          #   libzmq3-dev → zeromq.
          packages = common ++ (with pkgs; [
            cudaToolkit numactl openmpi czmq zeromq
          ]);
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          NIX_LD_LIBRARY_PATH = ldPath (with pkgs; [
            numactl openmpi czmq zeromq
          ]);
        };

        trt-llm = mkShell {
          name = "trt-llm";
          packages = common ++ (with pkgs; [ cudaToolkit openmpi zeromq ]);
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          NIX_LD_LIBRARY_PATH = ldPath (with pkgs; [ openmpi zeromq ]);
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
