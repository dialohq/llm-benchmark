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

      # ── CUDA versions required at runtime ─────────────────────────────────
      # Two CUDA major versions live in each engine venv simultaneously:
      #
      #   cu13 — torch 2.10+cu130 wheel: every nvidia/cu13/lib/lib*.so.13,
      #          libcublasLt.so.13, libcusparse.so.12, etc.; loaded by
      #          torch at import time. Headers under nvidia/cu13/include
      #          are what flashinfer's JIT compiler must use so the .so it
      #          produces is ABI-compatible with the torch runtime.
      #
      #   cu12 — vllm 0.19.1 wheel: vllm._C.abi3.so DT_NEEDED is libcudart.so.12.
      #          The vllm wheel does NOT bundle cu12 libs (the nvidia/cu12/
      #          dir is missing from the venv) — it relies on the host
      #          providing libcudart.so.12. We supply it from nixpkgs.
      #          *Runtime only* — never on LIBRARY_PATH (so the JIT linker
      #          can't accidentally pick up cu12 .so symlinks for -lcudart).
      #
      # Bumping vllm to a cu13-built wheel (when one ships) lets us drop the
      # cu12 runtime side. Until then both must coexist; mismatch is what
      # caused the iter-0 ImportError loop on this branch.
      cudaMajorRuntime = "12";    # vllm._C wheel is cu12-built
      cudaMajorJit     = "13";    # torch wheel + flashinfer JIT use cu13
      cu12Pkgs = with pkgs.cudaPackages; [
        cuda_nvcc cuda_cudart cuda_cccl
      ];
      cu12Toolkit = pkgs.symlinkJoin {
        name = "cuda-12.9-nvcc+runtime";
        paths = lib.concatMap (p: map (o: p.${o}) p.outputs) cu12Pkgs;
      };
      # Kept under the old name for the rest of the file; the value still
      # provides nvcc + cu12 runtime libs (libcudart.so.12 etc.).
      cudaToolkit = cu12Toolkit;

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

      # Per-engine env hook. Sets up paths for the engine's bundled cu13
      # stack AND the cu12 runtime side, then verifies both versions of
      # libcudart resolve before letting the shell continue.
      #
      # Order rationale:
      #   * NIX_LD_LIBRARY_PATH (runtime dlopen via nix-ld):
      #       cu13 venv libs ▸ cudnn ▸ nccl ▸ cu12 toolkit ▸ rest.
      #       libcudart.so.12 vs libcudart.so.13 are different SONAMEs so
      #       both resolve regardless of order; cu13 first so unversioned
      #       libs (e.g. libcusparse.so → 12 on both sides) prefer the venv.
      #   * LIBRARY_PATH (JIT linker `-lcudart`):
      #       cu13 venv only. NEVER include cu12 toolkit's lib here, or the
      #       JIT will produce a .so depending on libcudart.so.12 that then
      #       conflicts with torch's loaded libcudart.so.13.
      #   * CPATH (JIT preprocessor `-isystem`):
      #       cu13 headers only.
      cudaEnvHook = engineDir: cuMajor: ''
        # Locate repo root by walking up looking for flake.nix; fall back to
        # $PWD so callers entering from the repo root still work without it.
        _root="$PWD"
        while [ "$_root" != "/" ] && [ ! -e "$_root/flake.nix" ]; do
          _root="$(dirname "$_root")"
        done
        [ -e "$_root/flake.nix" ] || _root="$PWD"
        if [ -d "$_root/${engineDir}/.venv" ]; then
          NV="$_root/${engineDir}/.venv/lib/python3.12/site-packages/nvidia"
          for need in "$NV/cu${cuMajor}/lib/libcudart.so.${cuMajor}" \
                      "$NV/cu${cuMajor}/include/cublasLt.h" \
                      "${cudaToolkit}/lib/libcudart.so.${cudaMajorRuntime}"; do
            if [ ! -e "$need" ]; then
              echo "✘ ${engineDir} devshell: missing $need" >&2
              echo "  cu${cuMajor} venv libs come from \`uv sync\` in ${engineDir}/." >&2
              echo "  cu${cudaMajorRuntime} runtime libs come from nixpkgs cudaPackages." >&2
              return 1
            fi
          done
          export NIX_LD_LIBRARY_PATH="$NV/cu${cuMajor}/lib:$NV/cudnn/lib:$NV/nccl/lib:${cudaToolkit}/lib:''${NIX_LD_LIBRARY_PATH:-}"
          export LIBRARY_PATH="$NV/cu${cuMajor}/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
          export CPATH="$NV/cu${cuMajor}/include:$NV/cudnn/include:''${CPATH:-}"
          echo "✓ ${engineDir}: cu${cuMajor} (jit/torch) + cu${cudaMajorRuntime} (vllm._C runtime) verified" >&2
        else
          echo "ℹ ${engineDir}/.venv missing under $_root — run \`uv sync\` to materialize cu${cuMajor} libs" >&2
        fi
        unset _root
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
          shellHook = cacheHook + cudaEnvHook "vllm" cudaMajorJit;
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
          shellHook = cacheHook + cudaEnvHook "sglang" cudaMajorJit;
        };

        trt-llm = mkShell {
          name = "trt-llm";
          packages = common ++ (with pkgs; [ cudaToolkit openmpi zeromq ]);
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          NIX_LD_LIBRARY_PATH = ldPath (with pkgs; [ openmpi zeromq ]);
          shellHook = cacheHook + cudaEnvHook "trt-llm" cudaMajorJit + ''
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
