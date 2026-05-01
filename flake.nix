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
      # cudaToolkit is the *compiler*-only path. Keeping libs out of here is
      # deliberate: flashinfer's run_ninja hard-codes
      #   `-L${CUDA_HOME}/lib64 -lcudart`
      # so any libcudart.so.12 in cudaToolkit/lib would beat LIBRARY_PATH=cu13.
      # Empty lib/lib64 ensures `-lcudart` falls through to LIBRARY_PATH where
      # cu13 wins, producing a JIT .so that's ABI-compatible with torch.
      cudaToolkitPkgs = with pkgs.cudaPackages; [ cuda_nvcc cuda_cccl ];
      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-12.9-nvcc-only";
        paths = lib.concatMap (p: map (o: p.${o}) p.outputs) cudaToolkitPkgs;
      };
      # Separate cu12 runtime path. Reachable only via NIX_LD_LIBRARY_PATH
      # (runtime dlopen of libcudart.so.12 by vllm._C wheel) — never on
      # LIBRARY_PATH or -L flags. cuda_cudart is multi-output; the `.lib`
      # output (when present) holds the .so files; otherwise resolve the
      # named output via getOutput.
      cu12RuntimeLibs = lib.getOutput "lib" pkgs.cudaPackages.cuda_cudart;

      # cu13's cudaTypedefs.h dropped the unversioned `PFN_X` macro aliases
      # that cu12 carried (PFN_X is now only present as `PFN_X_v12000`).
      # flashinfer 0.6.6's bundled cutlass references the unversioned forms
      # (e.g. PFN_cuTensorMapEncodeTiled) and won't compile against cu13
      # headers alone. Build a shim include dir whose cudaTypedefs.h
      # `#include_next`s cu13's real header then adds back the macro alias
      # block extracted verbatim from cu12 cudart. Put this dir FIRST on
      # CPATH; everything else falls through to cu13.
      cu13TypedefShim = pkgs.runCommand "cu13-typedef-shim" { } ''
        mkdir -p $out/include
        aliases=$(mktemp)
        grep -E '^#define PFN_cu' \
          ${cu12RuntimeLibs}/include/cudaTypedefs.h > "$aliases"
        {
          echo '#pragma once'
          echo '#include_next <cudaTypedefs.h>'
          # Guard each alias so a future cu13 that re-adds them won't redefine.
          while read -r line; do
            name=$(echo "$line" | awk '{print $2}')
            echo "#ifndef $name"
            echo "$line"
            echo "#endif"
          done < "$aliases"
        } > $out/include/cudaTypedefs.h
        rm -f "$aliases"
      '';

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

      # Default cache lives inside the repo (.cache/) so it's discoverable,
      # gitignorable, and trivially relocatable with the checkout. Override
      # by exporting LLM_CACHE_ROOT to an absolute path before entering the
      # devshell.
      cacheHook = ''
        if [ -z "''${LLM_CACHE_ROOT:-}" ]; then
          _croot="$PWD"
          while [ "$_croot" != "/" ] && [ ! -e "$_croot/flake.nix" ]; do
            _croot="$(dirname "$_croot")"
          done
          [ -e "$_croot/flake.nix" ] || _croot="$PWD"
          export LLM_CACHE_ROOT="$_croot/.cache"
          unset _croot
        fi
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
                      "${cu12RuntimeLibs}/lib/libcudart.so.${cudaMajorRuntime}"; do
            if [ ! -e "$need" ]; then
              echo "✘ ${engineDir} devshell: missing $need" >&2
              echo "  cu${cuMajor} venv libs come from \`uv sync\` in ${engineDir}/." >&2
              echo "  cu${cudaMajorRuntime} runtime libs come from nixpkgs cudaPackages." >&2
              return 1
            fi
          done
          # The cu13 venv ships only versioned SONAMEs (libcudart.so.13). The
          # JIT linker's `-lcudart` looks for unversioned `libcudart.so` first,
          # so without these compat symlinks it falls through past LIBRARY_PATH
          # and fails ENOENT. Create the symlinks idempotently in the venv.
          for soname in cudart cudart_static cublas cublasLt cusparse cusolver \
                        cufft curand cufile cupti nvJitLink nvrtc nvrtc-builtins; do
            if [ -e "$NV/cu${cuMajor}/lib/lib''${soname}.so.${cuMajor}" ] && \
               [ ! -e "$NV/cu${cuMajor}/lib/lib''${soname}.so" ]; then
              ln -s "lib''${soname}.so.${cuMajor}" \
                    "$NV/cu${cuMajor}/lib/lib''${soname}.so" 2>/dev/null || true
            fi
          done
          export NIX_LD_LIBRARY_PATH="$NV/cu${cuMajor}/lib:$NV/cudnn/lib:$NV/nccl/lib:${cu12RuntimeLibs}/lib:''${NIX_LD_LIBRARY_PATH:-}"
          export LIBRARY_PATH="$NV/cu${cuMajor}/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
          # CPATH order: typedef shim FIRST (its cudaTypedefs.h adds the
          # unversioned PFN_* macro aliases cu13 dropped — flashinfer 0.6.6's
          # bundled cutlass needs them; #include_next falls through to cu13's
          # real cudaTypedefs.h for everything else), then cu13 venv includes,
          # then cudnn. Pulling all of cu12 cudart's include onto CPATH
          # instead would shadow cu13's cooperative_groups, cublasLt, etc.
          export CPATH="${cu13TypedefShim}/include:$NV/cu${cuMajor}/include:$NV/cudnn/include:''${CPATH:-}"
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
