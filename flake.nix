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

      # Two CUDA major versions live in each engine venv simultaneously:
      #
      #   cu13 — torch 2.10+cu130 wheel: every nvidia/cu13/lib/lib*.so.13,
      #          libcublasLt.so.13, etc.; loaded by torch at import time.
      #          Headers under nvidia/cu13/include are what flashinfer's
      #          JIT compiler must use so the .so it produces is ABI-
      #          compatible with the torch runtime.
      #
      #   cu12 — vllm 0.19.1 wheel: vllm._C.abi3.so DT_NEEDED is
      #          libcudart.so.12. The vllm wheel does NOT bundle cu12 libs
      #          — it relies on the host providing libcudart.so.12. We
      #          supply it from nixpkgs. *Runtime only* — never on
      #          LIBRARY_PATH (so the JIT linker can't accidentally pick
      #          up cu12 .so for `-lcudart`).
      #
      # Bumping vllm to a cu13-built wheel (when one ships) lets us drop the
      # cu12 side. Until then both must coexist; mismatch is what caused
      # the iter-0 ImportError loop on this branch.

      # cudaToolkit is the *compiler*-only path. Keeping libs out of here
      # is deliberate: flashinfer's run_ninja hard-codes
      #   `-L${CUDA_HOME}/lib64 -lcudart`
      # so any libcudart.so.12 in cudaToolkit/lib would beat LIBRARY_PATH=cu13.
      # Empty lib/lib64 ensures `-lcudart` falls through to LIBRARY_PATH where
      # cu13 (from the venv) wins, producing a JIT .so ABI-compatible with torch.
      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-12.9-nvcc-only";
        paths = lib.concatMap (p: map (o: p.${o}) p.outputs)
          (with pkgs.cudaPackages; [ cuda_nvcc cuda_cccl ]);
      };

      # cu12 cudart for vllm._C wheel's runtime dlopen of libcudart.so.12.
      # `.lib` output holds the .so files; `dev` (colocated with `out`) holds
      # the include/ used by cu13TypedefShim below.
      cu12Cudart = lib.getOutput "lib" pkgs.cudaPackages.cuda_cudart;

      # cu12 libs sglang's bundled sgl_kernel + flashinfer kernels dlopen at
      # runtime. Kept *outside* CUDA_HOME so the JIT linker doesn't pick up
      # cu12 .so for `-lcudart` / `-lcublas` (cu13 in LIBRARY_PATH must win
      # for ABI compat with torch). sglang only.
      cu12Extras = pkgs.symlinkJoin {
        name = "cu12-extras";
        paths = with pkgs.cudaPackages; [
          (lib.getOutput "lib" cuda_nvrtc)
          (lib.getOutput "lib" libcublas)
          (lib.getOutput "lib" libcusparse)
          (lib.getOutput "lib" libcusolver)
          (lib.getOutput "lib" libcurand)
          (lib.getOutput "lib" libcufft)
          (lib.getOutput "lib" cuda_cupti)
        ];
      };

      # cu13's cudaTypedefs.h dropped the unversioned `PFN_X` macro aliases
      # that cu12 carried (PFN_X is now only present as `PFN_X_v12000`).
      # flashinfer 0.6.6's bundled cutlass references the unversioned forms
      # and won't compile against cu13 headers alone. Build a shim include
      # dir whose cudaTypedefs.h `#include_next`s cu13's real header then
      # adds back the macro alias block extracted verbatim from cu12 cudart.
      # Put this dir FIRST on CPATH; everything else falls through to cu13.
      # vllm + sglang only (trt-llm doesn't JIT-compile cutlass).
      cu13TypedefShim = pkgs.runCommand "cu13-typedef-shim" { } ''
        mkdir -p $out/include
        aliases=$(mktemp)
        grep -E '^#define PFN_cu' \
          ${cu12Cudart}/include/cudaTypedefs.h > "$aliases"
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
      runtimeLibs = lib.makeLibraryPath (with pkgs; [
        stdenv.cc.cc.lib zlib glib libffi openssl ncurses xz
      ]);

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

      # Walk up from $PWD looking for flake.nix; sets $_root (caller unsets).
      findRoot = ''
        _root="$PWD"
        while [ "$_root" != "/" ] && [ ! -e "$_root/flake.nix" ]; do
          _root="$(dirname "$_root")"
        done
        [ -e "$_root/flake.nix" ] || _root="$PWD"
      '';

      # cu13 venv ships only versioned SONAMEs (libcudart.so.13). The JIT
      # linker's `-lcudart` looks for unversioned `libcudart.so` first, so
      # without these compat symlinks it falls through past LIBRARY_PATH
      # and fails ENOENT. Idempotent.
      cu13SymlinkLoop = ''
        for s in cudart cudart_static cublas cublasLt cusparse cusolver \
                 cufft curand cufile cupti nvJitLink nvrtc nvrtc-builtins; do
          if [ -e "$NV/cu13/lib/lib''${s}.so.13" ] && \
             [ ! -e "$NV/cu13/lib/lib''${s}.so" ]; then
            ln -s "lib''${s}.so.13" "$NV/cu13/lib/lib''${s}.so" \
              2>/dev/null || true
          fi
        done
      '';
    in {
      devShells.${system} = {
        # ─────────── vllm ────────────────────────────────────────────────────
        # cu13 venv libs (torch + flashinfer JIT) + cu12 cudart (vllm._C
        # runtime). Flashinfer JIT needs cu13TypedefShim for cutlass.
        vllm = pkgs.mkShellNoCC {
          name = "vllm";
          packages = common ++ [ cudaToolkit ];
          HF_HUB_ENABLE_HF_TRANSFER = "1";
          UV_PYTHON = "3.12";
          # Avoid pkgs.python312 so the venv's python is uv-managed
          # (nix-ld-loaded), then NIX_LD_LIBRARY_PATH alone is enough.
          UV_PYTHON_PREFERENCE = "only-managed";
          UV_HTTP_TIMEOUT = "600";
          CUDA_HOME = "${cudaToolkit}";
          CUDA_PATH = "${cudaToolkit}";
          # Skip Triton's libcuda discovery via /sbin/ldconfig — the
          # ldconfig wrapper here is a nushell script that crashes under
          # nixpkgs-unstable's glibc.
          TRITON_LIBCUDA_PATH = driverLib;
          # Build-time linker for flashinfer's `-lcuda`.
          LIBRARY_PATH = driverLib;
          shellHook = cacheHook + ''
            ${findRoot}
            if [ -d "$_root/vllm/.venv" ]; then
              NV="$_root/vllm/.venv/lib/python3.12/site-packages/nvidia"
              for need in "$NV/cu13/lib/libcudart.so.13" \
                          "$NV/cu13/include/cublasLt.h" \
                          "${cu12Cudart}/lib/libcudart.so.12"; do
                if [ ! -e "$need" ]; then
                  echo "✘ vllm devshell: missing $need" >&2
                  echo "  cu13 venv libs come from \`uv sync\` in vllm/." >&2
                  echo "  cu12 runtime libs come from nixpkgs cudaPackages." >&2
                  return 1
                fi
              done
              ${cu13SymlinkLoop}
              # NIX_LD_LIBRARY_PATH (runtime dlopen via nix-ld):
              #   cu13 venv ▸ cudnn ▸ nccl ▸ cu12 cudart ▸ generic runtime ▸ driver.
              #   libcudart.so.{12,13} are different SONAMEs so both resolve.
              export NIX_LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:${cu12Cudart}/lib:${runtimeLibs}:${driverLib}"
              # LIBRARY_PATH (JIT linker `-lcudart`): cu13 venv only.
              # NEVER include cu12 here, or the JIT will produce a .so
              # depending on libcudart.so.12 that conflicts with torch's
              # loaded libcudart.so.13.
              export LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
              # CPATH: typedef shim FIRST (its cudaTypedefs.h adds the
              # unversioned PFN_* macro aliases cu13 dropped — flashinfer
              # 0.6.6's cutlass needs them; #include_next falls through
              # to cu13's real header), then cu13 venv, cudnn, then
              # CUDA_HOME's include (`<nv/target>` etc).
              export CPATH="${cu13TypedefShim}/include:$NV/cu13/include:$NV/cudnn/include:${cudaToolkit}/include:''${CPATH:-}"
              echo "✓ vllm: cu13 (jit/torch) + cu12 (vllm._C runtime) verified" >&2
            else
              echo "ℹ vllm/.venv missing under $_root — run \`uv sync\` to materialize cu13 libs" >&2
            fi
            unset _root NV
          '';
        };

        # ─────────── sglang ──────────────────────────────────────────────────
        # cu13 venv libs + cu12 cudart + cu12Extras (sgl_kernel + bundled
        # flashinfer kernels dlopen libnvrtc / libcublas / libcusparse at
        # runtime). System deps for IPC + multi-GPU:
        #   libnuma-dev → numactl, libopenmpi-dev → openmpi,
        #   libczmq-dev → czmq (4.2.1 = libczmq.so.4), libzmq3-dev → zeromq.
        sglang = pkgs.mkShellNoCC {
          name = "sglang";
          packages = common ++ (with pkgs; [
            cudaToolkit numactl openmpi czmq zeromq
          ]);
          HF_HUB_ENABLE_HF_TRANSFER = "1";
          UV_PYTHON = "3.12";
          UV_PYTHON_PREFERENCE = "only-managed";
          UV_HTTP_TIMEOUT = "600";
          CUDA_HOME = "${cudaToolkit}";
          CUDA_PATH = "${cudaToolkit}";
          TRITON_LIBCUDA_PATH = driverLib;
          LIBRARY_PATH = driverLib;
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          shellHook = cacheHook + ''
            ${findRoot}
            if [ -d "$_root/sglang/.venv" ]; then
              NV="$_root/sglang/.venv/lib/python3.12/site-packages/nvidia"
              for need in "$NV/cu13/lib/libcudart.so.13" \
                          "$NV/cu13/include/cublasLt.h" \
                          "${cu12Cudart}/lib/libcudart.so.12"; do
                if [ ! -e "$need" ]; then
                  echo "✘ sglang devshell: missing $need" >&2
                  echo "  cu13 venv libs come from \`uv sync\` in sglang/." >&2
                  echo "  cu12 runtime libs come from nixpkgs cudaPackages." >&2
                  return 1
                fi
              done
              ${cu13SymlinkLoop}
              export NIX_LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:${cu12Cudart}/lib:${cu12Extras}/lib:${lib.makeLibraryPath (with pkgs; [ numactl openmpi czmq zeromq ])}:${runtimeLibs}:${driverLib}"
              export LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
              export CPATH="${cu13TypedefShim}/include:$NV/cu13/include:$NV/cudnn/include:${cudaToolkit}/include:''${CPATH:-}"
              echo "✓ sglang: cu13 (jit/torch) + cu12 (cudart + extras) verified" >&2
            else
              echo "ℹ sglang/.venv missing under $_root — run \`uv sync\` to materialize cu13 libs" >&2
            fi
            unset _root NV
          '';
        };

        # ─────────── trt-llm ─────────────────────────────────────────────────
        # cu13 venv libs + cu12 cudart. trt-llm doesn't JIT-compile cutlass
        # → no cu13TypedefShim; doesn't dlopen cu12 nvrtc/cublas → no
        # cu12Extras. openmpi + zeromq for IPC.
        #
        # tensorrt-llm 1.2 cu13 stack references CUDA Driver API symbols
        # (cuKernelGetName, added in CUDA 12.4) that aren't in libcuda.so
        # on r5xx (≤r575). Refuse shell entry rather than producing an
        # unrunnable venv.
        trt-llm = pkgs.mkShellNoCC {
          name = "trt-llm";
          packages = common ++ (with pkgs; [ cudaToolkit openmpi zeromq ]);
          HF_HUB_ENABLE_HF_TRANSFER = "1";
          UV_PYTHON = "3.12";
          UV_PYTHON_PREFERENCE = "only-managed";
          UV_HTTP_TIMEOUT = "600";
          CUDA_HOME = "${cudaToolkit}";
          CUDA_PATH = "${cudaToolkit}";
          TRITON_LIBCUDA_PATH = driverLib;
          LIBRARY_PATH = driverLib;
          MPICC = "${pkgs.openmpi}/bin/mpicc";
          shellHook = cacheHook + ''
            drv=$(LD_LIBRARY_PATH="${driverLib}" /usr/bin/nvidia-smi \
              --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
              | head -1 | cut -d. -f1)
            if [ -z "''${drv:-}" ] || [ "$drv" -lt 580 ] 2>/dev/null; then
              echo "✘ trt-llm devshell requires NVIDIA driver r580+; got r''${drv:-<none>}" >&2
              echo "  Use vllm or sglang on this host instead." >&2
              exit 1
            fi
            ${findRoot}
            if [ -d "$_root/trt-llm/.venv" ]; then
              NV="$_root/trt-llm/.venv/lib/python3.12/site-packages/nvidia"
              for need in "$NV/cu13/lib/libcudart.so.13" \
                          "$NV/cu13/include/cublasLt.h" \
                          "${cu12Cudart}/lib/libcudart.so.12"; do
                if [ ! -e "$need" ]; then
                  echo "✘ trt-llm devshell: missing $need" >&2
                  echo "  cu13 venv libs come from \`uv sync\` in trt-llm/." >&2
                  echo "  cu12 runtime libs come from nixpkgs cudaPackages." >&2
                  return 1
                fi
              done
              ${cu13SymlinkLoop}
              export NIX_LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:${cu12Cudart}/lib:${lib.makeLibraryPath (with pkgs; [ openmpi zeromq ])}:${runtimeLibs}:${driverLib}"
              export LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
              export CPATH="$NV/cu13/include:$NV/cudnn/include:${cudaToolkit}/include:''${CPATH:-}"
              echo "✓ trt-llm: cu13 (jit/torch) + cu12 (cudart) verified" >&2
            else
              echo "ℹ trt-llm/.venv missing under $_root — run \`uv sync\` to materialize cu13 libs" >&2
            fi
            unset _root NV
          '';
        };

        default = self.devShells.${system}.vllm;
      };
    };
}
