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

      # Two CUDA majors coexist in each engine venv:
      #   cu13 — torch wheel + flashinfer JIT (headers under nvidia/cu13/
      #          include must match what flashinfer compiles against).
      #   cu12 — vllm._C.abi3.so DT_NEEDED libcudart.so.12; supplied from
      #          nixpkgs. *Runtime only* — never on LIBRARY_PATH, or the
      #          JIT linker's `-lcudart` would pick cu12 over cu13 and
      #          break ABI compat with torch.
      #
      # Pin the exact 12.9 minor: this matches the cu12 versions the
      # h100/h200 venvs bundle (cudart 12.9.79, cublas 12.9.1.4, nvrtc
      # 12.9.86 — see uv.lock). nixpkgs `cudaPackages` and `cudaPackages_12`
      # are both currently aliases for 12.9 but could float to a different
      # minor; `cudaPackages_12_9` won't.
      cuda12 = pkgs.cudaPackages_12_9;

      # nvcc + cccl headers, no libs. flashinfer's run_ninja hard-codes
      # `-L${CUDA_HOME}/lib64 -lcudart`; an empty lib/ here ensures that
      # falls through to LIBRARY_PATH where cu13 (from the venv) wins.
      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-${cuda12.cuda_nvcc.version}-nvcc-only";
        paths = [ cuda12.cuda_nvcc cuda12.cuda_cccl ];
      };

      # cu12 cudart for vllm._C's libcudart.so.12. cuda_cudart is single-
      # output, so this path holds both lib/ and include/ (shim below
      # reads include/cudaTypedefs.h from it).
      cu12Cudart = cuda12.cuda_cudart;

      # cu12 libs sglang's bundled sgl_kernel dlopens at runtime that the
      # torch wheel does NOT bundle: libnvrtc.so.12, libcublas.so.12,
      # libcublasLt.so.12. Other cu12 libs sgl_kernel/torch reference
      # (cufft/cusparse/cusolver/curand/cupti) are already shipped inside
      # the venv's nvidia/cu13/lib/ — pulling those from nixpkgs would be
      # a no-op since the venv path is first on NIX_LD_LIBRARY_PATH.
      # Kept outside CUDA_HOME so the JIT linker can't pick cu12 over cu13
      # for `-lcudart`/`-lcublas`.
      cu12Extras = pkgs.symlinkJoin {
        name = "cu12-extras";
        paths = [ cuda12.cuda_nvrtc.lib cuda12.libcublas.lib ];
      };

      # cu13 dropped the unversioned `PFN_X` macro aliases (now only
      # `PFN_X_v12000`); flashinfer's bundled cutlass references the
      # unversioned forms. Shim cudaTypedefs.h: `#include_next`s cu13's
      # real header then adds back the cu12 macro aliases. Put FIRST on
      # CPATH so it shadows only that one header. vllm + sglang only.
      cu13TypedefShim = pkgs.runCommand "cu13-typedef-shim" { } ''
        mkdir -p $out/include
        {
          echo '#pragma once'
          echo '#include_next <cudaTypedefs.h>'
          # Guard each alias so a future cu13 that re-adds them won't redefine.
          grep -E '^#define PFN_cu' ${cu12Cudart}/include/cudaTypedefs.h \
            | awk '{print "#ifndef " $2 "\n" $0 "\n#endif"}'
        } > $out/include/cudaTypedefs.h
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

      # Runtime libs the manylinux wheels dlopen. Verified by surveying
      # DT_NEEDs across all three venvs: libstdc++/libgcc_s, libz, libssl/
      # libcrypto, liblzma. (libffi/glib/ncurses had zero hits — uv-managed
      # python-build-standalone statically links libffi, and no wheel here
      # links glib or ncurses.)
      runtimeLibs = lib.makeLibraryPath (with pkgs; [
        stdenv.cc.cc.lib zlib openssl xz
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
    in {
      devShells.${system} = {
        # ─────────── vllm ────────────────────────────────────────────────────
        # cu13 venv libs only (torch + flashinfer JIT). Nothing in the
        # vllm venv DT_NEEDs libcudart.so.12, so no nixpkgs cu12 supply.
        # flashinfer JIT needs cu13TypedefShim for the cutlass headers
        # that still reference unversioned PFN_X macro aliases.
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
                          "$NV/cu13/include/cublasLt.h"; do
                if [ ! -e "$need" ]; then
                  echo "✘ vllm devshell: missing $need (run \`uv sync\` in vllm/)" >&2
                  return 1
                fi
              done
              # NIX_LD_LIBRARY_PATH: cu13 venv ▸ cudnn ▸ nccl ▸ runtime ▸ driver.
              export NIX_LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:${runtimeLibs}:${driverLib}"
              # LIBRARY_PATH (JIT linker `-l*`): cu13 venv only. Driver lib
              # comes from the Nix attr above; we prepend cu13 here.
              export LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
              # CPATH: typedef shim FIRST (its cudaTypedefs.h adds back the
              # unversioned PFN_* aliases cu13 dropped — flashinfer's mamba
              # tensormap header needs them; #include_next falls through
              # to cu13's real header), then cu13 venv, cudnn, then
              # CUDA_HOME's include (`<nv/target>` etc).
              export CPATH="${cu13TypedefShim}/include:$NV/cu13/include:$NV/cudnn/include:${cudaToolkit}/include:''${CPATH:-}"
              echo "✓ vllm: cu13 venv libs verified" >&2
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
                  echo "  cu12 runtime libs come from nixpkgs cudaPackages_12_9." >&2
                  return 1
                fi
              done
              export NIX_LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:${cu12Cudart}/lib:${cu12Extras}/lib:${lib.makeLibraryPath (with pkgs; [ numactl openmpi czmq zeromq ])}:${runtimeLibs}:${driverLib}"
              export LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
              export CPATH="${cu13TypedefShim}/include:$NV/cu13/include:$NV/cudnn/include:${cudaToolkit}/include:''${CPATH:-}"
              echo "✓ sglang: cu13 venv + cu12 (cudart, nvrtc, cublas) verified" >&2
            else
              echo "ℹ sglang/.venv missing under $_root — run \`uv sync\` to materialize cu13 libs" >&2
            fi
            unset _root NV
          '';
        };

        # ─────────── trt-llm ─────────────────────────────────────────────────
        # cu13 venv libs + cu12 cudart (torchao's _C.abi3.so still
        # DT_NEEDs libcudart.so.12 even on the cu13 trt-llm stack).
        # No cu13TypedefShim (no flashinfer JIT) and no cu12Extras
        # (nothing here DT_NEEDs cu12 nvrtc/cublas). openmpi + zeromq
        # for the MPI executor pool.
        #
        # tensorrt-llm 1.3 cu13 stack references CUDA Driver API symbols
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
                  echo "  cu12 cudart comes from nixpkgs cudaPackages_12_9." >&2
                  return 1
                fi
              done
              export NIX_LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:${cu12Cudart}/lib:${lib.makeLibraryPath (with pkgs; [ openmpi zeromq ])}:${runtimeLibs}:${driverLib}"
              export LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib:''${LIBRARY_PATH:-}"
              export CPATH="$NV/cu13/include:$NV/cudnn/include:${cudaToolkit}/include:''${CPATH:-}"
              echo "✓ trt-llm: cu13 venv + cu12 cudart verified" >&2
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
