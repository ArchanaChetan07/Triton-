#!/usr/bin/env bash
# Build + verify the triton-lang/triton#10987 fix on a fresh Linux GPU box
# (e.g. vast.ai RTX 4090/5090). Keep this file next to the patch + repro scripts.
#
# If edited on Windows, strip CR first:  sed -i 's/\r$//' build_and_test_4090.sh
# Usage:  bash build_and_test_4090.sh [TRITON_DIR] [PATCH_FILE]
#   TRITON_DIR   default ~/triton  (cloned fresh if absent)
#   PATCH_FILE   default ./10987-fix-final.patch (falls back to 10987-fix.patch)
#
# Gaps fixed on vast.ai / vLLM images (2026-07-28):
# 1. Venvs with include-system-site-packages=true shadow the editable build with
#    the system triton wheel — force PYTHONPATH=<repo>/python after install.
# 2. Prefer `git apply` (no committer identity needed). `git am` needs a local
#    user.name/email if you want a real commit.
# 3. lit needs the *build* lit.site.cfg.py + FileCheck from Triton's downloaded
#    LLVM, not the source-tree lit.cfg.py alone.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:-$HOME/triton}"
if [ -n "${2:-}" ]; then
  PATCH="$2"
elif [ -f "$HERE/10987-fix-final.patch" ]; then
  PATCH="$HERE/10987-fix-final.patch"
else
  PATCH="$HERE/10987-fix.patch"
fi

# Prefer an activated venv's python if present; else python3.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
  elif [ -x /venv/main/bin/python ]; then
    # vast.ai default app env
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
    PYTHON=/venv/main/bin/python
  else
    PYTHON=python3
  fi
fi

echo "==== environment ===="
nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 1; }
"$PYTHON" --version; git --version

echo "==== source (${REPO}) ===="
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/triton-lang/triton.git "$REPO"
fi
cd "$REPO"

if [ -f "$PATCH" ]; then
  if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"; echo "applied $PATCH"
  elif git am --check "$PATCH" 2>/dev/null; then
    # Only works if committer identity is configured (local or global).
    git am "$PATCH" && echo "applied via git am $PATCH"
  else
    echo "NOTE: $PATCH did not apply cleanly (already applied, or main moved)."
  fi
fi

echo "==== build deps ===="
# Fresh GPU boxes often lack a C++ toolchain / Python headers. Install if missing
# (needs sudo, which rented boxes have). Triton downloads its own LLVM.
if ! command -v g++ >/dev/null 2>&1 || ! "$PYTHON" -c "import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path('include'),'Python.h')) else 1)"; then
  echo "installing build-essential + python3-dev ..."
  sudo apt-get update -y && sudo apt-get install -y build-essential python3-dev zlib1g-dev \
    || echo "WARN: apt install failed; ensure g++ and python3-dev are present"
fi
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q cmake ninja pybind11 lit

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
echo "==== build triton (editable, MAX_JOBS=$MAX_JOBS) — 20-40 min first time ===="
"$PYTHON" -m pip install -e . 2>&1 | tail -25

# CRITICAL: on images where the venv includes system site-packages, a preinstalled
# triton wheel (e.g. 3.6.0 next to torch) wins over the editable finder. Pin the
# just-built tree first on sys.path.
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c "import triton; print('built triton', triton.__version__, 'from', triton.__file__); assert 'triton/python' in triton.__file__.replace('\\\\','/') or triton.__file__.startswith('$REPO')" || {
  echo "ERROR: import is not the editable build — set PYTHONPATH=$REPO/python"
  "$PYTHON" -c "import triton,sys; print(triton.__file__); print(sys.path)"
  exit 1
}

echo "==== regression test (#10987) ===="
# Expected: kernel B < 128 (it was 4128 before the fix, ~64 after).
"$PYTHON" "$HERE/test_10987_div_blowup.py"
echo "---- full A/B/C/D table ----"
"$PYTHON" "$HERE/triton_aot_repro.py"

echo "==== unit tests for the pass (no regressions) ===="
TRITON_BIN=$(find "$REPO/build" "$REPO/python/build" -name triton-opt -type f 2>/dev/null | head -1)
LLVM_BIN=$(find "$HOME/.triton/llvm" -path '*/bin/FileCheck' -type f 2>/dev/null | head -1)
BUILD_TEST=""
if [ -n "$TRITON_BIN" ]; then
  BUILD_ROOT=$(cd "$(dirname "$TRITON_BIN")/.." && pwd)
  BUILD_TEST="$BUILD_ROOT/test"
  export PATH="$(dirname "$TRITON_BIN"):$PATH"
fi
if [ -n "$LLVM_BIN" ]; then
  export PATH="$(dirname "$LLVM_BIN"):$PATH"
fi
# New lit file from git apply may not be in the cmake test tree yet.
if [ -f "$REPO/test/TritonGPU/remove-layout-conversions-10987.mlir" ] && [ -n "$BUILD_TEST" ]; then
  mkdir -p "$BUILD_TEST/TritonGPU"
  cp -f "$REPO/test/TritonGPU/remove-layout-conversions-10987.mlir" \
        "$BUILD_TEST/TritonGPU/remove-layout-conversions-10987.mlir"
fi
if [ -n "$BUILD_TEST" ] && [ -f "$BUILD_TEST/lit.site.cfg.py" ]; then
  lit -v \
    "$BUILD_TEST/TritonGPU/combine.mlir" \
    "$BUILD_TEST/TritonGPU/remove-layout-conversions-disable-remat.mlir" \
    "$BUILD_TEST/TritonGPU/remove-layout-conversions-backward-remat-reuse.mlir" \
    "$BUILD_TEST/TritonGPU/remove-layout-conversions-scf-cleanup.mlir" \
    "$BUILD_TEST/TritonGPU/remove-layout-conversions-10987.mlir" \
    || echo "WARN: some lit tests failed"
else
  echo "  (skip lit: need build/.../test/lit.site.cfg.py + FileCheck on PATH)"
fi

echo "==== DONE ===="
echo "PASS criterion: regression test above prints B ... < 128 and exits 0."
