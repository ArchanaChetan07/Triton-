"""
Torch-free, ahead-of-time reproduction of triton-lang/triton #10987.

The bug is pure codegen (TTGIR -> LLVM lowering), so we do NOT need torch or a
kernel launch: we compile each kernel with triton.compile(ASTSource(...)) and
inspect the emitted LLVM IR / PTX directly.

Kernels A/B/C/D are logically identical to the upstream repro. We compile each
for sm_90 (to compare against the H100 report's numbers) and sm_75 (local T1000).
"""

import os
import time
import tempfile

import triton
import triton.language as tl
from triton.compiler import ASTSource

try:
    from triton.backends.compiler import GPUTarget
except Exception:  # pragma: no cover - older layout
    from triton.compiler.compiler import GPUTarget  # type: ignore

RDIM = 256
BLOCK = 16
NUM_WARPS = 4

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="trit10987_")
_SRC_DIR = tempfile.mkdtemp(prefix="trit10987_src_")
_COUNTER = [0]

KERNELS = {
    "A_single_div_reduce": (
        "    y = x / s\n"
        "    z = tl.reshape(tl.sum(y, 1), [S, 1]) + y\n"
    ),
    "B_chained_div_reduce": (
        "    y = x / s\n"
        "    w = y / s\n"
        "    z = tl.reshape(tl.sum(w, 1), [S, 1]) + w\n"
    ),
    "C_chained_div_noreduce": (
        "    y = x / s\n"
        "    w = y / s\n"
        "    z = w + x\n"
    ),
    "D_chained_div_reduce_reciprocal": (
        "    r = 1.0 / s\n"
        "    y = x * r\n"
        "    w = y * r\n"
        "    z = tl.reshape(tl.sum(w, 1), [S, 1]) + w\n"
    ),
}


def make_kernel(body_src):
    src = (
        "import triton, triton.language as tl\n"
        "@triton.jit\n"
        "def k(a, out, S: tl.constexpr):\n"
        "    i0 = tl.arange(0, S)[:, None]\n"
        f"    i1 = tl.arange(0, {RDIM})[None, :]\n"
        f"    x = tl.load(a + i0 * {RDIM} + i1)\n"
        "    s = tl.reshape(tl.sum(x, 1), [S, 1])\n"
        f"{body_src}"
        f"    tl.store(out + i0 * {RDIM} + i1, z)\n"
    )
    _COUNTER[0] += 1
    path = os.path.join(_SRC_DIR, f"kmod_{_COUNTER[0]}.py")
    with open(path, "w") as f:
        f.write(src)
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns["k"]


def make_source(k):
    """Build an ASTSource across triton API variants (constexprs vs constants)."""
    sig = {"a": "*fp32", "out": "*fp32", "S": "constexpr"}
    errs = []
    for kwargs in (
        {"signature": sig, "constexprs": {"S": BLOCK}},
        {"signature": sig, "constants": {"S": BLOCK}},
        {"signature": sig},
    ):
        try:
            return ASTSource(fn=k, **kwargs)
        except TypeError as e:
            errs.append(repr(e))
    raise RuntimeError("ASTSource construction failed: " + " | ".join(errs))


def compile_one(name, body, cap):
    k = make_kernel(body)
    src = make_source(k)
    target = GPUTarget("cuda", cap, 32)
    opts = {"num_warps": NUM_WARPS}
    t0 = time.time()
    try:
        cc = triton.compile(src, target=target, options=opts)
    except TypeError:
        cc = triton.compile(src, target=target)
    dt = time.time() - t0

    asm = cc.asm
    ptx = asm.get("ptx", "")
    llir = asm.get("llir", "")
    ptx_lines = ptx.count("\n") + 1 if ptx else 0
    llir_lines = llir.count("\n") + 1 if llir else 0
    ndiv_ptx = ptx.count("div.full.f32")
    nfdiv_llir = llir.count("fdiv")
    print(f"  {name:32s}: compile {dt:6.2f}s  llir_lines={llir_lines:6d}  "
          f"ptx_lines={ptx_lines:6d}  div.full.f32={ndiv_ptx:5d}  fdiv(llir)={nfdiv_llir}")
    return dt, ptx_lines, ndiv_ptx


def main():
    print(f"triton {triton.__version__}")
    print(f"tile=[{BLOCK}, {RDIM}], num_warps={NUM_WARPS}\n")
    for cap, label in ((90, "sm_90  (matches H100 report)"), (75, "sm_75  (local T1000)")):
        print(f"===== target {label} =====")
        for name, body in KERNELS.items():
            try:
                compile_one(name, body, cap)
            except Exception as e:
                print(f"  {name:32s}: FAILED -> {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
