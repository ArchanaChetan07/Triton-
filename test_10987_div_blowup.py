"""Regression test for triton-lang/triton#10987.

A chained fp32 division-by-broadcast feeding a reduction must not be lowered in a
replicated layout. Before the fix, kernel B emitted 4128 `div.full.f32` (vs 64
for the equivalent kernel C that lacks the trailing reduce). After the fix, B is
expected to emit ~64.

Pure ahead-of-time compilation — no GPU launch required. Run with either:
    pytest test_10987_div_blowup.py
    python  test_10987_div_blowup.py
"""
import os
import tempfile

import triton
import triton.language as tl
from triton.compiler import ASTSource

try:
    from triton.backends.compiler import GPUTarget
except Exception:  # pragma: no cover
    from triton.compiler.compiler import GPUTarget

RDIM, BLOCK = 256, 16
# Pre-fix B was 4128. Post-fix should match C (~64). Guard well below the bug.
THRESHOLD = 128

BODIES = {
    "A_single_div_reduce":
        "    y = x / s\n"
        "    z = tl.reshape(tl.sum(y, 1), [S, 1]) + y\n",
    "B_chained_div_reduce":
        "    y = x / s\n"
        "    w = y / s\n"
        "    z = tl.reshape(tl.sum(w, 1), [S, 1]) + w\n",
    "C_chained_div_noreduce":
        "    y = x / s\n"
        "    w = y / s\n"
        "    z = w + x\n",
    "D_reciprocal":
        "    r = 1.0 / s\n"
        "    y = x * r\n"
        "    w = y * r\n"
        "    z = tl.reshape(tl.sum(w, 1), [S, 1]) + w\n",
}


def compile_div_count(body, cap=90):
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="t10987_")
    d = tempfile.mkdtemp(prefix="t10987_src_")
    src = (
        "import triton, triton.language as tl\n"
        "@triton.jit\n"
        "def k(a, out, S: tl.constexpr):\n"
        "    i0 = tl.arange(0, S)[:, None]\n"
        f"    i1 = tl.arange(0, {RDIM})[None, :]\n"
        f"    x = tl.load(a + i0 * {RDIM} + i1)\n"
        "    s = tl.reshape(tl.sum(x, 1), [S, 1])\n"
        f"{body}"
        f"    tl.store(out + i0 * {RDIM} + i1, z)\n"
    )
    path = os.path.join(d, "k.py")
    open(path, "w").write(src)
    ns = {}
    exec(compile(src, path, "exec"), ns)
    sig = {"a": "*fp32", "out": "*fp32", "S": "constexpr"}
    try:
        source = ASTSource(fn=ns["k"], signature=sig, constexprs={"S": BLOCK})
    except TypeError:
        source = ASTSource(fn=ns["k"], signature=sig, constants={"S": BLOCK})
    cc = triton.compile(source, target=GPUTarget("cuda", cap, 32),
                        options={"num_warps": 4})
    return cc.asm["ptx"].count("div.full.f32")


def test_chained_div_reduce_is_not_replicated():
    """The #10987 guard: chained div->reduce must not blow up the div count."""
    n = compile_div_count(BODIES["B_chained_div_reduce"])
    assert n < THRESHOLD, (
        f"kernel B emitted {n} div.full.f32 (expected <{THRESHOLD}, ~64). "
        f"Replicated-layout regression, see triton-lang/triton#10987."
    )


if __name__ == "__main__":
    print(f"triton {triton.__version__}  (threshold for B: <{THRESHOLD})\n")
    for name, body in BODIES.items():
        n = compile_div_count(body)
        flag = ""
        if name.startswith("B"):
            flag = "  <-- PASS" if n < THRESHOLD else "  <-- FAIL (#10987)"
        print(f"  {name:24s}: div.full.f32 = {n:5d}{flag}")
