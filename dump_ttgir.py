"""Print TTGIR for kernel B so we can read w's layout (source of the blowup)."""
import os, tempfile
import triton, triton.language as tl
from triton.compiler import ASTSource
try:
    from triton.backends.compiler import GPUTarget
except Exception:
    from triton.compiler.compiler import GPUTarget

RDIM, BLOCK = 256, 16
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="ttgir_")
D = tempfile.mkdtemp(prefix="ttgir_src_")
src_txt = (
    "import triton, triton.language as tl\n"
    "@triton.jit\n"
    "def k(a, out, S: tl.constexpr):\n"
    "    i0 = tl.arange(0, S)[:, None]\n"
    f"    i1 = tl.arange(0, {RDIM})[None, :]\n"
    f"    x = tl.load(a + i0 * {RDIM} + i1)\n"
    "    s = tl.reshape(tl.sum(x, 1), [S, 1])\n"
    "    y = x / s\n"
    "    w = y / s\n"
    "    z = tl.reshape(tl.sum(w, 1), [S, 1]) + w\n"
    f"    tl.store(out + i0 * {RDIM} + i1, z)\n"
)
kpath = os.path.join(D, "kB.py")
open(kpath, "w").write(src_txt)
ns = {}
exec(compile(src_txt, kpath, "exec"), ns)
sig = {"a": "*fp32", "out": "*fp32", "S": "constexpr"}
try:
    source = ASTSource(fn=ns["k"], signature=sig, constexprs={"S": BLOCK})
except TypeError:
    source = ASTSource(fn=ns["k"], signature=sig, constants={"S": BLOCK})
cc = triton.compile(source, target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
print("========== TTGIR (kernel B) ==========")
print(cc.asm["ttgir"])
