import triton, triton.language as tl
@triton.jit
def k(a, out, S: tl.constexpr):
    i0 = tl.arange(0, S)[:, None]
    i1 = tl.arange(0, 256)[None, :]
    x = tl.load(a + i0 * 256 + i1)
    s = tl.reshape(tl.sum(x, 1), [S, 1])
    y = x / s
    w = y / s
    z = tl.reshape(tl.sum(w, 1), [S, 1]) + w
    tl.store(out + i0 * 256 + i1, z)
