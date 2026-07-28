# Ready-to-post claim comment for triton-lang/triton#10987

> **Before you post:** you must actually understand and be able to defend this —
> read `ROOT_CAUSE.md`, and re-run the repro yourself once so the claim is
> genuinely yours:
>
> ```
> wsl python3 /mnt/c/Users/archa/OneDrive/Desktop/OpenAI/triton-10987-repro/triton_aot_repro.py
> wsl python3 /mnt/c/Users/archa/OneDrive/Desktop/OpenAI/triton-10987-repro/dump_ttgir.py
> ```
>
> Then post the text below on https://github.com/triton-lang/triton/issues/10987

---

Reproduced on **Triton 3.7.1** (pip, Linux), compiling the four kernels
ahead-of-time and counting `div.full.f32` in the PTX — still present, and the B
case is a bit worse than reported:

| Kernel | ptx lines | `div.full.f32` |
|---|---:|---:|
| A — single `x / s` → reduce | 1,777 | 32 |
| **B — chained `x / s / s` → reduce** | **~24,000** | **4,128** |
| C — chained, no reduce | 1,206 | 64 |
| D — reciprocal `x * r * r` → reduce | 13,695 | 16 |

Bisecting by IR stage, the division count stays at **2** through TTIR and TTGIR
and only explodes in `ConvertTritonGPUToLLVM` — so it's a **layout** problem, not
a division-lowering one. In the TTGIR the second division is computed in a
degenerate `#linear` layout:

```mlir
#linear = #ttg.linear<{register=[[1,0],[2,0],[4,0],[8,0]],
                       lane=[[0,0],[0,0],[0,0],[0,0],[0,0]],   // all zero
                       warp=[[0,0],[0,0]], block=[]}>

%y_11 = arith.divf %x_7, %y_9  : tensor<16x256xf32, #blocked>   // 1st div: distributed  -> ~32
%y_12 = ttg.convert_layout %y_11 : #blocked -> #linear
%w    = arith.divf %y_12, %y_10 : tensor<16x256xf32, #linear>   // 2nd div: replicated   -> blows up
```

Every `lane` and `warp` basis of that `#linear` is `[0,0]`, i.e. the `[16,256]`
tensor is **replicated across all 32 lanes × 4 warps**, so materialising the
element-wise `divf` in it emits it ~128× more than the distributed `#blocked`
layout does. The `#linear` layout originates at the `tt.reshape` of the first
reduction's result and is propagated forward into the second division to line up
with the trailing `tt.reduce`. That also explains C (nothing forces `#linear`,
no blowup) and D (the division happens once on the small `[16,1]` reduced tensor
before broadcast).

I'd like to take this if nobody's on it. The direction I want to try: stop the
expensive element-wise chain from being pushed into the replicated `#linear`
layout — keep `w` in `#blocked` and only convert to the reduce's required layout
after the elementwise ops (or fix the reshape/reduce layout inference so it
doesn't select an all-zero-lane/warp `#linear` for a full 2D tile). Will follow
up with a minimal IR diff and a fix PR. Standalone repro + dumped `.llir`/`.ptx`
available.
