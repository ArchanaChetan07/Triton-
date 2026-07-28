# triton-lang/triton#10987 — root-cause analysis

## Symptom
Kernel B (`y = x/s; w = y/s; sum(w,1)`) emits 4,128 `div.full.f32` / ~24k PTX
lines / ~25s compile, vs ~32 divs / ~1.8k lines for the near-identical kernel A.

## Where it explodes (bisected by IR stage, kernel B)
| Stage | lines | `arith.divf` | `div.full.f32` |
|---|---:|---:|---:|
| TTIR  | 86    | 2 | 0 |
| TTGIR | 88    | 2 | 0 |
| LLIR  | 22,441| 0 | 0 |
| PTX   | 24,018| 0 | **4,128** |

Division count is **2** all the way through TTGIR, then explodes only when
`ConvertTritonGPUToLLVM` materializes per element. So the divisions themselves
are fine — the *layout* the second division is computed in is not.

## Root cause: a replicated `#linear` layout
TTGIR (trimmed):
```mlir
#blocked = #ttg.blocked<{sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[1,4], order=[1,0]}>
#linear  = #ttg.linear<{register=[[1,0],[2,0],[4,0],[8,0]],
                        lane=[[0,0],[0,0],[0,0],[0,0],[0,0]],   // <-- all zero
                        warp=[[0,0],[0,0]],                     // <-- all zero
                        block=[]}>

%s     = tt.reduce(%x_7) axis=1 : ... -> slice<dim1, #blocked>
%s_8   = tt.reshape %s   : slice<dim1,#blocked> -> tensor<16x1xf32, #linear>   // #linear is BORN here
%y_11  = arith.divf %x_7, %y_9  : tensor<16x256xf32, #blocked>                 // 1st div: distributed (fine)
%y_12  = ttg.convert_layout %y_11 : #blocked -> #linear
%w     = arith.divf %y_12, %y_10 : tensor<16x256xf32, #linear>                 // 2nd div: REPLICATED (blows up)
%z     = tt.reduce(%w) axis=1 : tensor<16x256xf32, #linear> -> ...
```

In Triton's linear-layout encoding, a basis of `[0,0]` means that hardware axis
(lane / warp) does **not** move you within the tensor — i.e. the data is
*replicated* across it. Here **every** `lane` basis (5) and **every** `warp`
basis (2) is `[0,0]`, so the `[16,256]` tensor `w` is replicated across all
32 lanes × 4 warps. Materialising an element-wise `divf` in that layout emits it
~128× more than a distributed layout would → 4,128 `div.full.f32`.

The first division stays in `#blocked` (distributed) and contributes the normal
~32. The second division is pushed into `#linear` by a `convert_layout` so that
it lines up with the second `tt.reduce`, which took its input layout from the
`#linear` produced by the reshape of the first reduction.

## Why the workarounds behave as observed
- **C (no trailing reduce):** nothing forces the `#linear` layout, `w` stays
  `#blocked`, no blowup.
- **D (`r = 1/s; x*r*r`):** the division is done once on the small `[16,1]`
  reduced tensor (16 elements) before broadcasting, so only 16 `div.full.f32`;
  the replicated multiplies are cheap and don't show up as divisions.

## Fix direction (to investigate)
The expensive element-wise op should not be computed in the replicated `#linear`
layout. Candidate angles:
1. Layout inference for `tt.reshape` of a reduction result / reduce input layout
   selection — avoid emitting an all-zero-lane/warp `#linear` for a full 2D tile.
2. A layout-optimization heuristic (RemoveLayoutConversions / reduce-input layout)
   that keeps `w` in the distributed `#blocked` layout and only converts to the
   reduce's required layout *after* the elementwise chain, not before.

Relevant source (Triton main @ 9d93126):
- `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp` (reduce input layout)
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` (reshape/reduce layouts)
- `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` (propagation)

Testing any fix requires building Triton from source (C++/MLIR) — best done on a
full Linux + datacenter-GPU box.
