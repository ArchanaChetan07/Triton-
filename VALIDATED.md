# #10987 fix — EMPIRICALLY VALIDATED ✅

Re-verified 2026-07-28 on rented **vast.ai RTX 5090 / CUDA 13 / Triton main
3.8.0** (commit `7842017`), after fixing validation gaps on the vLLM image.

## Before / after (real build, sm_90)

| Kernel | Unpatched main | Patched main |
|---|---:|---:|
| A — single div → reduce | 32 | 32 |
| **B — chained div → reduce** | **4,128** | **64** ✅ |
| C — chained, no reduce | 64 | 64 |
| D — reciprocal | 16 | 16 |

Kernel B: **4,128 → 64** `div.full.f32` (matches kernel C). Compile time for B
dropped **257s → 4s**. TTGIR after the fix: both `arith.divf` stay `#blocked`;
`convert_layout` to `#linear` happens only for the trailing reduce.

## Regression check (patched `triton-opt`)

| Test | Result |
|---|---|
| `test/TritonGPU/combine.mlir` | PASS |
| `remove-layout-conversions-disable-remat.mlir` | PASS |
| `remove-layout-conversions-backward-remat-reuse.mlir` | PASS |
| `remove-layout-conversions-scf-cleanup.mlir` | PASS |
| new `remove-layout-conversions-10987.mlir` | PASS |

No regressions.

## Gaps fixed in this repro kit (vast.ai / vLLM images)

1. **Wrong Triton import** — venvs with `include-system-site-packages=true` let
   the system `triton==3.6.0` (torch pin) win over the editable 3.8.0 build.
   `build_and_test_4090.sh` now forces `PYTHONPATH=<repo>/python` and asserts
   the import path.
2. **`git am` needs identity** — script prefers `git apply` (no committer
   config). Set a *local* `user.name`/`user.email` in the clone if you want
   `git am` to create a commit.
3. **lit harness** — must use the build tree's `lit.site.cfg.py` plus
   `FileCheck` from `~/.triton/llvm/*/bin`, and copy the new `.mlir` into
   `build/.../test/TritonGPU/` after a plain `git apply`.

## The deliverable

`10987-fix-final.patch` — a `git am`-able patch (fix + regression test) against
current `main`, commit message ends with `Fixes #10987`. This is what goes in
the PR.

## To open the PR
```bash
# fork triton-lang/triton on GitHub first, then:
git clone https://github.com/<you>/triton.git && cd triton
git checkout -b fix/10987-elementwise-replicated-layout
# local identity only (do not use --global on shared boxes):
git config user.email "you@example.com" && git config user.name "Your Name"
git am /path/to/10987-fix-final.patch
git push origin fix/10987-elementwise-replicated-layout
# open the PR against triton-lang/triton:main, referencing #10987
```
