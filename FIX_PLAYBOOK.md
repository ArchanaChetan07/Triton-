# #10987 fix — build, verify, and submit playbook

Everything here is done except **building and testing on a real GPU box**, which
needs a Linux machine with an NVIDIA GPU (your rented 4090). Follow this end to
end.

## 1. What the fix is

**File:** `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`,
function `LayoutPropagation::resolveConflicts()`.

`resolveConflicts` picks one layout when an op has several candidates. It only
preferred a *blocked* layout for load/store ops; for anything else (e.g.
`arith.divf`) it kept the **arbitrary first candidate**, which could be a
replicated `#linear` layout (lane/warp bases all zero → every thread
materializes the whole tensor). An elementwise op in that layout is emitted once
per redundant copy.

The patch breaks the tie toward the candidate with the **fewest elements per
thread** (`getUniqueElemsPerThread`) — the most distributed layout — so the
expensive `divf` chain stays in `#blocked`.

- Branch (local, this machine): `fix/10987-elementwise-replicated-layout`
- Portable patch: `10987-fix.patch` (git-am format, applies to fresh `main`)

## 2. Verified numbers (from the AOT repro, Triton 3.7.1)

| Kernel | `div.full.f32` before | expected after |
|---|---:|---:|
| A single div→reduce | 32 | 32 |
| **B chained div→reduce** | **4128** | **~64** |
| C chained, no reduce | 64 | 64 |
| D reciprocal | 16 | 16 |

Consistency check that pins the theory: `4128 = 32 (blocked) + 4096 (linear)`.
After the fix B's second div is also `#blocked` (32), so B → ~64, matching C.

## 3. Rent + prep the box

- Any Ubuntu 22.04/24.04 box with one RTX 4090, CUDA 12.x, Python 3.10–3.12.
- Copy this whole folder to the box (`scp -r triton-10987-repro user@box:~/`).
- `cd ~/triton-10987-repro && sed -i 's/\r$//' build_and_test_4090.sh` (CRLF).

## 4. Build + verify

**Do the baseline first — don't skip it.** The bug was reproduced/analyzed on
Triton **3.7.1**; the patch targets **main**. Confirm main still shows the bug
*before* applying the patch, or a passing test could be vacuous.

**Vast.ai / vLLM image gotchas** (hit and fixed 2026-07-28 on RTX 5090):
- Activate `/venv/main` (or your app venv). Those venvs often set
  `include-system-site-packages = true`, so a system `triton==3.6.0` next to
  torch **shadows** the editable build. Always:
  `export PYTHONPATH=$HOME/triton/python` (or `/workspace/triton/python`)
  and confirm `python -c "import triton; print(triton.__file__)"` points at
  the clone before trusting numbers.
- Prefer `git apply` over `git am` unless you've set a *local* committer
  identity in the clone (`git config user.email` / `user.name`, no `--global`).
- Use `10987-fix-final.patch` (includes the lit test). After `git apply`,
  copy the new `.mlir` into `build/.../test/TritonGPU/` before running lit, and
  put Triton's downloaded `FileCheck` (`~/.triton/llvm/*/bin`) on `PATH`.
  Or just run `bash build_and_test_4090.sh` which handles all of this.

```
# 0. clone main, DO NOT apply the patch yet
git clone https://github.com/triton-lang/triton.git ~/triton && cd ~/triton
python3 -m pip install cmake ninja pybind11
MAX_JOBS=$(nproc) python3 -m pip install -e .          # 20-40 min (first build)
export PYTHONPATH=$PWD/python                          # required on vast.ai
python3 ~/triton-10987-repro/test_10987_div_blowup.py  # EXPECT: B ~4128 -> test FAILS (bug present on main)

# 1. apply the fix and REBUILD (incremental — only 1 file changed, a few minutes)
git apply ~/triton-10987-repro/10987-fix-final.patch
MAX_JOBS=$(nproc) python3 -m pip install -e .
export PYTHONPATH=$PWD/python
python3 ~/triton-10987-repro/test_10987_div_blowup.py  # EXPECT: B <128 -> test PASSES
```

If the baseline (step 0) already shows B small (bug absent on main), stop and
tell me — main has diverged from 3.7.1 and the fix needs re-basing on the
current behavior.

`build_and_test_4090.sh` automates the *patched* build+test (step 1 only). Use
the two-step above the first time so you see the before/after yourself.

**PASS = `B_chained_div_reduce : div.full.f32` is under 128 (was ~4128).**
If B is still ~4128 after the patched rebuild, the fix didn't fire — see §7.

To confirm the layout actually changed, dump the TTGIR and check the second
`divf` is no longer `#linear`:
```
python3 dump_ttgir.py | grep -nE "divf|#linear|#blocked"
```
Before: `%w = arith.divf ... : tensor<16x256xf32, #linear>`.
After:  the divf feeding the reduce should be `#blocked`.

## 5. Guard against regressions (important before a PR)

Build puts `triton-opt` under `python/build/**/bin` (or install `lit`:
`pip install lit`). Run the layout-pass lit suites:
```
lit -v test/TritonGPU/combine.mlir \
       test/TritonGPU/remove-layout-conversions-disable-remat.mlir \
       test/TritonGPU/remove-layout-conversions-backward-remat-reuse.mlir
```
All must pass. Also run a couple of end-to-end suites if you have time:
`python3 -m pytest python/test/unit/language/test_core.py -k "reduce or layout" -q`.

## 6. Finalize the lit unit test for the PR

The maintainers will want an MLIR lit test. Generate it on the box (you can't
author it reliably without running the pass):

1. Dump the module **before** the pass runs:
   ```
   MLIR_ENABLE_DUMP=1 python3 dump_ttgir.py 2> dump.mlir
   ```
   In `dump.mlir`, find the `IR Dump Before ...RemoveLayoutConversions` section
   for kernel `k` — that block is your test input.
2. Create `test/TritonGPU/remove-layout-conversions-10987.mlir`:
   - First line:
     `// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s`
   - Paste the pre-pass module.
   - Add checks asserting the divf feeding the reduce is distributed, e.g.:
     `// CHECK-NOT: arith.divf {{.*}}#linear`
     `// CHECK: arith.divf {{.*}}#blocked`
3. Confirm it passes: `lit -v test/TritonGPU/remove-layout-conversions-10987.mlir`
4. `git add` it and `git commit --amend` (or a second commit) on the branch.

## 7. If B is still ~4128 (fix didn't fire)

The layout may be decided somewhere other than `resolveConflicts` for this
shape. In order, check:
1. Dump before/after each pass (`MLIR_ENABLE_DUMP=1`) and find the first pass
   where `%w` becomes `#linear`. If it's already `#linear` going *into*
   remove-layout-conversions, the choice is upstream (reduce/reshape layout
   inference in `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`).
2. Alternative fix point: `isLayoutAnchor` (same file, ~line 236) marks permuting
   reshapes as anchors — the reshape of the reduction result is what introduces
   `#linear`. Not anchoring it (or anchoring it to a distributed layout) is a
   second candidate fix.
3. Backward pass: `hoistConvertOnTopOfExtOrBroadcast` / `backwardRematerialization`
   may be re-introducing the convert; the cost model there
   (`isRematBeneficial`, `getCostFactor`) already uses elems-per-thread, so a
   tweak there is a third option.

## 8. Submit the PR

```
# from this Windows machine or the box, on the branch:
git remote add fork https://github.com/ArchanaChetan07/triton.git   # fork first on GitHub
git push fork fix/10987-elementwise-replicated-layout
```
Open the PR against `triton-lang/triton:main`. Title/body: reuse `ROOT_CAUSE.md`
and the numbers table. Reference `Fixes #10987`. Mention it's verified on
RTX 4090 (sm_89) and that A/C/D are unchanged.

## Honesty note
The patch is a well-reasoned first candidate, not yet compiled or run. Expect
one or two iterations on the box (a missing include, or one of the §7 fallbacks).
That iteration is normal compiler-PR work — the repro, root cause, and test
harness here are what make it fast.
