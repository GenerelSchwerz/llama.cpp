# Defect register: KV offload and CUDA FlashAttention

Every entry was re-confirmed against **`beellama/dev` at `946c1e5b6`** on a
**fresh build tree**, or is explicitly marked as not confirmed. Nothing here is
carried forward on trust.

Each entry gives the exact command that reproduces it, the exact output that
constitutes the failure, and who owns the fix. If an entry cannot be reproduced
from what is written here, that is a bug in this document.

## Validation state

| # | Defect | Status | Re-confirmed how |
|---|---|---|---|
| D1 | D320 routed to a missing vector kernel | **confirmed** | `docs/repro/d1-vector-d320.sh`, exit 134 |
| D2 | Divergent `__syncthreads()` | **confirmed, localized, reported upstream** | `docs/repro/d2-barrier.sh` on a pristine `-lineinfo` build; independent upstream-only reproduction is [ggml-org/llama.cpp#27678](https://github.com/ggml-org/llama.cpp/issues/27678) |
| D3 | MTP + host KV aborts | **confirmed; cause open** | `docs/repro/d3-mtp-abort.sh`, 1 abort in 3 runs, twice independently; correlation with D2 is not a causal finding |
| D4 | `test-kvarn` on default builds | **confirmed; not planned** | `docs/repro/d4-test-kvarn.sh`; see R7 disposition |
| D5 | KVarN host-resident workspace | **confirmed**, both arms | re-measured on a fresh build once D7 was patched: 2,195.90 MiB vs 20.28 MiB, matching the carried-forward figures exactly; only the context-scaling table is from an earlier session |
| D6 | Quantized × `logit_softcap` | **not reproduced** | `docs/repro/d6-softcap.sh`; coverage is one case wide |
| D7 | `GGML_CUDA_KVARN=ON` does not compile | **fixed by #34; pending merge and validation on `dev`** | historical base failure: 82 errors; PR #34 contains the verified two-line repair |
| D8 | `q2_0s` FlashAttention returns wrong values at `hsk=64` | **confirmed** | `docs/repro/d8-q2-fa64.sh`; 1 FAIL in 2,673 cases, `ERR ~= 1.7`, at three seeds |
| D9 | KVarN is refused for a device-resident cache on a default build | **confirmed** | `docs/repro/d9-kvarn-device.sh`; context creation fails at `-ngl 99` |
| D10 | `test-upstream-merge-keepers-static` fails on a clean checkout | **confirmed** | `docs/repro/d10-agents-static.sh`; fails in 0.03 s, no GPU |

**Historical D7 blocker.** On the recorded base, the KVarN build did not compile,
so D5 could not be regenerated without the two-line probe patch. PR #34 now
contains that repair; until it merges and is validated on `beellama/dev`, this
register keeps the original failure and its base commit as evidence rather than
claiming that current `dev` has already changed.

## Reference environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB, compute capability 8.9 |
| Driver / CUDA | 610.57.04 / 13.3 (`nvcc` V13.3.73) |
| CPU / RAM | Intel Core i5-13400F, 31 GiB |
| OS | Linux 7.1.6, CachyOS |
| Model | `Qwen3.8-27B-UD-IQ2_M.gguf`, 9.60 GiB, 65 layers, 16 full-attention |
| Source | `946c1e5b6` |

Two build trees are referenced below:

```bash
# default tree, used unless stated otherwise
cmake -B build-clean -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-clean --parallel

# same, plus line information, for sanitizer source locations
cmake -B build-li  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build build-li --parallel
```

Adapt `CMAKE_CUDA_ARCHITECTURES` to your GPU. Everything below assumes a
discrete NVIDIA GPU; the integrated case differs and is called out where it
matters.

---

## D1. Head dimension 320 routed to a vector kernel that does not exist

**Status: CONFIRMED on `946c1e5b6`.** **Owner: BeeLlama** (vector routing).
Severity: aborts the process. Blocks the documented full-suite validation.

### Reproduce

```bash
bash docs/repro/d1-vector-d320.sh
```

### Failure

Exit 134. The last case printed before the abort is the one that aborts:

```
FLASH_ATTN_EXT(hsk=320,hsv=256,nh=1,nr23=[4,1],kv=512,nb=75,mask=1,sinks=1,
  max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,
  permute=[0,2,1,3] ... native_quants=0)
../../../../ggml/src/ggml-cuda/fattn.cu:412: fatal error
#4  ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context&, ggml_tensor*)
```

`fattn.cu:412` is the `GGML_ABORT("fatal error")` that terminates
`ggml_cuda_flash_attn_ext_vec` when `fattn-vec-dispatch.cuh` matched no
instance. No quantized cache is involved -- this is `f16`/`f16`.

### Notes

PR #4 recorded this with `nr23=[32,1], kv=512, nb=1` and described the confirmed
reproducer as GQA 32. On this run it aborts at **`nr23=[4,1], nb=75`**, so the
defect is reachable at a lower ratio than recorded and the earlier description
is narrower than the behaviour.

The case is `f16` K/V at `hsk=320, hsv=256`: `320 % 64 == 0` passes
`can_use_vector_kernel`, Ada's single-query preference selects the vector kernel
before MMA is considered, and no D320 vector instance exists.

---

## D2. Divergent `__syncthreads()` in upstream's MMA FlashAttention kernel

**Status: CONFIRMED and localized on `946c1e5b6`.** **Owner: upstream
llama.cpp.** Severity: undefined behaviour. It is independently reproduced on
current upstream and reported as [ggml-org/llama.cpp#27678](https://github.com/ggml-org/llama.cpp/issues/27678).
It co-occurs with D3, but that is not proof that it causes D3's illegal-address
abort.

### Reproduce

```bash
export LLAMA_KV_MODEL=/path/to/Qwen3.8-27B-UD-IQ2_M.gguf
LLAMA_KV_BUILD=build-li bash docs/repro/d2-barrier.sh    # ~40 min under the sanitizer
```

The script runs `compute-sanitizer --tool synccheck --show-backtrace device`
over `build-li/bin/llama-server` with a host-resident `q8_0` cache, `-c 8192`,
`--spec-type draft-mtp --spec-draft-n-max 5`, and a 900-token request.
`--show-backtrace device` is what adds the source location; without it the
report names only the kernel.

### Failure

100 barrier errors, **all at one site**:

```
Barrier error detected. Divergent thread(s) in block.
  at flash_attn_ext_f16_process_tile<256,256,4,8,4,0,0,0,1,F16,F16>
     +0x176b0 in fattn-mma-f16.cuh:1732
  Device Frame: flash_attn_ext_f16<256,256,4,8,0,0,F16,F16>
     +0x10640 in fattn-mma-f16.cuh:2104
```

Both fixup instantiations appear (`needs_fixup=1,is_fixup=0` and
`needs_fixup=0,is_fixup=1`) and no other site does.

**Reproduced twice on separate build trees**, once on a `-lineinfo` tree
carrying only the transport-telemetry patch and once on a **pristine
`-lineinfo` build of `946c1e5b6`** produced by `rm -rf build-li` and a fresh
configure. Both give 100 errors, the same single site, the same two
instantiations, and the same call frame at `:2104`. The second sanitized run
also aborts the request (`REQUEST_FAILED RemoteDisconnected`). That observation
is consistent with D3, but it is not a causal attribution.

### The code

`ggml/src/ggml-cuda/fattn-mma-f16.cuh`:

```c
if (np > 1 && threadIdx.y % np == 0) {   // 1687
    ...
    __syncthreads();                     // 1732  <-- reported
    ...
} else if (np > 1) {                     // 1764
    // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
    // Therefore, all other warps also need to execute a __syncthreads().
    __syncthreads();                     // 1768
}
```

For this instantiation `ncols = ncols1*ncols2 = 4*8 = 32`,
`cols_per_warp = T_B_KQ::I = 32`, `nwarps = 4`, so
`np = nwarps*cols_per_warp/ncols = 4`. With `np == nwarps`, only
`threadIdx.y == 0` takes the `if` and warps 1-3 take the `else`.

**Balancing the count is necessary but not sufficient.** `__syncthreads()` is
defined only where its guarding condition evaluates identically across the whole
block, and `threadIdx.y % np == 0` does not. Two textually distinct barriers in
complementary branches are not the same barrier in the CUDA model; they work in
practice because the compiler assigns both to the same physical barrier.

### Ownership

`git blame` puts `:1732` on `0208355f42` (2025-05-10) and `:1764-1768` on
`95e18884fc` (2025-05-12), both by upstream's CUDA maintainer. The fork owns
neither line. The `is_kvarn_kv` conditions nearby are fork additions and are not
on the reported path.

### What is not established

The same barrier pair compiles into every `np > 1` instantiation, including
shapes plain decode reaches, and plain decode does not abort. The causal chain
from this barrier to D3's `illegal memory access` is therefore **not** proven.
What is proven: a defect at a known line, in a kernel the aborting configuration
provably runs, that `memcheck` cannot see. Fixing it is the first thing to try.

PR #4 recorded the same divergence at `ncols1` = 1, 2 and 4 with
512 / 1024 / 1984 errors and `ncols1 = 8` clean. This run reproduces the
`ncols1 = 4` shape.

---

## D3. MTP with a host-resident KV cache aborts during sustained generation

**Status: CONFIRMED on `946c1e5b6`, 1 abort in 3 runs.** **Owner: upstream** if
D2 is the cause; **this fork** to confirm and to carry an interim guard.
Severity: kills the server process mid-request.

### Reproduce

```bash
export LLAMA_KV_MODEL=/path/to/Qwen3.8-27B-UD-IQ2_M.gguf
bash docs/repro/d3-mtp-abort.sh
```

Three server runs, each requesting 900 tokens against a host-resident `q8_0`
cache with `--spec-type draft-mtp --spec-draft-n-max 5`, greedy, `seed 1234`.

### Failure

```
run1: tg=42.630 n=900 draft=337/391      completed
run2: REQUEST_FAILED RemoteDisconnected  ABORTED
run3: tg=42.848 n=900 draft=337/391      completed
```

A second, independent execution of the packaged script reproduced it with the
same shape: runs 1 and 3 completed, run 2 aborted.

Run 2, at `n_decoded = 477`:

```
E CUDA error: an illegal memory access was encountered
E   current device: 0, in function ggml_backend_cuda_synchronize
E     at ggml/src/ggml-cuda/ggml-cuda.cu:2752
E   cudaStreamSynchronize(cuda_ctx->stream())
ggml/src/ggml-cuda/ggml-cuda.cu:109: CUDA error
```

### Notes

Runs 1 and 3 report identical draft statistics (`337/391`), so the run is
deterministic up to the abort -- consistent with a race rather than a
deterministic wrong path.

**The abort rate is not fixed.** Two passes of three runs each gave 1 abort
apiece; an earlier session on the same source measured 2 in 3. Anyone verifying
a fix should run more than three times before concluding anything; the
acceptance gate in the roadmap asks for 10.

The configuration is residency-specific. With the same model, cache type and
MTP settings but the KV cache device-resident, 900 tokens complete.

**This is the highest-value defect in the register.** Where it completes, MTP on
a host-resident cache is **+17.7%** at `-c 8192` and **+75.9%** on a
24,000-token prompt, the largest decode gains measured anywhere in this lane.

---

## D4. `test-kvarn` aborts on any default build with a CUDA device present

**Status: CONFIRMED on `946c1e5b6`; no implementation planned.** **Owner:
BeeLlama** (KVarN test). Severity: fails the documented validation command on a
stock build.

### Reproduce

```bash
bash docs/repro/d4-test-kvarn.sh
```

`build-clean` is configured **without** `GGML_CUDA_KVARN`, which is the default
(`CLAUDE.md`: "Fresh CMake caches omit KVarN").

### Failure

```
test-kvarn .... Subprocess aborted
ggml_cuda_graph_evaluate_and_capture: op not supported  (view) (KVARN_STORE)
ggml/src/ggml-cuda/ggml-cuda.cu:4425: GGML_ASSERT(ok) failed
0% tests passed, 1 tests failed out of 1
```

### Notes

The test exercises CUDA KVarN paths the default build does not compile. The
decision is not to extend default-build coverage for this path, because its CUDA
build-time cost is unacceptable; run it only in an explicitly enabled KVarN
build when that feature is being changed.

This is why `ctest -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"`, the
command `CLAUDE.md` documents for KVarN changes, cannot pass as written on a
default CUDA build.

---

## D5. KVarN and host-resident KV: silent CPU fallback with unbounded workspace

**Status: CONFIRMED on a fresh build, both arms; the context-scaling table is
carried forward.** **Owner: BeeLlama and this fork jointly -- unresolved.**
Severity: not a crash; a configuration that is silently many times more
expensive than the alternative.

**On the recorded base, this defect cannot be observed without first applying
[D7](#d7-ggml_cuda_kvarnon-does-not-compile)'s two-line repair**, since
`GGML_CUDA_KVARN=ON` did not compile. With that repair applied and a fresh build,
`kvarn5` at `-c 8192` host-resident reserves:

| Cache, `-c 8192` host-resident | CUDA0 compute buffer | **CUDA_Host compute buffer** |
|---|---:|---:|
| `kvarn5` | 122.01 MiB | **2,195.90 MiB** |
| `q8_0` | 142.27 MiB | **20.28 MiB** |

Both arms reproduce the carried-forward figures exactly: **108x more host
workspace** for the cache that is supposed to be the smaller one. The
context-scaling table below and the per-node breakdown are from an earlier
session and are labelled as such.

Note the device side moves the other way -- `kvarn5` reserves *less* on the GPU
(122.01 against 142.27 MiB) -- which is the tell that the work moved to the host
rather than being avoided.

### Reproduce

Requires a KVarN build, then `bash docs/repro/d5-kvarn-workspace.sh`:

```bash
cmake -B build-kvarn -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
      -DGGML_CUDA_KVARN=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-kvarn --parallel

GGML_SCHED_DEBUG=2 build-kvarn/bin/llama-server -m $MODEL -ngl 99 -sm none -mg 0 \
  -nkvo --kv-cpu-pinned --recurrent-state-offload -fa on \
  -ctk kvarn5 -ctv kvarn5 -c 8192 --parallel 1 -v
```

Compare the reported host compute buffer against the same command with
`-ctk q8_0 -ctv q8_0`.

### Behaviour

`ggml_backend_cuda_device_supports_buft` accepts a host buffer only on an
integrated GPU. On a discrete one it refuses, so every node touching the
host-resident KVarN records is placed on the CPU backend and host workspace is
reserved for their intermediates.

Per-node breakdown, from an earlier session with `GGML_SCHED_DEBUG=2`:

| Cache | CPU-assigned nodes | host compute buffer |
|---|---|---:|
| `q8_0` | 9 × `GET_ROWS` | 20.28 MiB |
| `kvarn5` | 576 `SET_ROWS`, 576 `KVARN_STOR`, 384 `MUL_MAT`, 384 `CONCAT`, 288 `KVARN_WHT` | **2,195.90 MiB** |

Scaling with context, `kvarn5`:

| Context | host workspace | host KVarN cache |
|---:|---:|---:|
| 8,192 | 2,195.90 MiB | — |
| 32,768 | 3,309.90 MiB | — |
| 65,536 | 6,541.90 MiB | — |
| 262,144 | 25,933.90 MiB | 5,528.00 MiB |

About **0.099 MiB per context token**, identical for `kvarn8`, `kvarn6` and
`kvarn5`, so it tracks context and not cache width. `--live-context-workspace`
does not bound it. At 262,144 the workspace is roughly 4.7x the cache it
compresses, and attention also runs on the CPU, which an earlier experiment
measured as needing about a 6.6x CPU-attention speedup merely to break even.

A standard quantized cache avoids this because its K/V reach attention as
copyable split inputs the scheduler stages to the device. KVarN's records reach
attention through `KVARN_VIEW` chains, so there is nothing to stage and the
scheduler falls back wholesale.

### Why no guard is proposed here

A fail-closed guard was written and discarded. It converts a slow configuration
into an error message, which is not a fix and gets in the way of the real one.
The integration is item 5 of the roadmap.

---

## D6. Quantized K/V with non-zero `logit_softcap`

**Status: NOT REPRODUCED on `946c1e5b6`.** Recorded so the claim is not carried
forward on trust.

PR #4 lists "quantized K/V + non-zero `logit_softcap` aborts the CUDA backend",
reproducing through the materializing path with `--flash-attn-native-quants`
withheld.

### What was tried

```bash
bash docs/repro/d6-softcap.sh
```

241 cases, **all pass**. Of those, exactly **one** has a quantized cache:

```
FLASH_ATTN_EXT(hsk=128,hsv=128,nh=4,nr23=[2,1],kv=256,nb=4,mask=1,sinks=0,
  max_bias=0.000000,logit_softcap=10.000000,prec=f32,type_K=q4_0,type_V=q4_0,...)
```

A model-level check was attempted with the only locally available softcap-family
model and could not run: `gemma4_26b-a4b-DSpark-Q8_0.gguf` fails to load on this
tree with `check_tensor_dims: tensor 'markov_w2.weight' has wrong shape`, which
is unrelated to this defect.

### Consequence

Either the defect was fixed, or it lives in a shape the suite does not cover.
`test-backend-ops` enumerates `logit_softcap != 0` only at `hsk == 128`
(`tests/test-backend-ops.cpp:10446`), and the quantized cross-product with
softcap is one case wide. **Widening that coverage is a roadmap item**, and
until it exists this entry can be neither confirmed nor closed.

---

## D7. `GGML_CUDA_KVARN=ON` does not compile

**Historical status: CONFIRMED on `946c1e5b6`. Current disposition: fixed by
[PR #34](https://github.com/GenerelSchwerz/llama.cpp/pull/34), pending merge and
validation on `beellama/dev`.** **Owner: BeeLlama.** The failure below remains
the exact base-state evidence; it must not be presented as a current-`dev`
failure after #34 lands.

### Reproduce

```bash
bash docs/repro/d7-kvarn-build.sh
```

### Failure

82 compile errors, all the same:

```
ggml/src/ggml-cuda/template-instances/../fattn-mma-kvarn-case.cuh(178):
  error: no instance of function template "flash_attn_ext_f16_process_tile"
         matches the argument list
```

### Cause

`flash_attn_ext_f16_process_tile` gained a parameter,
`int32_t * causal_prefix_first_bound_smem`, immediately after
`compact_causal_prefix` (`fattn-mma-f16.cuh:1250-1258`). The non-KVarN caller
passes it:

```c
// fattn-mma-f16.cuh:2105
(Q_f2, K_h2, V_h2, mask_h, compact_causal_prefix, &causal_prefix_first_bound_smem,
 sinks_f, dstk, ...)
```

Both KVarN call sites still pass the old list:

```c
// fattn-mma-kvarn-case.cuh:178 and :245
(Q_f2, K_h2, V_h2, mask_h, false, sinks_f, dstk, ...)
```

The parameter was added in `0a85856ee` (2026-08-23, *"cuda: preserve compact
causal bound across MMA phases"*). `git blame` attributes that commit **and**
both stale call sites to the same author, so the signature change and the call
sites it broke are owned by the same layer.

### Fix

Two lines, one per call site: insert `nullptr` for the new parameter.
[`probes/06-kvarn-build-fix.patch`](probes/06-kvarn-build-fix.patch) applies
cleanly to `946c1e5b6`.

**Verified:** with the patch applied, `cmake --build build-kvarn` completes with
exit 0 and zero errors; without it, 82. PR #34 carries that two-line repair.
After its merge, re-run this build on the resulting `beellama/dev` commit and
record that commit as the resolution validation.

### Note on reproducibility across machines

This defect was previously reported as not occurring on a colleague's machine.
It is a template argument-count mismatch, so it cannot depend on the toolchain:
any build that actually compiles `fattn-mma-kvarn-case.cuh` fails. A machine
that does not see it is not building with `GGML_CUDA_KVARN=ON`, or is not on
`946c1e5b6`.

---

## D8. `q2_0s` FlashAttention returns wrong values at head dimension 64

**Status: CONFIRMED on `946c1e5b6`.** **Owner: BeeLlama** (`q2_0s` is the fork's
own retained 32-element q2 type, presented as `q2_0` at the cache CLI boundary,
`common/arg.cpp:334-338`). Severity: **silently wrong output**. This is the only
entry in this register that produces a wrong answer rather than a loud failure.

### Reproduce

```bash
bash docs/repro/d8-q2-fa64.sh      # no model, no server, about a minute
```

### Failure

One case out of the 2,673 the suite generates at `hsk=64` fails, and the error
is not a tolerance margin:

```
[FLASH_ATTN_EXT] ERR = 1.703930844 > 0.000500000
  FLASH_ATTN_EXT(hsk=64,hsv=64,nh=4,nr23=[2,1],kv=256,nb=2,mask=1,sinks=0,
    max_bias=0.000000,logit_softcap=0.000000,prec=f32,
    type_K=q2_0s,type_V=q2_0s,permute=[0,1,2,3] ... native_quants=0): FAIL
selected_cases=2673
  Backend CUDA0: FAIL
```

Deterministic, and the magnitude says the result is uncorrelated with the
reference rather than imprecise:

| `--seed` | ERR |
|---|---:|
| 0 | 1.748040631 |
| 1 | 2.112992357 |
| 42 | 1.707560517 |

### How narrow it is

At the **same shape** (`hsk=64, kv=256, nb=2`), every other pair the suite
generates passes:

```
q2_0s q2_0s => FAIL
q2_1  q2_1  => OK      q3_0 q3_0 => OK      q3_1 q3_1 => OK
q4_0  q4_0  => OK      q6_0 q6_0 => OK      q6_1 q6_1 => OK
```

And `q2_0s` itself passes everywhere else that is covered: all 24 `hsk=64`
cases at `nb=3, 16, 32, 64` over `kv=113/512` and `nr23=[1,1] .. [16,1]`, the
`sinks`, `max_bias`, `prec=def` and mixed-V variants at `nb=16`, and every case
at `hsk=40, 72, 80, 96, 112, 128, 192, 256, 576`. `nb=2` at `hsk=64` is the
single shape that breaks.

### What is not established

Two things, and neither is guessed at here:

- **Which side is wrong.** `test-backend-ops` compares CUDA against the CPU
  backend. A wrong `q2_0s` CPU reference at this shape would produce the same
  report. Dumping both outputs is the first step of the fix, not of this
  register.
- **Whether it reaches a real model.** It needs a model with 64-wide K/V heads
  and a 2-token ubatch. Speculative decoding produces 2-token batches routinely,
  so the shape is not exotic, but no model-level reproduction was attempted --
  the reference model here has 128-wide heads.

Both are stated as open rather than assumed, in either direction.

---

## D9. KVarN is refused for a device-resident cache on a default build

**Status: CONFIRMED on `946c1e5b6`.** **Owner: BeeLlama.** On that base, D7
meant KVarN had no GPU-capable build at all. PR #34 repairs D7 but remains
pending merge and validation on `beellama/dev`.

### Reproduce

```bash
export LLAMA_KV_MODEL=/path/to/Qwen3.8-27B-UD-IQ2_M.gguf
bash docs/repro/d9-kvarn-device.sh
```

### Failure

```
--- -ngl 99
llama_init_from_model: cannot enable kvarn_k4v4_g128: KVarN requires a backend
  with KVarN store and materialization support
llama_bench: error: failed to create context with model '...IQ2_M.gguf'
```

It is not a fallback -- `fail_if_unsupported` is set, so context creation
returns null and the process exits.

The configurations that do initialize are exactly those where the cache does not
live on the device:

| Arm | KVarN | tg4 |
|---|---|---:|
| `-ngl 99` | **refused, context creation fails** | — |
| `-ngl 0` | accepted | 1.41 t/s |
| `-ngl 99 -nkvo 1` | accepted | 13.86 t/s |
| `-ngl 0`, `q8_0/q8_0`, for comparison | n/a | 1.42 t/s |

### Cause

`llama_context` ANDs `llama_kvarn_backend_supports_ops()` over every KV layer,
passing `model->dev_layer(il)` when `offload_kqv` is set and `nullptr` when it
is not (`src/llama-context.cpp:4596-4607`). That function returns `true` for
`nullptr` -- "the built-in CPU backend implements store + materialize" -- and
otherwise asks the device's backend registry for `ggml_backend_kvarn_ops`
(`src/llama-kv-cache-kvarn.cpp:260-285`). A default build never exports it,
because `ggml/src/ggml-cuda/CMakeLists.txt:117-119` filters `kvarn.cu` and
`kvarn-wht.cu` out of the source list when `GGML_CUDA_KVARN` is off.

So the predicate is doing its job; what is wrong is the shape of the policy
around it. `-nkvo` returning `true` through the `nullptr` branch is why
[D5](#d5-kvarn-and-host-resident-kv-silent-cpu-fallback-with-unbounded-workspace)'s slow
CPU arm is reachable while the fast device arm is not.

### What this corrects

`CLAUDE.md` and `AGENTS.md` both say that KVarN's fast-decode pair matrix is
what `GGML_CUDA_KVARN` selects, and that "every valid KVarN bit pair remains
available through descriptor-native MMA when it is outside the fast matrix".
On a default build no KVarN bit pair is available on the device at all. On the
recorded base, together with D7, this meant **no user could run KVarN on a GPU
by any route**. After PR #34 merges, this statement needs a fresh enabled-KVarN
build validation rather than assumption.

`-ngl 1` also initializes, which looks like a counterexample and is not: the
model has 65 KV layers and `load_tensors: offloaded 1/66 layers to GPU` places
the non-KV output layer there, leaving every KV layer on the CPU device.

---

## D10. `test-upstream-merge-keepers-static` fails on a clean checkout

**Status: CONFIRMED on `946c1e5b6`.** **Owner: BeeLlama.** Severity: a red test
on a clean tree, which trains everyone to ignore a red test.

### Reproduce

```bash
bash docs/repro/d10-agents-static.sh    # 0.03 s, no GPU, no model
```

### Failure

```
1/1 Test #18: test-upstream-merge-keepers-static ...***Failed    0.03 sec
AssertionError: AGENTS.md does not describe the v0.4.0 CUDA policy
```

### Cause

`tests/test-upstream-merge-keepers-static.py:256` requires the literal string
`50 standard vector pairs` in `AGENTS.md`. The text is there, but `775450a68`
(*"build: make KVarN compilation explicitly opt-in"*) rewrapped the paragraph and
the phrase now straddles a line break:

```
AGENTS.md:54: The minimal fresh-cache CUDA FlashAttention build contains 50 standard vector
AGENTS.md:55: pairs and omits KVarN. ...
```

### Fix

One line, either side: rewrap `AGENTS.md:54-55` so the phrase is contiguous, or
normalize whitespace in `require()` before the substring check. The second is
the better fix -- it is a documentation-content test, and it should not fail on
a reflow -- but it changes the behaviour of every other `require()` call in the
file, so it is a decision rather than a typo correction.

---

## Checks that came back clean

Recorded so the same ground is not covered twice. All on `946c1e5b6`,
`build-clean`, same reference environment.

| Check | Result |
|---|---|
| All 169 ordered `-ctk`/`-ctv` pairs over the 13 CLI cache types, `-fa 1`, `llama-bench` | 169/169 exit 0, no abort, no fallback warning |
| All 16 default-tier pairs with `--flash-attn-native-quants` (`q8_0, q6_0, q5_0, q4_0` squared) | 16/16 run, none emits the "no kernel for" decline |
| All 13 homogeneous cache types with `-nkvo 1`, with and without `--flash-attn-native-quants` | 26/26 run |
| `test-backend-ops test` over all 136 non-`FLASH_ATTN_EXT` ops on CUDA0 | `Backend CUDA0: OK`, 0 failures |
| `FLASH_ATTN_EXT` at `hsk=40, 72, 80, 96, 112, 128, 192, 256, 576` | OK; the only failures anywhere are D1 (`hsk=320`) and D8 (`hsk=64`) |
| CPU-only build (`-DGGML_CUDA=OFF`), full compile | exit 0 |
| Configure with `FA_ALL_QUANTS=ON`, `KVARN=ON`, both, and `GGML_CUDA_FA=OFF` | all four configure with exit 0; on the recorded base only the `KVARN=ON` **build** fails (D7) |
| `ctest` over 92 tests (downloads and `test-backend-ops` excluded) | 3 real failures: D4, D10, and `test-tokenizers-ggml-vocabs` |

`test-tokenizers-ggml-vocabs` is **not** a fork defect: it clones
`models/ggml-vocabs` and this host has no `git-lfs`, so the vocab files are LFS
pointer text and `gguf_init_from_reader` reports `invalid magic characters:
'vers'`. Upstream test, environment cause. The three `test-save-load-state*`
failures seen in the same run were an artefact of excluding their download
fixture and are not defects either.
