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
| D2 | Divergent `__syncthreads()` | **confirmed, localized** | `synccheck` on a `-lineinfo` build; 100/100 errors at one line — see the provenance note in D2 |
| D3 | MTP + host KV aborts | **confirmed** | `docs/repro/d3-mtp-abort.sh`, 1 abort in 3 runs, twice independently |
| D4 | `test-kvarn` on default builds | **confirmed** | `docs/repro/d4-test-kvarn.sh` |
| D5 | KVarN host-resident workspace | **carried forward, not re-run** | needs a `GGML_CUDA_KVARN=ON` tree; script provided, numbers are from an earlier session |
| D6 | Quantized × `logit_softcap` | **not reproduced** | `docs/repro/d6-softcap.sh`; coverage is one case wide |

D5 is the only entry whose numbers were not regenerated in this pass, and it is
labelled as such wherever it appears. Everything else in this file was produced
by running the script named next to it against a fresh build of `946c1e5b6`.

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
llama.cpp.** Severity: undefined behaviour; the observed failure mode is an
`illegal memory access` abort (see D3) and, per upstream's own reasoning,
silently wrong output is possible.

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

> **Provenance.** This run was taken on a `-lineinfo` tree whose only difference
> from `946c1e5b6` was the transport-telemetry patch
> ([`probes/01`](probes/01-transport-telemetry.patch) and
> [`02`](probes/02-transport-telemetry-d2h.patch)), which touches only
> `ggml-cuda.cu` and not `fattn-mma-f16.cuh`, and is inert unless
> `GGML_KV_TRANSPORT_STATS` is set. The kernel under test is therefore
> bit-identical to a pristine build. A re-run on a fully pristine `-lineinfo`
> tree is the one verification step this document does not yet carry; the source
> lines, the blame and the reasoning below do not depend on it.

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

**Status: CONFIRMED on `946c1e5b6`.** **Owner: BeeLlama** (KVarN test).
Severity: fails the documented validation command on a stock build.

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

The test exercises CUDA KVarN paths the default build does not compile. It
should skip them when `GGML_CUDA_KVARN` is off, the way the four static KVarN
tests already pass on the same build.

This is why `ctest -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"`, the
command `CLAUDE.md` documents for KVarN changes, cannot pass as written on a
default CUDA build.

---

## D5. KVarN and host-resident KV: silent CPU fallback with unbounded workspace

**Status: measured previously, NOT re-confirmed on a fresh `GGML_CUDA_KVARN=ON`
build in this pass.** **Owner: BeeLlama and this fork jointly -- unresolved.**
Severity: not a crash; a configuration that is silently many times more
expensive than the alternative.

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
