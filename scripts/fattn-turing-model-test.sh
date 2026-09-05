#!/usr/bin/env bash
#
# Quantized-native CUDA FlashAttention on Turing: real-model test.
#
# Run this AFTER test-backend-ops has passed. That proves the kernels are
# correct and that the dispatcher picks the right path. This one answers the
# next two questions:
#
#     Does a real model still produce sane text with the route active?
#     Is it any faster on Turing?
#
# Run it from the repository root, after building:
#     scripts/fattn-turing-model-test.sh -m /path/to/model.gguf
#     scripts/fattn-turing-model-test.sh -m model.gguf --ab   # + real speedup number
#
# THE MODEL MATTERS. The route only covers head dim 256 and 512. A normal
# head-dim-128 model never touches it and the run proves nothing. The script
# reads the geometry out of the model and tells you which case you are in
# before it measures anything.
#
# Known-good family: Gemma 2 / Gemma 3, any size. Head dim 256, GQA 2, and
# widely available. Use it with a context of 1024 or less (see below).
#
# Send back the whole results directory.

set -u -o pipefail

MODEL=""
BUILD=""
OUT="$PWD/fattn-turing-model-results"
CTX=0
DEEP=0
CTK="q4_0"
NGL=999
NPREDICT=128
REPS=3
DEVICE_ENV=""
DO_AB=0
FORCE=0

usage() { sed -n '3,24p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'

Options:
  -m, --model FILE   GGUF model (required).
  --build DIR        llama.cpp build directory (default: ./build, then ./build-turing).
  --out DIR          Results directory.
  -c, --ctx N        Context size (default: chosen to keep the route engaged).
  --cache-type T     KV cache type: q4_0 (default) or q8_0.
  --deep             Measure in the long-context regime (n_kv >= 16384) instead
                     of the short one. q4_0 and a GQA>4 model only.
  --ngl N            Layers on GPU (default 999 = all).
  -n N               Tokens to generate in the correctness run (default 128).
  -r N               llama-bench repetitions (default 3).
  --gpu N            Restrict to one GPU via CUDA_VISIBLE_DEVICES.
  --ab               Also build a route-disabled copy and report the real delta.
  --force            Measure even if the model cannot reach the route.
  -h, --help         This text.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)   MODEL="$2"; shift 2 ;;
        --build)      BUILD="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        -c|--ctx)     CTX="$2"; shift 2 ;;
        --cache-type) CTK="$2"; shift 2 ;;
        --ngl)        NGL="$2"; shift 2 ;;
        -n)           NPREDICT="$2"; shift 2 ;;
        -r)           REPS="$2"; shift 2 ;;
        --gpu)        DEVICE_ENV="$2"; shift 2 ;;
        --ab)         DO_AB=1; shift ;;
        --deep)       DEEP=1; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
good() { printf '   \033[32m%s\033[0m\n' "$*"; }
warnp(){ printf '   \033[33m%s\033[0m\n' "$*"; }
badp() { printf '   \033[31m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mfatal:\033[0m %s\n' "$*" >&2; exit 1; }

[ -n "$MODEL" ] || { usage >&2; die "no model given (-m /path/to/model.gguf)"; }
[ -f "$MODEL" ] || die "model not found: $MODEL"

case "$CTK" in q4_0|q8_0) ;; *) die "--cache-type must be q4_0 or q8_0 (others need GGML_CUDA_FA_ALL_QUANTS)";; esac

if [ -z "$BUILD" ]; then
    for d in build build-turing build-turing-test; do
        [ -x "$d/bin/llama-bench" ] && { BUILD="$d"; break; }
    done
fi
[ -n "$BUILD" ] || die "no build directory found; pass --build DIR"
CLI="$BUILD/bin/llama-cli"
BENCH="$BUILD/bin/llama-bench"
[ -x "$CLI" ]   || die "llama-cli not found at $CLI"
[ -x "$BENCH" ] || die "llama-bench not found at $BENCH"

mkdir -p "$OUT" || die "cannot create $OUT"
OUT="$(cd "$OUT" && pwd)"
[ -n "$DEVICE_ENV" ] && export CUDA_VISIBLE_DEVICES="$DEVICE_ENV"

# ------------------------------------------------------------------ context --

step "Environment"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | sed 's/^/   GPU /'
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q '7\.5' \
    && good "Turing card present" \
    || warnp "no compute 7.5 device visible - this will not test the Turing route"
info "build      : $BUILD"
info "model      : $MODEL"
info "kv cache   : $CTK"
[ "$CTX" -eq 0 ] && info "context    : auto (picked to keep the route engaged)" \
                 || info "context    : $CTX"

# ------------------------------------------------------- 1. model geometry --

step "1/4  Model geometry and route eligibility"

PROMPT="Explain in three sentences why the sky looks blue during the day and red near sunset."

# The geometry only prints under -v, and under -v llama-cli interleaves the
# generated tokens with log lines on the same output lines. So this is a
# separate, deliberately tiny run; the readable text comes from a clean one.
info "reading the model geometry"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$CLI" -m "$MODEL" -c 512 -ngl "$NGL" -fa on \
    -ctk "$CTK" -ctv "$CTK" -n 1 -p "hi" --seed 1 -st -v \
    > "$OUT/geometry.log" 2>&1 \
    || { tail -20 "$OUT/geometry.log"; die "the model would not load; see $OUT/geometry.log"; }

# print_info: lines carry the geometry. n_head/n_head_kv/n_gqa may be printed
# per layer for models with mixed attention, so take the first number.
first_num() { sed -n "s/.*: *$1 *= *\[\?\([0-9]\+\).*/\1/p" "$OUT/geometry.log" | head -1; }
D_K="$(first_num n_embd_head_k)"
D_V="$(first_num n_embd_head_v)"
GQA="$(first_num n_gqa)"
NAME="$(sed -n 's/.*: general\.name *= *\(.*\)/\1/p' "$OUT/geometry.log" | head -1)"

if [ -z "${D_K:-}" ]; then
    tail -25 "$OUT/geometry.log"
    die "could not read the model geometry; see $OUT/geometry.log"
fi

info "model name    : ${NAME:-unknown}"
info "n_embd_head_k : $D_K"
info "n_embd_head_v : ${D_V:-?}"
info "gqa ratio     : ${GQA:-?}"

# Mirror of ggml_cuda_fattn_native_supported() and _profitable() in fattn.cu.
# Device-resident cache assumed, which is what -ngl 999 gives.
# ROUTE_MAX_KV is the top of the short window, ROUTE_DEEP_MIN the bottom of the
# long one; 0 means that window does not exist for this combination.
ROUTE="no"; ROUTE_WHY=""; ROUTE_MAX_KV=0; ROUTE_DEEP_MIN=0
if [ "$D_K" != "$D_V" ]; then
    ROUTE_WHY="K and V head dims differ; the route needs them equal"
elif [ "$D_K" = "512" ]; then
    if [ "${GQA:-0}" -gt 4 ]; then
        ROUTE="yes"; ROUTE_WHY="D=512 row, any KV length"; ROUTE_MAX_KV=1000000
    else
        ROUTE_WHY="D=512 needs a GQA ratio above 4, this model has ${GQA:-?}"
    fi
elif [ "$D_K" = "256" ]; then
    if [ "${GQA:-0}" = "2" ]; then
        ROUTE="yes"; ROUTE_WHY="D=256 GQA-2 row, prefill batches above 16 tokens"
        ROUTE_MAX_KV=1024
    elif [ "${GQA:-0}" = "8" ]; then
        ROUTE_WHY="D=256 with GQA 8 is deliberately excluded (open memory-safety question)"
    elif [ "${GQA:-0}" -gt 4 ]; then
        ROUTE="yes"; ROUTE_WHY="D=256 GQA-$GQA row, prefill batches above 4 tokens"
        if [ "$CTK" = "q4_0" ]; then
            ROUTE_MAX_KV=1024; ROUTE_DEEP_MIN=16384
        else
            ROUTE_MAX_KV=512
        fi
    else
        ROUTE_WHY="D=256 needs a GQA ratio of 2 or above 4, this model has ${GQA:-?}"
    fi
else
    ROUTE_WHY="head dim $D_K is not covered; the route only handles 256 and 512"
fi

echo
if [ "$ROUTE" = "yes" ]; then
    good "this model CAN take the native route: $ROUTE_WHY"

    # The route is only taken inside a KV-length window, so pick a context and
    # bench depths that sit inside it. Measuring outside it times the unchanged
    # F16 path and looks like "no speedup".
    PP=512
    if [ "$DEEP" = 1 ]; then
        if [ "$ROUTE_DEEP_MIN" -eq 0 ]; then
            die "--deep needs the long window, which this model and cache type do not have (try --cache-type q4_0)"
        fi
        BENCH_DEPTHS="16384"
        [ "$CTX" -eq 0 ] && CTX=$((16384 + 4096))
        info "long-context regime: the route is active from n_kv >= $ROUTE_DEEP_MIN"
    else
        if [ "$ROUTE_MAX_KV" -ge 1024 ]; then
            BENCH_DEPTHS="0,512"
            [ "$CTX" -eq 0 ] && CTX=1024
        else
            # q8_0 at GQA>4 only reaches n_kv <= 512, so pp512 at depth 0 is all
            # that fits.
            BENCH_DEPTHS="0"
            [ "$CTX" -eq 0 ] && CTX=512
            info "this cache type only reaches n_kv <= $ROUTE_MAX_KV; q4_0 has a wider window"
        fi
        info "short-context regime: the route is active up to n_kv = $ROUTE_MAX_KV"
        if [ "$ROUTE_DEEP_MIN" -gt 0 ]; then
            info "there is also a long-context window from n_kv >= $ROUTE_DEEP_MIN: use --deep"
        fi
    fi
    info "using context $CTX, benching at depth(s) $BENCH_DEPTHS"
    info "note: token generation (batch of 1) never takes the route; only prefill does"
else
    badp "this model can NOT take the native route"
    info "reason: $ROUTE_WHY"
    info ""
    info "Measuring would time the unchanged F16 path and prove nothing, so"
    info "this stops here. Use a head-dim-256 model instead - Gemma 2 or"
    info "Gemma 3, any size - or pass --force to measure anyway."
    [ "$FORCE" = 1 ] || exit 1
    warnp "--force given, continuing on a model the route cannot reach"
    PP=512; BENCH_DEPTHS="0"; [ "$CTX" -eq 0 ] && CTX=1024
fi

# ----------------------------------------------------------- 2. is it sane --

step "2/4  Output with a $CTK KV cache"

# Without -v the log is the banner, the prompt echo and the completion, so the
# tail of it is the text. Drop the spinner and the timing footer.
extract_text() {
    # llama-cli echoes the prompt back on a "> " line; the completion follows.
    awk -v anchor="${PROMPT:0:30}" 'index($0, anchor) {f=1} f' "$1" \
        | grep -vE '^\s*$|^\[ (Prompt|Load)|Exiting\.\.\.' | head -25
}

generate() {
    local ctype="$1" out="$2"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$CLI" -m "$MODEL" -c "$CTX" -ngl "$NGL" -fa on \
        -ctk "$ctype" -ctv "$ctype" -n "$NPREDICT" -p "$PROMPT" \
        --seed 1 -st > "$out" 2>&1
}

info "generating with the $CTK cache (this is the run the route acts on)"
if generate "$CTK" "$OUT/generate-quant.log"; then
    extract_text "$OUT/generate-quant.log" > "$OUT/text-quant.txt"
    sed 's/^/   | /' "$OUT/text-quant.txt"
    echo
    warnp "read that: it has to be coherent prose, not repetition or garbage"
else
    tail -25 "$OUT/generate-quant.log"
    badp "generation FAILED - see $OUT/generate-quant.log"
    die "the route produces no usable output on this card"
fi

info ""
info "generating the same prompt with an f16 cache for comparison"
if generate f16 "$OUT/generate-f16.log"; then
    extract_text "$OUT/generate-f16.log" > "$OUT/text-f16.txt"
    echo
    sed 's/^/   > /' "$OUT/text-f16.txt"
    echo
    info "the two will not match token for token - a quantized cache is lossy."
    info "they should be comparable in quality. Wild divergence is a red flag."
else
    warnp "f16 reference run failed - see $OUT/generate-f16.log"
fi

# ------------------------------------------------------------ 3. throughput --

step "3/4  Throughput"

# pp exercises the route (large prefill batch), tg cannot (batch of 1), so tg
# doubles as a control: it should not move between the two builds.
TG=64

run_bench() {
    local bin="$1" tag="$2"
    info "running llama-bench ($tag) - a few minutes"
    local args=(-m "$MODEL" -ngl "$NGL" -fa 1 -ctk "$CTK" -ctv "$CTK" -r "$REPS" -o json
                -p "$PP" -n "$TG" -d "$BENCH_DEPTHS")
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "$bin" "${args[@]}" > "$OUT/bench-$tag.json" 2> "$OUT/bench-$tag.log"
    return $?
}

if run_bench "$BENCH" on; then
    python3 - "$OUT/bench-on.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
print("   %-22s %12s" % ("test", "t/s"))
for r in rows:
    if r.get("n_prompt", 0):
        name = "pp%d" % r["n_prompt"]
    else:
        name = "tg%d" % r.get("n_gen", 0)
    if r.get("n_depth", 0):
        name += " @ d%d" % r["n_depth"]
    print("   %-22s %12.2f" % (name, r["avg_ts"]))
PY
    good "throughput recorded"
else
    tail -15 "$OUT/bench-on.log"
    badp "llama-bench failed - see $OUT/bench-on.log"
fi

# --------------------------------------------------------------- 4. the A/B --

step "4/4  Route on vs route off"

if [ "$DO_AB" != 1 ]; then
    info "skipped. The numbers above are absolute; on their own they do not say"
    info "whether the native route helped."
    info ""
    info "Re-run with --ab to get the actual delta. That builds a second copy"
    info "with the route switched off and benches it the same way. It is the"
    info "only honest speedup number, and it costs one more build."
else
    SRCDIR="$(cd "$BUILD/.." && pwd)"
    OFFSRC="$SRCDIR/../fattn-routeoff-$$"
    git -C "$SRCDIR" worktree add --detach "$OFFSRC" HEAD > "$OUT/worktree.log" 2>&1 \
        || die "could not create a route-off worktree (is $SRCDIR a git checkout?)"
    OFFSRC="$(cd "$OFFSRC" && pwd)"
    trap 'git -C "$SRCDIR" worktree remove --force "$OFFSRC" >/dev/null 2>&1 || true' EXIT

    python3 - "$OFFSRC/ggml/src/ggml-cuda/fattn.cu" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = ("static bool ggml_cuda_fattn_native_profitable("
          "const ggml_tensor * dst, const int gqa_ratio) {\n")
if anchor not in src:
    sys.exit("could not find ggml_cuda_fattn_native_profitable() to disable")
open(path, "w").write(src.replace(anchor, anchor + "    return false; // route off\n", 1))
PY
    [ $? -eq 0 ] || die "could not patch the route-off worktree"
    grep -q "route off" "$OFFSRC/ggml/src/ggml-cuda/fattn.cu" \
        || die "route-off patch did not apply; the A/B would compare two identical builds"
    info "route-off copy prepared, building it (one more full nvcc run)"

    cmake -S "$OFFSRC" -B "$OFFSRC/build" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DLLAMA_CURL=OFF > "$OUT/configure-off.log" 2>&1 \
        || { tail -20 "$OUT/configure-off.log"; die "route-off configure failed"; }
    cmake --build "$OFFSRC/build" --target llama-bench -j"$(nproc)" > "$OUT/build-off.log" 2>&1 \
        || { tail -30 "$OUT/build-off.log"; die "route-off build failed"; }

    run_bench "$OFFSRC/build/bin/llama-bench" off \
        || { tail -15 "$OUT/bench-off.log"; die "route-off bench failed"; }

    python3 - "$OUT/bench-on.json" "$OUT/bench-off.json" "$OUT/comparison.txt" <<'PY'
import json, sys

def load(path):
    out = {}
    for r in json.load(open(path)):
        if r.get("n_prompt", 0):
            name = "pp%d" % r["n_prompt"]
        else:
            name = "tg%d" % r.get("n_gen", 0)
        if r.get("n_depth", 0):
            name += " @ d%d" % r["n_depth"]
        out[name] = r["avg_ts"]
    return out

on, off = load(sys.argv[1]), load(sys.argv[2])
lines = ["Native route on vs off, same build otherwise. Positive = faster with",
         "the route. tg rows are the control: a batch of 1 never takes the",
         "route, so they should sit near zero.", "",
         "%-22s %10s %10s %9s" % ("test", "off t/s", "on t/s", "delta")]
for name in sorted(set(on) & set(off)):
    a, b = off[name], on[name]
    d = 100.0 * (b - a) / a if a else 0.0
    lines.append("%-22s %10.2f %10.2f %+8.1f%%" % (name, a, b, d))
text = "\n".join(lines) + "\n"
open(sys.argv[3], "w").write(text)
print("\n".join("   " + l for l in text.splitlines()))
PY
fi

# ----------------------------------------------------------------- wrap up --

step "Done"
{
    echo "date    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "model   : $MODEL"
    echo "name    : ${NAME:-unknown}"
    echo "head dim: $D_K / ${D_V:-?}   gqa: ${GQA:-?}"
    echo "route   : $ROUTE ($ROUTE_WHY)"
    echo "kv cache: $CTK    context: $CTX"
    echo "gpus    :"
    nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | sed 's/^/  /'
    [ -f "$OUT/comparison.txt" ] && { echo; cat "$OUT/comparison.txt"; }
} > "$OUT/summary.txt"
cat "$OUT/summary.txt" | sed 's/^/   /'
echo
info "everything written to $OUT"
info "please send that directory back"
