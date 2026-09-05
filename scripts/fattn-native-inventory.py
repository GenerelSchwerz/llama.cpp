#!/usr/bin/env python3
"""Report and check the compiled quantized-native MMA FlashAttention inventory.

The route's build switches are coupled in a way that no runtime test observes.
GGML_CUDA_FA_ALL_QUANTS decides which cache types get kernels at all, and an
extern-macro mistake can instantiate a whole attention body per tile shape in
one translation unit. Both compile and pass every runtime test.

So this reads the built library back and compares it against the exact set of
cases the generated instance files declare, filtered by the manifest tiers in
ggml/src/ggml-cuda/fattn-mma-quant-types.h. Missing, unexpected and duplicate
kernels all fail, as does any mixed K/V kernel or any non-zero logit softcap
specialization, neither of which the route can reach.

Usage:
    scripts/fattn-native-inventory.py build/bin/libggml-cuda.so [--all-quants]
                                      [--manifest PATH] [--json PATH]

Exit status is non-zero when an invariant fails.
"""

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_DIR = os.path.join(REPO_ROOT, "ggml", "src", "ggml-cuda")
DEFAULT_MANIFEST = os.path.join(CUDA_DIR, "fattn-mma-quant-types.h")
INSTANCE_GLOB = os.path.join(CUDA_DIR, "template-instances", "fattn-mma-quant-instance-*.cu")

ENTRY_RE = re.compile(
    r"^\s*ENTRY\(\s*(GGML_TYPE_\w+)\s*,\s*(\w+)\s*,\s*(DEFAULT|EXTRA)\s*,\s*ARGS\s*\)")

# ggml_type values appear in demangled names as "(ggml_type)N", so the manifest's
# enum names have to be resolved to numbers. ggml.h is the source for that.
GGML_TYPE_ENUM_RE = re.compile(r"^\s*GGML_TYPE_(\w+)\s*=\s*(\d+)\s*,")

CASE_RE = re.compile(
    r"^\s*DECL_FATTN_MMA_QUANT_CASE\(\s*(GGML_TYPE_\w+)\s*,"
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

# <DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, compact_causal_prefix, type_K, type_V>
KERNEL_RE = re.compile(
    r"flash_attn_ext_f16<\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),"
    r"\s*(\w+),\s*(\w+),\s*(\w+),"
    r"\s*\(ggml_type\)(\d+),\s*\(ggml_type\)(\d+)\s*>")


def read_manifest(path):
    tiers = {}
    with open(path) as f:
        for line in f:
            m = ENTRY_RE.match(line)
            if m:
                tiers[m.group(1)] = m.group(3)
    if not tiers:
        sys.exit(f"{path}: no manifest entries found")
    return tiers


def read_ggml_type_values():
    header = os.path.join(REPO_ROOT, "ggml", "include", "ggml.h")
    values = {}
    with open(header) as f:
        for line in f:
            m = GGML_TYPE_ENUM_RE.match(line)
            if m:
                values["GGML_TYPE_" + m.group(1)] = int(m.group(2))
    if not values:
        sys.exit(f"{header}: no ggml_type enumerators found")
    return values


def read_expected_cases(all_quants):
    """The cases the generated instance files declare for this build.

    Reading the sources rather than restating the route table keeps this script
    from becoming a second copy of it: what it checks is that the library holds
    exactly what the build was told to instantiate.
    """
    cases = set()
    files = sorted(glob.glob(INSTANCE_GLOB))
    if not files:
        sys.exit(f"{INSTANCE_GLOB}: no generated instance files found")
    for path in files:
        guarded = False
        with open(path) as f:
            for line in f:
                if line.startswith("#ifdef GGML_CUDA_FA_ALL_QUANTS"):
                    guarded = True
                elif line.startswith("#endif"):
                    guarded = False
                m = CASE_RE.match(line)
                if m and (all_quants or not guarded):
                    cases.add((m.group(1),) + tuple(int(m.group(i)) for i in range(2, 6)))
    if not cases:
        sys.exit(f"{INSTANCE_GLOB}: no DECL_FATTN_MMA_QUANT_CASE entries found")
    return cases


def read_symbols(library):
    """Demangled symbol names from the library.

    The kernels have internal linkage, so they are local symbols; nm lists them
    unless the library was stripped. cuobjdump is the fallback for a stripped
    build, where the device images still carry the names.
    """
    out = subprocess.run(["nm", "-C", library], capture_output=True, text=True).stdout
    if "flash_attn_ext_f16" in out:
        return out, "nm"
    out = subprocess.run(["cuobjdump", "--dump-elf-symbols", library],
                         capture_output=True, text=True).stdout
    if "flash_attn_ext_f16" in out:
        return out, "cuobjdump"
    sys.exit(f"{library}: no flash_attn_ext_f16 symbols found via nm or cuobjdump")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--all-quants", action="store_true",
                    help="the build set GGML_CUDA_FA_ALL_QUANTS")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--json", help="write the inventory to this file")
    args = ap.parse_args()

    tiers = read_manifest(args.manifest)
    type_values = read_ggml_type_values()
    for name in tiers:
        if name not in type_values:
            sys.exit(f"{args.manifest}: {name} is not a ggml_type enumerator")
    value_to_name = {type_values[n]: n for n in tiers}

    expected = read_expected_cases(args.all_quants)
    symbols, source = read_symbols(args.library)

    failures = []
    found = collections.Counter()   # case -> distinct kernel symbols
    variants = {}                   # case -> the bool triples seen
    mixed = set()
    softcap = set()
    duplicates = collections.Counter()

    seen_symbols = collections.Counter()
    for m in KERNEL_RE.finditer(symbols):
        geometry = tuple(int(m.group(i)) for i in range(1, 5))
        bools = (m.group(5), m.group(6), m.group(7))
        k, v = int(m.group(8)), int(m.group(9))
        if k not in value_to_name and v not in value_to_name:
            continue  # an F16/BF16 kernel, not this route's
        if k != v:
            mixed.add((geometry, k, v))
            continue
        case = (value_to_name[k],) + geometry
        seen_symbols[(case, bools)] += 1
        if bools[0] != "false":
            softcap.add(case)
        variants.setdefault(case, set()).add(bools)

    for (case, bools), count in seen_symbols.items():
        found[case] += 1
        if count > 1:
            duplicates[(case, bools)] = count

    missing    = sorted(expected - set(found))
    unexpected = sorted(set(found) - expected)

    print(f"library      : {args.library}")
    print(f"size         : {os.path.getsize(args.library)} bytes")
    print(f"symbol source: {source}")
    print(f"all-quants   : {'ON' if args.all_quants else 'OFF'}")
    print()
    print(f"{'type':<16}{'tier':<10}{'DKQ':>6}{'DV':>6}{'ncols1':>8}{'ncols2':>8}{'symbols':>9}")
    for case in sorted(expected):
        print("{:<16}{:<10}{:>6}{:>6}{:>8}{:>8}{:>9}".format(
            case[0], tiers[case[0]], case[1], case[2], case[3], case[4], found.get(case, 0)))

    for case in missing:
        failures.append(f"declared but not compiled: {case}")
    for case in unexpected:
        failures.append(f"compiled but not declared: {case}")
    for (case, bools), count in sorted(duplicates.items()):
        failures.append(f"duplicate instantiation ({count}x): {case} {bools}")
    for geometry, k, v in sorted(mixed):
        failures.append(f"mixed K/V kernel, which the route cannot select: {geometry} K={k} V={v}")
    for case in sorted(softcap):
        failures.append(f"logit softcap specialization, which the route cannot select: {case}")

    # Every case must carry the same kernel variants, otherwise one of them lost
    # or gained a compact-causal-mask specialization.
    shapes = {frozenset(v) for v in variants.values()}
    if len(shapes) > 1:
        failures.append(f"kernel variants differ between cases: {sorted(map(sorted, shapes))}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "library": args.library,
                "size_bytes": os.path.getsize(args.library),
                "all_quants": args.all_quants,
                "expected": sorted(expected),
                "found": {str(k): v for k, v in sorted(found.items())},
                "tiers": tiers,
            }, f, indent=2, sort_keys=True)

    print()
    if failures:
        for f in failures:
            print("FAIL: " + f)
        return 1
    print(f"OK: {len(expected)} declared cases, all compiled, nothing else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
