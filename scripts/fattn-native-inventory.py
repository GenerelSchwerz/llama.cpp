#!/usr/bin/env python3
"""Report and check the compiled quantized-native MMA FlashAttention inventory.

The native route's build switches are coupled in a way that no runtime test
observes: GGML_CUDA_FA_ALL_QUANTS decides which cache types get kernels at all,
and an inventory mistake shows up as kernels that exist but are unreachable, or
as hundreds of kernels landing in one translation unit. Both compile and pass.

So this reads the built library back and checks it against the single-source
type manifest, ggml/src/ggml-cuda/fattn-mma-quant-types.h:

  * every default-tier type has symmetric and runtime-V kernels;
  * extra-tier types have them exactly when the build set GGML_CUDA_FA_ALL_QUANTS;
  * no type outside the manifest has native kernels.

Usage:
    scripts/fattn-native-inventory.py build/bin/libggml-cuda.so [--all-quants]
                                      [--manifest PATH] [--json PATH]

Exit status is non-zero when an invariant fails.
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "ggml", "src", "ggml-cuda", "fattn-mma-quant-types.h")

ENTRY_RE = re.compile(
    r"^\s*ENTRY\(\s*(GGML_TYPE_\w+)\s*,\s*(\w+)\s*,\s*(DEFAULT|EXTRA)\s*,\s*ARGS\s*\)")

# ggml_type values appear in demangled names as "(ggml_type)N", so the manifest's
# enum names have to be resolved to numbers. ggml.h is the source for that.
GGML_TYPE_ENUM_RE = re.compile(r"^\s*GGML_TYPE_(\w+)\s*=\s*(\d+)\s*,")

# One kernel template, two roles: a symmetric pair instantiates V as itself, a
# mixed pair instantiates V as the runtime sentinel and selects the V loader
# inside the kernel. The sentinel's value is read from the source rather than
# hard-coded, because it is defined relative to GGML_TYPE_COUNT.
KERNEL_RE = re.compile(
    r"flash_attn_ext_f16<\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*\w+,\s*\w+,"
    r"\s*\(ggml_type\)(\d+),\s*\(ggml_type\)(\d+)\s*>")
V_RUNTIME_SENTINEL_RE = re.compile(
    r"GGML_CUDA_FATTN_QUANT_V_RUNTIME\s*=\s*\(ggml_type\)\s*\(GGML_TYPE_COUNT\s*\+\s*(\d+)\)")


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


def read_v_runtime_sentinel(type_values):
    header = os.path.join(REPO_ROOT, "ggml", "src", "ggml-cuda", "fattn-mma-quant.cuh")
    m = V_RUNTIME_SENTINEL_RE.search(open(header).read())
    if not m:
        sys.exit(f"{header}: GGML_CUDA_FATTN_QUANT_V_RUNTIME not found")
    return type_values["GGML_TYPE_COUNT"] + int(m.group(1))


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

    v_runtime = read_v_runtime_sentinel(type_values)
    symbols, source = read_symbols(args.library)

    symmetric = {}
    runtime_v = {}
    foreign = set()
    for m in KERNEL_RE.finditer(symbols):
        k, v = int(m.group(5)), int(m.group(6))
        if v == v_runtime:
            name = value_to_name.get(k)
            if name is None:
                foreign.add(k)  # a runtime V loader for a type the manifest does not name
            else:
                runtime_v[name] = runtime_v.get(name, 0) + 1
        elif k == v and k in value_to_name:
            symmetric[value_to_name[k]] = symmetric.get(value_to_name[k], 0) + 1

    expected = {n: (t == "DEFAULT" or args.all_quants) for n, t in tiers.items()}

    print(f"library      : {args.library}")
    print(f"size         : {os.path.getsize(args.library)} bytes")
    print(f"symbol source: {source}")
    print(f"all-quants   : {'ON' if args.all_quants else 'OFF'}")
    print()
    print(f"{'type':<16}{'tier':<10}{'expected':<10}{'symmetric':>10}{'runtime-V':>11}")

    failures = []
    for name, tier in tiers.items():
        sym, rt = symmetric.get(name, 0), runtime_v.get(name, 0)
        want = expected[name]
        print(f"{name:<16}{tier:<10}{('yes' if want else 'no'):<10}{sym:>10}{rt:>11}")
        if want and (sym == 0 or rt == 0):
            failures.append(f"{name} ({tier}) is expected in this build but has "
                            f"{sym} symmetric and {rt} runtime-V kernels")
        if not want and (sym or rt):
            failures.append(f"{name} ({tier}) must not be compiled without "
                            f"GGML_CUDA_FA_ALL_QUANTS, found {sym} symmetric and "
                            f"{rt} runtime-V kernels")
        if want and sym != rt:
            failures.append(f"{name} has {sym} symmetric but {rt} runtime-V kernels; "
                            "the two entries are generated together and must match")

    if foreign:
        failures.append("runtime-V kernels for types outside the manifest: "
                        + ", ".join(str(v) for v in sorted(foreign)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "library": args.library,
                "size_bytes": os.path.getsize(args.library),
                "all_quants": args.all_quants,
                "symmetric": symmetric,
                "runtime_v": runtime_v,
                "tiers": tiers,
            }, f, indent=2, sort_keys=True)

    print()
    if failures:
        for f in failures:
            print("FAIL: " + f)
        return 1
    print("OK: compiled inventory matches the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
