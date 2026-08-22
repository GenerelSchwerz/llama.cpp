#!/usr/bin/env python3
import csv, sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    print("no rows")
    sys.exit(0)

def col(name):
    return [float(r[name]) for r in rows if r[name] not in ("", None)]

vram = col("vram_mib")
rss = col("rss_kib")
hwm = col("vmhwm_kib")

print(f"samples: {len(rows)}")
if vram:
    print(f"vram_mib: min={min(vram):.0f} max={max(vram):.0f} last={vram[-1]:.0f}")
if rss:
    print(f"rss_mib: min={min(rss)/1024:.1f} max={max(rss)/1024:.1f} last={rss[-1]/1024:.1f}")
if hwm:
    print(f"vmhwm_mib: max={max(hwm)/1024:.1f}")
