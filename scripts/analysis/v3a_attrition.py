"""v3-A attrition: design configurations with no usable autotuned flight, by airframe and (X500) by payload band.

Reads the v3-A design and the canonical v3-A analysis set. The breakdown by cause (no tuned gains, takeoff timeouts,
other failures) needs the per-run status and autotune files, which are not deposited; it is recorded in
results/campaign_v3a/attrition_causes.txt.
usage: python scripts/analysis/v3a_attrition.py
"""
import csv
import os
from collections import Counter

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
design = list(csv.DictReader(open("generated/designs/v3a/v3_a.csv", encoding="utf-8")))
usable = {r["run_id"] for r in csv.DictReader(open("results/campaign_v3a/analysis_cruise.csv", encoding="utf-8"))}

n, miss = Counter(), Counter()
for r in design:
    n[r["airframe_id"]] += 1
    miss[r["airframe_id"]] += r["run_id"] not in usable
print(f"designed {len(design)}  usable {len(design) - sum(miss.values())}  missing {sum(miss.values())}")
for af in sorted(n):
    print(f"   {af:<10} missing {miss[af]:>5} / {n[af]}  ({100 * miss[af] / n[af]:.1f}%)")

x500 = [r for r in design if r["airframe_id"] == "px4_x500"]
print("X500 missing by payload fraction:")
for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.31)):
    band = [r for r in x500 if lo <= float(r["payload_mass_pct"]) / 100 < hi]
    m = sum(r["run_id"] not in usable for r in band)
    print(f"   {lo:.1f}-{min(hi, 0.3):.1f}: {m} / {len(band)}  ({100 * m / len(band):.0f}%)")
