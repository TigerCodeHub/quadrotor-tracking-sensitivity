"""Write the subsets behind the paper's robustness checks.

  v3a_no_x500.csv   v3-A combined table without the X500 (whose autotuned configurations are 31.6% missing)
  v2_noretry.csv    v2 canonical set without flights that succeeded only after a retry
  v4_noretry.csv    v4 canonical set without flights that succeeded only after a retry

usage: robustness_subsets.py <out_dir>   (run from anywhere; inputs are read from the repo)
"""
import csv
import os
import sys

csv.field_size_limit(1 << 30)
out_dir = os.path.abspath(sys.argv[1])
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
os.makedirs(out_dir, exist_ok=True)


def retried(r):
    return str(r.get("succeeded_after_retry", "")).strip().lower() in ("true", "1")


def subset(src, name, keep):
    with open(src, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = [r for r in rd if keep(r)]
        fields = rd.fieldnames
    with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows from {src}")


subset("results/campaign_v3a/combined_cruise.csv", "v3a_no_x500.csv", lambda r: r["airframe_id"] != "px4_x500")
subset("results/campaign_v2/analysis_cruise.csv", "v2_noretry.csv", lambda r: not retried(r))
subset("results/campaign_v4/analysis_cruise.csv", "v4_noretry.csv", lambda r: not retried(r))
