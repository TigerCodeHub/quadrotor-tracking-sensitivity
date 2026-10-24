"""Union the pre-fly2 v3-A combined CSV with the post-fly2 combined CSV (dedup by run_id).

The fly2 per-host analyze overwrote host CSVs with only the runs still on each host, so the
earlier scored flights (7,413 pairs) survive only in combined_<seg>_pre_fly2.csv. Keep the
pre-fly2 row when a run_id appears in both (earliest flight), add every new run_id.
usage: merge_v3a_combined.py <dir> <seg>
"""
import csv
import sys
from pathlib import Path

d = Path(sys.argv[1]); seg = sys.argv[2]
pre = d / f"combined_{seg}_pre_fly2.csv"
new = d / f"combined_{seg}_postfly2_only.csv"
out = d / f"combined_{seg}.csv"
if not new.exists():
    out.replace(new)  # keep what tdr.sh combine produced
rows_pre = list(csv.DictReader(open(pre, encoding="utf-8")))
rows_new = list(csv.DictReader(open(new, encoding="utf-8")))
fields = list(rows_pre[0].keys())
for r in rows_new:
    for k in r:
        if k not in fields:
            fields.append(k)
seen, merged = set(), []
for r in rows_pre + rows_new:
    rid = r["run_id"]
    if rid in seen:
        continue
    seen.add(rid)
    merged.append(r)
with open(out, "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(merged)
print(f"pre={len(rows_pre)} new={len(rows_new)} merged unique={len(merged)} -> {out}")
