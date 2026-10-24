"""Extra paired-contrast tables for the paper: v3-B b2 mean paired change by plan; v3-B b1 by heading.
Paired delta = log(metric_v3) - log(metric_v2 control), control found via paired_with_run_id.
Mean, SE of the mean (sd/sqrt(n); rows are distinct configs), t and two-sided normal p.
usage: TDR_METRIC=cross_track_rmse paper_tables_extra.py"""
import csv
import math
import os
import statistics
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
M = os.environ.get("TDR_METRIC", "cross_track_rmse")
v2 = {r["run_id"]: r for r in csv.DictReader(open("results/campaign_v2/combined_cruise.csv", encoding="utf-8"))}
v3 = list(csv.DictReader(open("results/campaign_v3/analysis_cruise.csv", encoding="utf-8")))


def p2(t):
    return math.erfc(abs(t) / math.sqrt(2))


def f(r):
    try:
        v = float(r[M])
        return v if v > 0 else None
    except Exception:
        return None


groups = defaultdict(list)
for r in v3:
    blk = r.get("v3_block", "")
    c = v2.get(r.get("paired_with_run_id") or "")
    if c is None or f(r) is None or f(c) is None:
        continue
    d = math.log(f(r)) - math.log(f(c))
    plan = r.get("plan", "")
    if blk == "b2":
        groups[("b2", plan)].append(d)
    elif blk == "b1":
        groups[("b1", plan)].append(d)
print("paired mean change in log(3-D SPE) vs the config's v2 straight flight")
print(f"{'block':<5}{'plan':<14}{'n':>6}{'mean':>9}{'SE':>8}{'t':>8}{'p':>10}")
tot = defaultdict(list)
for (blk, plan), v in sorted(groups.items()):
    n = len(v); m = statistics.mean(v); se = statistics.stdev(v) / math.sqrt(n); t = m / se
    print(f"{blk:<5}{plan:<14}{n:>6}{m:>+9.4f}{se:>8.4f}{t:>+8.1f}{p2(t):>10.2g}")
    fam = ("turns" if plan.startswith("turn") else "climbs" if plan.startswith("climb") else "headings")
    tot[(blk, fam)].extend(v)
print("-- pooled families")
for (blk, fam), v in sorted(tot.items()):
    n = len(v); m = statistics.mean(v); se = statistics.stdev(v) / math.sqrt(n); t = m / se
    print(f"{blk:<5}{fam:<14}{n:>6}{m:>+9.4f}{se:>8.4f}{t:>+8.1f}{p2(t):>10.2g}")
