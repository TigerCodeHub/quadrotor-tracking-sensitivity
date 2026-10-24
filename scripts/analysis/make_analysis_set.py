"""Materialize the single canonical analysis set per campaign (audit pass-2 #3, revised).

Canonical N = unique, valid, not-cooked flights:
  * unique      : deduplicated by run_id
  * valid       : not from_failed_attempt
  * not-cooked  : takeoff did NOT use 2x sim speed
  * one per configuration : dominant repeat stratum  (Sobol campaigns only; NOT Saltelli,
                            whose rows are structured A/B/AB_i samples, not free configs)

For each campaign/segment it writes, next to combined_<seg>.csv:
  * analysis_<seg>.csv  -- the canonical set (THE pipeline input from now on)
  * excluded_<seg>.csv  -- every dropped row PLUS an `exclusion_reason` column (RECOVERABLE:
                           union(analysis, excluded) reconstructs combined exactly; combined is
                           also retained untouched).
and an analysis_manifest.json at the repo root with per-campaign counts.

Usage: make_analysis_set.py [seg ...]   (default: cruise armed)
"""
import os, sys, csv, json, collections, hashlib

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
# campaign -> apply dominant-repeat dedup? (Saltelli: no)
CAMPS = {"v2": ("results/campaign_v2", True),
         "v3a": ("results/campaign_v3a", True),
         # v3-B (b1 headings + b2 maneuvers, first-half design) lives in campaign_v3; the v3-C arcs
         # (v3_block == "c1") are only in campaign_v3_full, so they get their own entry + block filter.
         "v3b": ("results/campaign_v3", True),
         "v3c": ("results/campaign_v3_full", True),
         "v3d": ("results/campaign_v3d", True),
         "v4": ("results/campaign_v4", True),
         "saltelli": ("results/campaign_saltelli_merged", False)}


def sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


PRIMARY_METRIC = "cross_track_rmse"  # 3-D SPE; rows without a finite positive value are not "valid"
BLOCK_FILTER = {"v3c": "c1"}  # campaign key -> required v3_block value
# v4: keep only run_ids that appear in the c1..c4 design CSVs (their gap-fill CSVs are subsets of them).
ID_FILTER = {"v4": [f"generated/designs/v4_delivery/v4_c{i}.csv" for i in (1, 2, 3, 4)],
             # v3-B = blocks b1 (headings) + b2 (maneuvers) of the first-half design only; stray a1/a2 rows
             # from the abandoned retuning arms are not v3-B
             "v3b": ["generated/designs/v3/v3_b1.csv", "generated/designs/v3/v3_b2.csv"]}


def process(path, dedup_config, block=None, id_files=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if block is not None:
        rows = [r for r in rows if r.get("v3_block") == block]
    outside = []
    if id_files:
        keep = set()
        for fp in id_files:
            keep |= {r["run_id"] for r in csv.DictReader(open(fp, encoding="utf-8"))}
        outside = [r for r in rows if r.get("run_id") not in keep]
        rows = [r for r in rows if r.get("run_id") in keep]
    fields = rows[0].keys() if rows else []
    rep = collections.Counter(r.get("repeat") for r in rows if r.get("repeat"))
    keep_rep = None
    if dedup_config and rep:
        top = max(rep.values())
        keep_rep = {k for k, v in rep.items() if v >= 0.99 * top}
    seen = set()
    incl, excl = [], []
    for r in rows:
        reason = None
        try:
            pm = float(r.get(PRIMARY_METRIC) or "nan")
        except ValueError:
            pm = float("nan")
        if r.get("from_failed_attempt") == "1":
            reason = "failed_attempt"
        elif not (pm == pm and pm > 0):  # finite and positive primary metric required ("valid")
            reason = "no_primary_metric"
        elif (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            reason = "double_takeoff"
        elif keep_rep is not None and r.get("repeat") not in keep_rep:
            reason = "nondominant_repeat"
        else:
            rid = r.get("run_id") or ""
            if rid and rid in seen:
                reason = "duplicate_run_id"
            elif rid:
                seen.add(rid)
        if reason:
            rr = dict(r); rr["exclusion_reason"] = reason; excl.append(rr)
        else:
            incl.append(r)
    for r in outside:
        rr = dict(r); rr["exclusion_reason"] = "outside_experiment_design"; excl.append(rr)
    return fields, incl, excl, dict(rep)


def main():
    segs = sys.argv[1:] or ["cruise", "armed"]
    manifest = {"rule": "unique(run_id) & valid(not failed_attempt) & not-cooked(not 2x-takeoff)"
                        " & one-per-config(dominant repeat; Saltelli exempt)",
                "segments": {}}
    for seg in segs:
        manifest["segments"][seg] = {}
        grand = 0
        for name, (d, dedup) in CAMPS.items():
            p = f"{d}/combined_{seg}.csv"
            if not os.path.exists(p):
                print(f"[{seg}] {name}: no {p}; skip"); continue
            fields, incl, excl, rep = process(p, dedup, BLOCK_FILTER.get(name), ID_FILTER.get(name))
            ap = f"{d}/analysis_{seg}.csv"; ep = f"{d}/excluded_{seg}.csv"
            with open(ap, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(fields)); w.writeheader(); w.writerows(incl)
            with open(ep, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(fields) + ["exclusion_reason"])
                w.writeheader(); w.writerows(excl)
            reasons = collections.Counter(e["exclusion_reason"] for e in excl)
            manifest["segments"][seg][name] = {
                "combined": len(incl) + len(excl), "analysis_N": len(incl),
                "excluded": dict(reasons), "dedup_config": dedup,
                "analysis_sha256": sha(ap), "excluded_sha256": sha(ep)}
            grand += len(incl)
            print(f"[{seg}] {name:9} combined={len(incl)+len(excl):6d}  analysis_N={len(incl):6d}  "
                  f"excluded={dict(reasons)}")
        manifest["segments"][seg]["_TOTAL_analysis_N"] = grand
        print(f"[{seg}] TOTAL canonical analysis N = {grand}")
    json.dump(manifest, open("analysis_manifest.json", "w"), indent=2)
    print("wrote analysis_manifest.json")


if __name__ == "__main__":
    main()
