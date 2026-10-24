"""Build the v2 design: Sobol 2^14, one replicate, plus a 10% duplicated stratum.

Critical precondition checked here first: the first 2^11 points of the 2^14 draw must
equal the existing v1 design's points. Scipy's scrambled Sobol is deterministic from
the seed, so drawing 16,384 points should reproduce the first 2,048 exactly -- if it
does not, v1 and v2 do not live in the same sample space and nothing downstream is
comparable.

The 10% stratum is chosen with a fixed seed so the design is reproducible, and is
stratified on mean wind so the replicate estimate of seed noise is not dominated by
calm-air runs (failures and variability both concentrate at high wind).

The input is the raw 2^14 draw, which is rewritten IN PLACE (never point it at the flown design):
  python -m tdresearch generate-sobol-design --samples-per-airframe 16384 --repetitions 1 --output raw.csv
  python scripts/v2/make_v2_design.py raw.csv     # -> byte-identical to generated/designs/sobol_v2_2e14.csv
"""
import csv, os, random, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
V1 = os.path.join(ROOT, "generated/designs/multi_airframe_sobol.csv")
if len(sys.argv) != 2:
    raise SystemExit("usage: make_v2_design.py <raw_2e14_design.csv>  (rewritten in place; see docstring)")
V2 = sys.argv[1]
REPL_FRAC = 0.10
REPL_SEED = 20260728

# columns that describe the CONFIG (independent of which replicate it is)
# execution_order is a shuffle index over the whole design, so it legitimately
# differs between an 81,920-row and a 102,400-row file. It is bookkeeping, not a
# design parameter.
SKIP = {"run_id", "repeat", "wind_seed", "case", "run_name", "execution_order"}


def load(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


v2 = load(V2)
print("v2 raw rows: %d" % len(v2))
by_si = collections.Counter(int(r["sample_index"]) for r in v2)
print("  sample_index range: %d..%d  distinct=%d  rows per index=%s"
      % (min(by_si), max(by_si), len(by_si), sorted(set(by_si.values()))))

# ---- precondition: first 2^11 points must match v1 ----
v1 = [r for r in load(V1) if r["repeat"] == "1"]
v1k = {(r["airframe_id"], r["sample_index"]): r for r in v1}
overlap = [r for r in v2 if int(r["sample_index"]) < 2048]
print("\nreproduction check on the %d overlapping configs" % len(overlap))
cols = [c for c in v2[0] if c not in SKIP and c in v1[0]]
bad, checked = [], 0
for r in overlap:
    o = v1k.get((r["airframe_id"], r["sample_index"]))
    if o is None:
        bad.append((r["airframe_id"], r["sample_index"], "missing in v1"))
        continue
    checked += 1
    for c in cols:
        a, b = r[c], o[c]
        if a == b:
            continue
        try:
            if abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b))):
                continue
        except ValueError:
            pass
        bad.append((r["airframe_id"], r["sample_index"], "%s: %s != %s" % (c, a, b)))
print("  compared %d configs across %d design columns" % (checked, len(cols)))
if bad:
    percol = collections.Counter(x[2].split(":")[0] for x in bad)
    print("  MISMATCHES: %d   by column: %s" % (len(bad), dict(percol)))
    for x in bad[:5]:
        print("    ", x)
    raise SystemExit("ABORT: v2 does not reproduce v1's points; do not generate assets")
print("  IDENTICAL -- v1 is exactly the first 2^11 of this 2^14 design")

# ---- 10% replicated stratum, stratified on wind ----
rng = random.Random(REPL_SEED)
by_bin = collections.defaultdict(list)
for r in v2:
    try:
        b = min(int(float(r["mean_wind_m_s"])), 5)
    except Exception:
        b = 0
    by_bin[b].append(r)
chosen = set()
for b, rs in sorted(by_bin.items()):
    k = int(round(len(rs) * REPL_FRAC))
    for r in rng.sample(rs, k):
        chosen.add(r["model_name"])
print("\n10%% replicated stratum: %d configs (%.2f%% of %d), stratified over %d wind bins"
      % (len(chosen), 100.0 * len(chosen) / len(v2), len(v2), len(by_bin)))

out = []
for r in v2:
    r = dict(r)
    r["repeat"] = "1"
    r["run_id"] = "%s_n01" % r["model_name"]
    out.append(r)
dupes = 0
for r in v2:
    if r["model_name"] not in chosen:
        continue
    d = dict(r)
    d["repeat"] = "2"
    d["run_id"] = "%s_n02" % d["model_name"]
    out.append(d)
    dupes += 1

out.sort(key=lambda r: (int(r["sample_index"]), r["airframe_id"], r["repeat"]))
tmp = V2 + ".tmp"
with open(tmp, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
os.replace(tmp, V2)
print("\nwrote %s" % V2)
print("  configs: %d   runs: %d  (%d singles + %d duplicated)"
      % (len(v2), len(out), len(v2), dupes))
print("  run_id uniqueness: %s" % ("OK" if len({r["run_id"] for r in out}) == len(out) else "COLLISION"))
