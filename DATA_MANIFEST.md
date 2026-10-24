# Data manifest

The per-flight tables are too large for git and are deposited on Zenodo (DOI: [10.5281/zenodo.23749893](https://doi.org/10.5281/zenodo.23749893)). The archive
mirrors this repository's layout: extract it at the repo root and every analysis script finds its
inputs at the paths it expects. Raw PX4 ULogs are not deposited.

## Contents of the deposit (cruise segment, ~650 MB uncompressed)
| path | rows | what |
|---|---|---|
| `results/campaign_v2/combined_cruise.csv` | 90,104 | v2 baseline |
| `results/campaign_v3/combined_cruise.csv` | 20,005 | v3-B (headings, turns, climbs) |
| `results/campaign_v3_full/combined_cruise.csv` | 66,031 | v3-B + v3-C (arcs, `v3_block == c1`) |
| `results/campaign_v3a/combined_cruise.csv` | 18,391 | v3-A autotune |
| `results/campaign_v3d/combined_cruise.csv` | 19,982 | v3-D Dryden-form vs white |
| `results/campaign_v4/combined_cruise.csv` | 80,847 | v4 delivery routes |
| `results/campaign_saltelli_merged/combined_cruise.csv` | 70,958 | v5 Saltelli, as first flown |
| `results/campaign_saltelli_merged/excluded_plugin_failure.csv` | 5,087 | v5 flights flown without wind (below) |
| `results/campaign_saltelli_refly/combined_cruise.csv` | 4,979 | re-flights of those runs |
| `results/campaign_saltelli_final/analysis_cruise.csv` | 69,329 | **v5 set used in the paper** |
| `generated/designs/` | | flown design matrices: `sobol_v2_2e14.csv` (v2), `v3/v3_b{1,2}.csv`, `v3a/v3_a.csv`, `v3_curv/v3_c.csv`, `v3_dryden/v3_d.csv`, `v4_delivery/v4_c{1..4}.csv`, `saltelli/saltelli_all.csv`; `multi_airframe_sobol.csv` (v1 pilot design, read by `make_v2_design.py`) |

Each `combined_cruise.csv` holds every flight that produced metrics, one row per run. The canonical
`analysis_*.csv` / `excluded_*.csv` tables are not deposited; `make_analysis_set.py` regenerates them
deterministically, and `analysis_manifest.json` records their expected SHA-256 (after
running `make_analysis_set.py cruise`, `git diff analysis_manifest.json` is empty when the tables match). The flown mission
plans (`generated/sobol_v2/plans/`) are small and are tracked in git. The `armed` segment
(whole armed interval) is not used in the paper and is not deposited.

## Canonical analysis set
The paper reports one N per experiment: the **canonical analysis set** of *unique, valid,
not-cooked* flights, produced by `scripts/analysis/make_analysis_set.py`:
- **unique**: deduplicated by `run_id`;
- **valid**: not `from_failed_attempt`;
- **not-cooked**: takeoff did not run at 2× sim speed;
- **one per configuration**: dominant repeat stratum (Sobol experiments only; the Saltelli design is
  deduplicated by run, not by configuration).

It writes `analysis_<seg>.csv` and `excluded_<seg>.csv` beside each `combined_<seg>.csv`
(union = combined, so every exclusion is recoverable), plus `analysis_manifest.json` at the repo
root with per-experiment counts and SHA-256 hashes.

**v5 wind-free flights and re-flights.** On two hosts the Dryden gust plugin failed to load for
part of the v5 run (`gazebo.log`: "Failed to load system plugin [DrydenWind]"), so 5,087 of the
69,538 canonical v5 flights had no wind or turbulence. They are listed in
`excluded_plugin_failure.csv` (per-run scan: `scripts/saltelli/scan_plugin_failures.py`) and were
re-flown with the same design rows; the scan was repeated on the re-flights to confirm the plugin
loaded. `results/campaign_saltelli_final/analysis_cruise.csv` = the 64,451 unaffected flights + 4,878 valid re-flights =
69,329 runs, the set behind every v5 number in the paper. The paper's total of 309,667 analyzed
flights is the `_TOTAL_analysis_N` in `analysis_manifest.json` (309,876) with the v5 canonical set
(69,538) replaced by this final set. The v5 estimator scripts read it when
`TDR_V5_CSV` points at it (set `TDR_V5_TAG=final` to name their outputs `*_final`).

## Per-flight table schema
- key: `run_id`; `sample_index`, `airframe_id`, `plan`, `turb_model`
- design factors: `nominal_mass_kg`, `payload_mass_kg`, `payload_mass_pct`, `thrust_to_weight`,
  `inertia_scale`, `com_y_m`, `mean_wind_m_s`, `turbulence_rms_m_s`, `wind_direction_deg`
- outcome metrics: `cross_track_rmse` (primary, see below), `sob2`, `rms_error`, `cross_track_max`,
  `max_error`, `altitude_rmse`, `samples`, `duration_s`
- bookkeeping: `from_failed_attempt`, `repeat`, `takeoff_sim_speed_factor`; the run-location columns
  (`log_file`, `ulog_path`, `canonical_ulog_path`, `attempt_dir`) are relative to the original run
  directory (`runs/...`) and the simulation hosts are anonymized in `src_host` (`host01`–`host12`).
  Raw ULogs are not deposited, so these columns are provenance only; no analysis reads them.
- pairing: `paired_with_run_id` links a v3/v4 flight to its matched control flight
- Saltelli rows map to `saltelli_index` via `generated/designs/saltelli/saltelli_all.csv`; never
  shuffle it, the estimators read runs in sample order.

Segments: `cruise` (from 1 m of commanded horizontal motion to the end of the armed log; used in the
paper) and `armed` (whole armed interval).

**Metric naming.** The column `cross_track_rmse` is the paper's **3-D setpoint-path RMSE (3-D SPE)**,
not conventional lateral cross-track error: the RMS over setpoint samples of the 3-D distance from the
vehicle to PX4's commanded position-setpoint polyline (`_cross_track_errors` in
`tdresearch/metrics.py`). The column keeps its historical name. The horizontal point-to-point RMSE is
exactly `sqrt(rms_error² − altitude_rmse²)`.

## Regenerating the tables from ULogs
```
python -m tdresearch analyze <ulog_dir> --metadata <design.csv> --flight-segment cruise \
    --jobs 8 --output results/campaign_<exp>/combined_cruise.csv
# (repeat with --flight-segment armed; concatenate if flights were split across machines)
python scripts/analysis/make_analysis_set.py cruise armed
```

## Building the deposit archive
The archive holds exactly the files listed in `DATA_CHECKSUMS.txt`, plus this manifest:
```
tar -czf tdresearch_data.tar.gz DATA_MANIFEST.md DATA_CHECKSUMS.txt $(awk '!/^#/{print $2}' DATA_CHECKSUMS.txt)
```

## Checksums
`DATA_CHECKSUMS.txt` lists SHA-256 hashes of the deposited tables so a download or a regenerated
table can be verified against the version the paper was computed from.
