#!/bin/bash
# Re-run every analysis behind the paper from the deposited per-flight tables.
#   bash scripts/reproduce_paper.sh            # logs (stderr, status) under reproduce_out/
# Every committed output in results/ is regenerated in place (figures are drawn into figures/);
# afterwards `python scripts/check_reproduction.py` fails on anything that did not reproduce
# (it ignores the `generated_at` timestamps and last-digit float noise). Takes about 15 minutes.
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${PY:-}" ]; then  # first interpreter that actually runs (skips the Windows Store python3 stub)
  for c in python3 python; do "$c" -c "import sys" >/dev/null 2>&1 && { PY=$c; break; }; done
fi
OUT=${OUT:-$R/reproduce_out}
mkdir -p "$OUT"
: > "$OUT/status.tsv"
export TDR_METRIC=cross_track_rmse MPLBACKEND=Agg
cd "$R"   # relative paths below match the provenance recorded in the committed outputs
S=scripts
A=results

run() {  # run <dest> <cmd...>; stdout -> <dest>, stderr -> $OUT/<name>.err
  local dest=$1; shift
  local name; name=$(basename "$dest")
  "$@" > "$dest" 2> "$OUT/$name.err"
  local rc=$?
  printf "%s\t%s\n" "$dest" "$rc" | tee -a "$OUT/status.tsv"
}

# Canonical analysis sets (analysis_manifest.json; Table 2)
run "$OUT/make_analysis_set.txt" $PY $S/analysis/make_analysis_set.py cruise

# v2 / v3-D / v4 primary fits (Table 3, Fig. 1, v4 fit) and global fits (transfer table, Table 13)
for c in v2 v3d v4; do
  run $A/campaign_$c/refit_ct_cruise.txt $PY $S/analysis/refit_robust.py $A/campaign_$c/analysis_cruise.csv $c --json $A/campaign_$c/refit_ct_cruise.json
  run $A/campaign_$c/global_logmass.txt $PY $S/analysis/global_logmass.py $A/campaign_$c/analysis_cruise.csv $c
done
# Every regression the paper quotes on the primary footing: v2 wind direction, LOAO, v3-B, v3-C,
# v3-D, log-log exponents, metric-robustness table (Table 6)
run $A/primary_footing.txt $PY $S/analysis/primary_footing.py
run $A/campaign_v2/wind_direction.txt $PY $S/v2/wind_direction_analysis.py $A/campaign_v2/combined_cruise.csv v2
run $A/campaign_v2/loao.txt $PY $S/extensions/loao_analysis.py $A/campaign_v2/combined_cruise.csv v2
run $A/campaign_v2/dimensionless_cross_track.txt $PY $S/extensions/dimensionless_analysis.py $A/campaign_v2/combined_cruise.csv v2
run $A/campaign_v2/loao_descriptor_comparison.txt $PY $S/analysis/loao_descriptor_comparison.py
run $A/campaign_v4/loao_descriptor_comparison.txt $PY $S/analysis/loao_descriptor_comparison_v4.py
run $A/campaign_v3a/v3a_analysis.txt $PY $S/v3/v3a_analysis.py $A/campaign_v3a/combined_cruise.csv $A/campaign_v2/combined_cruise.csv generated/designs/v3a/v3_a.csv v3a
run $A/campaign_v3/v3b_analysis.txt $PY $S/v3/v3b_analysis.py $A/campaign_v3/combined_cruise.csv $A/campaign_v2/combined_cruise.csv v3b
run $A/paper_tables_extra.txt $PY $S/analysis/paper_tables_extra.py
run $A/campaign_v3_full/v3c_curvature.txt $PY $S/v3/curviness_analysis.py $A/campaign_v3_full/analysis_cruise.csv $A/campaign_v2/combined_cruise.csv generated/sobol_v2/plans v3c
run $A/campaign_v3d/pair_check.txt $PY $S/analysis/v3d_pair_check.py
run $A/campaign_v4/v4_paired_analysis_primary.txt $PY $S/v4/v4_paired_analysis.py $A/campaign_v4/combined_cruise.csv $A/campaign_v3d/combined_cruise.csv
run $A/campaign_v4/route_wind_interaction.txt $PY $S/v4/route_wind_interaction.py $A/campaign_v4/analysis_cruise.csv $A/campaign_v3d/analysis_cruise.csv --json $A/campaign_v4/route_wind_interaction.json
TDR_PAIR_SUFFIX="" run $A/campaign_v4/route_wind_interaction_allpaired.txt $PY $S/v4/route_wind_interaction.py $A/campaign_v4/analysis_cruise.csv $A/campaign_v3d/analysis_cruise.csv $A/campaign_v2/analysis_cruise.csv --json $A/campaign_v4/route_wind_interaction_allpaired.json
for c in v2 v3a v3d v4; do run "$OUT/retry_$c.txt" $PY $S/analysis/retry_selection_test.py $A/campaign_$c/analysis_cruise.csv $c; done
cat "$OUT"/retry_v2.txt "$OUT"/retry_v3a.txt "$OUT"/retry_v3d.txt "$OUT"/retry_v4.txt > $A/retry_selection_test.txt
run $A/v2_replicate_noise.txt $PY $S/analysis/v2_replicate_noise.py
run $A/campaign_v3a/v3a_attrition.txt $PY $S/analysis/v3a_attrition.py
# Robustness: v3-A without the X500 (attrition), v2/v4 primary fits without retried flights
SUB=reproduce_out/subsets   # repo-relative so the outputs carry no machine paths
run "$OUT/robustness_subsets.txt" $PY $S/analysis/robustness_subsets.py $SUB
run $A/campaign_v3a/v3a_analysis_no_x500.txt $PY $S/v3/v3a_analysis.py $SUB/v3a_no_x500.csv $A/campaign_v2/combined_cruise.csv generated/designs/v3a/v3_a.csv v3a_no_x500
for c in v2 v4; do run $A/campaign_$c/refit_ct_cruise_noretry.txt $PY $S/analysis/refit_robust.py $SUB/${c}_noretry.csv ${c}_noretry --responses cross_track_rmse; done

# v5 Saltelli indices (Fig. 2, Tables 14-15) on the final 69,329-run set
export TDR_V5_CSV=$A/campaign_saltelli_final/analysis_cruise.csv TDR_V5_TAG=final
run $A/campaign_saltelli_final/completeblock.txt $PY $S/saltelli/saltelli_completeblock.py generated/designs/saltelli $A/campaign_saltelli_final --segments cruise
run $A/campaign_saltelli_merged/final_crosscheck_given_data.txt $PY $S/saltelli/saltelli_crosscheck_given_data.py
run $A/campaign_saltelli_merged/final_second_order_given_data.txt $PY $S/saltelli/saltelli_second_order_given_data.py
run $A/campaign_saltelli_merged/final_second_order_all_pairs.txt $PY $S/saltelli/saltelli_second_order_all_pairs.py
unset TDR_V5_CSV TDR_V5_TAG

# Tables and figures
run "$OUT/make_paper_tables_md.txt" $PY $S/analysis/make_paper_tables_md.py
run "$OUT/make_paper_figures.txt" $PY $S/analysis/make_paper_figures.py
# Dryden-form gust generator, Python mirror (Table 7); the paper's run used 3e6 steps
run "$OUT/validate_dryden.txt" $PY $S/analysis/validate_dryden.py --steps 3000000 --json $A/dryden_validation/validate_dryden.json
run $A/dryden_validation/validate_dryden_extended.txt $PY $S/analysis/validate_dryden_extended.py --json $A/dryden_validation/validate_dryden_extended.json

echo "logs in $OUT; failures:"; awk -F'\t' '$2 != 0' "$OUT/status.tsv"
