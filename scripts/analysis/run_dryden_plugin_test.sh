#!/bin/bash
# Tier-2 Dryden validation: run the compiled DrydenWind.so headless in gz-sim on ONE host and
# analyse its own log. Run where gz-sim and the built plugin are available, e.g.:
#   nice -n 19 bash scripts/analysis/run_dryden_plugin_test.sh
# One gz process at a time; scratch under /tmp ($W). Set PY to a Python with numpy/scipy if
# python3 on PATH lacks them.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PLUG=$REPO/gz_plugins/dryden_wind/build
STEPS=${STEPS:-1000000}               # x 0.004 s = 4000 s sim
PY=${PY:-python3}
W=${W:-/tmp/dryden_plugin_test}
OUT=$REPO/results/dryden_validation
mkdir -p "$W" "$OUT"

# mkworld <name> <seed> <z> <plugin-params-xml>
# Heavy zero-gravity probe with 3 wind-enabled links. force_scaling_factor is tiny (1e-5) so the
# probe stays put (V held at min_airspeed, altitude constant) -> stationary gust, constant tau.
mkworld() {
  local name=$1 seed=$2 z=$3 params=$4 links=""
  for k in 1 2 3; do
    links+="<link name=\"b$k\"><enable_wind>true</enable_wind><inertial><mass>1000.0</mass><inertia><ixx>100</ixx><iyy>100</iyy><izz>100</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial><collision name=\"c\"><geometry><box><size>0.2 0.2 0.2</size></box></geometry></collision></link>"
  done
  cat > "$W/$name.sdf" <<EOF
<?xml version="1.0" ?>
<sdf version="1.9"><world name="dryden_test">
<gravity>0 0 0</gravity>
<physics name="fast" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>0</real_time_factor></physics>
<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
<plugin filename="DrydenWind" name="tdr::DrydenWind">
<mean_wind>0 0 0</mean_wind>
$params
<min_airspeed>5.0</min_airspeed>
<force_scaling_factor>1e-5</force_scaling_factor>
<seed>$seed</seed>
<log_file>$W/$name.csv</log_file>
</plugin>
<model name="probe"><pose>0 0 $z 0 0 0</pose>$links</model>
</world></sdf>
EOF
}

FIX='<sigma>2.0 2.0 2.0</sigma><length_scale>50 50 50</length_scale>'
mkworld fixed_s1 12345 30 "$FIX"
mkworld fixed_s2 987654 30 "$FIX"
mkworld alt10m   4242  10 '<w20>7.0</w20>'
mkworld alt60m   777   60 '<w20>7.0</w20>'

export GZ_SIM_SYSTEM_PLUGIN_PATH=$PLUG
for n in fixed_s1 fixed_s2 alt10m alt60m; do
  echo "== running $n $(date +%T)"
  rm -f "$W/$n.csv"
  nice -n 19 gz sim -s -r --iterations "$STEPS" "$W/$n.sdf" > "$W/$n.gzlog" 2>&1
  echo "   rc=$? rows=$(wc -l < "$W/$n.csv")"; tail -n 3 "$W/$n.gzlog"
done

[ -x "$PY" ] || { echo "no $PY here (container?): run analysis on host:"; echo "$PY $REPO/scripts/analysis/validate_dryden_plugin.py --case ... (see below)"; }
[ -x "$PY" ] && nice -n 19 "$PY" "$REPO/scripts/analysis/validate_dryden_plugin.py" \
  --case "fixed_s1:$W/fixed_s1.csv:fixed:2,2,2" --case "fixed_s2:$W/fixed_s2.csv:fixed:2,2,2" \
  --case "alt10m:$W/alt10m.csv:w20:7.0" --case "alt60m:$W/alt60m.csv:w20:7.0" \
  --json "$OUT/validate_dryden_plugin.json" --txt "$OUT/validate_dryden_plugin.txt"
[ -x "$PY" ] && rm -rf "$W"
