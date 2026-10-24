"""Generate constant-arc-length circular-arc plans to isolate curvature from path length.

The v3-B turn plans confounded curvature with path length (a sharper dogleg was also a longer
route), and SOB2 normalises per step -- so the v3-B "curvier = lower error" was mostly a
length-normalisation artifact (see results/campaign_v3/curviness.md). These plans fix the
length at 1000 m (matching the v2 straight control exactly) and vary ONLY the curvature, so a
config paired against its v2 straight control gives a clean within-config, length-matched
curvature contrast.

Each plan is a smooth circular arc of fixed length L=1000 m at altitude 25 m (horizontal, no
climb), swept by total turn angle Phi in [0, 180] deg. For a circular arc of length L and turn
Phi, radius R = L/Phi and mean curvature kappa = Phi/L (1/m); Phi=0 is the straight baseline
(= the v2 control geometry). N nav waypoints trace the arc so the commanded path has
distributed, near-constant curvature rather than a single corner.

  usage: make_arc_plans.py <out_plans_dir> [--step-deg 3] [--nwp 12] [--max-deg 180]
"""
import argparse
import json
import math
from pathlib import Path

LAT0, LON0, ALT = 47.3979711, 8.5461637, 25.0
HOME_AMSL = 488.0
R_EARTH = 6378137.0
L = 1000.0  # constant arc length (m) -- MUST match the v2 straight route length


def enu_to_latlon(x, y):
    lat = LAT0 + math.degrees(y / R_EARTH)
    lon = LON0 + math.degrees(x / (R_EARTH * math.cos(math.radians(LAT0))))
    return lat, lon


def arc_points(turn_deg, nwp):
    """N points along a left-curving arc of length L, starting at origin heading east."""
    phi = math.radians(turn_deg)
    pts = []
    for i in range(1, nwp + 1):
        s = L * i / nwp
        if phi < 1e-9:
            x, y = s, 0.0
        else:
            R = L / phi
            th = s / R
            x = R * math.sin(th)
            y = R * (1.0 - math.cos(th))
        pts.append(enu_to_latlon(x, y))
    return pts


def item(command, lat, lon, jump):
    return {
        "AMSLAltAboveTerrain": None, "Altitude": ALT, "AltitudeMode": 1,
        "autoContinue": True, "command": command, "doJumpId": jump, "frame": 3,
        "params": [0, 0, 0, None, lat, lon, ALT], "type": "SimpleItem",
    }


def build_plan(turn_deg, nwp):
    items = [item(22, LAT0, LON0, 1)]  # takeoff at origin, exactly like the existing plans
    for i, (lat, lon) in enumerate(arc_points(turn_deg, nwp), start=2):
        items.append(item(16, lat, lon, i))
    return {
        "fileType": "Plan", "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 5, "firmwareType": 12, "globalPlanAltitudeMode": 1,
            "hoverSpeed": 5, "items": items,
            "plannedHomePosition": [LAT0, LON0, HOME_AMSL],
            "vehicleType": 2, "version": 2,
        },
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2}, "version": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--step-deg", type=int, default=3)
    ap.add_argument("--nwp", type=int, default=12)
    ap.add_argument("--max-deg", type=int, default=180)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    levels = list(range(0, args.max_deg + 1, args.step_deg))
    for deg in levels:
        plan = build_plan(deg, args.nwp)
        name = "arc_a%03d" % deg
        (out / (name + ".plan")).write_text(json.dumps(plan, indent=2))
    kmin = math.radians(levels[1]) / L if len(levels) > 1 else 0
    kmax = math.radians(levels[-1]) / L
    print("wrote %d arc plans (arc_a000..arc_a%03d) to %s" % (len(levels), levels[-1], out))
    print("  L=%.0f m, nwp=%d, kappa 0..%.5f 1/m (R %.0f m at max), turn 0..%d deg"
          % (L, args.nwp, kmax, L / math.radians(levels[-1]) if levels[-1] else 0, levels[-1]))


if __name__ == "__main__":
    main()
