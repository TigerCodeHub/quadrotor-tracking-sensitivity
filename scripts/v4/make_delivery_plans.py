"""Generate delivery-route .plan files anchored to published delivery-UAV mission profiles.

Three archetypes, each an out-and-back sortie (every commercial operator flies out-and-back;
multi-stop exists only in the VRP literature and is included as the explicitly-extrapolated
archetype C). Geometry anchors, with sources:

  A "urban last-mile" (single stop, short):
    - one-way length 2,000 m: Rodrigues et al. 2022 base route (2 km one-way, quadrotor,
      188 empirical flights; https://pmc.ncbi.nlm.nih.gov/articles/PMC9403403/); Manna Aero
      service radius 2-3 km (https://dronelife.com/2022/04/25/manna-in-the-u-s/).
    - cruise altitude 60 m AGL: Manna 50-65 m
      (https://www.suasnews.com/2026/02/not-your-hobby-drone-manna-aero/); Amazon MK30
      outbound 61 m (FAA Written Re-Evaluation, Pendleton OR, 2025).
    - delivery: descend to 15 m, hover 30 s, tether drop, climb back: Manna ~15 m hover-drop
      (https://dronexl.co/2025/04/12/manna-aero-drone-delivery-revolution/); 30 s from the
      DJI FlyCart 30 winch cycle (20 m cable at 0.8 m/s, https://www.dji.com/flycart-30/specs).
    - cruise speed 10 m/s (design column, not encoded here): Matternet M2 cruise 10 m/s
      (https://www.matternet.com/our-system-aircraft); Rodrigues base 12 m/s.

  B "suburban" (single stop, long):
    - one-way length 8,000 m: Wing service radius ~9.7 km one-way
      (https://www.unmannedsystemstechnology.com/2024/01/wing-new-delivery-drone/); Antwork
      urban air-route network mean route ~10 km
      (https://www.sciencedirect.com/science/article/pii/S1366554522002526). 8 km sits just
      inside both while bounding sim cost.
    - cruise altitude 100 m AGL: Rodrigues base 100 m; Matternet 50-100 m; Zipline P2 >=91 m
      (https://www.businesswire.com/news/home/20230315005438/en). All under the FAA Part 107
      Sec. 107.51 ceiling of 120 m.
    - delivery: descend to 5 m, hover 20 s: Amazon MK30 drops from ~3.7 m
      (https://fpvcopilot.com/amazon-mk30-drone-delivery-specs/); Wing tethers from ~7 m
      (https://wing.com/technology). 5 m splits the pair.
    - cruise speed 15 m/s (design column): DJI FlyCart 30 auto-route max 15 m/s
      (https://www.dji.com/flycart-30/specs); below Manna's 17 m/s.

  C "multi-stop" (EXTRAPOLATED -- no commercial operator flies multi-stop; the archetype
    follows energy-aware VRP models, e.g. https://arxiv.org/html/2509.15830):
    - 3 stops, legs 3,000 / 2,000 / 1,500 m plus direct return (~13 km total incl.
      per-stop verticals), inside
      archetype B's battery envelope.
    - cruise altitude 80 m AGL (interpolated A..B), speed 12 m/s (Rodrigues base).
    - per-stop delivery maneuver = archetype A's (15 m, 30 s, Manna-style).

Route shape: legs carry mid-route doglegs (routes follow street/airspace corridors, not
geodesics -- Antwork network study above). Three shape variants per archetype. Course
bearing is NOT varied: v3-B b1 showed heading main effects are ~0 (isotropy) and wind
direction is a design column, so course-relative wind is swept there; the base course stays
due east for comparability with the v2/v3 straight controls.

Vertical profile: the takeoff item stays at 25 m (identical to v2/v3, keeping the validated
1x-speed takeoff path and its MIS_TAKEOFF_ALT workarounds); the first route waypoint climbs
to cruise altitude at route (6x) speed. Climb/descent rates come from PX4 MPC_Z_VEL_* (see
the design; Rodrigues publishes 2.5 up / 2.0 down m/s).

NOTE sim.py flies items itself via goto_location (no mission upload): command 16 = waypoint,
command 19 params[0] = hover seconds. Plan cruiseSpeed is IGNORED; per-archetype speed must
be applied as MPC_XY_CRUISE via the design (columns cruise_speed_m_s etc.).

  usage: make_delivery_plans.py <out_plans_dir>
"""
import json
import math
import sys
from pathlib import Path

LAT0, LON0 = 47.3979711, 8.5461637
HOME_AMSL = 488.0
R_EARTH = 6378137.0
TAKEOFF_ALT = 25.0  # matches v2/v3 plans; route then climbs to cruise altitude


def enu_to_latlon(x, y):
    lat = LAT0 + math.degrees(y / R_EARTH)
    lon = LON0 + math.degrees(x / (R_EARTH * math.cos(math.radians(LAT0))))
    return lat, lon


def item(command, x, y, alt, jump, hold_s=0.0):
    lat, lon = enu_to_latlon(x, y)
    return {
        "AMSLAltAboveTerrain": None, "Altitude": alt, "AltitudeMode": 1,
        "autoContinue": True, "command": command, "doJumpId": jump, "frame": 3,
        "params": [hold_s, 0, 0, None, lat, lon, alt], "type": "SimpleItem",
    }


# Max straight-line spacing between consecutive route waypoints. sim.py flies each item as
# one goto_location with its own per-waypoint timeout (default 360 s wall). The realized sim
# real-time factor on these long, fast delivery routes is ~1.5x (well below v2's short-route
# ~4.5x), so an 8 km leg flown as a single goto does not finish inside 360 s and times out on
# the return -- the v4 canary failed exactly there (waypoint N/N still 1.4-3.8 km out, closing,
# at the deadline). Densifying every leg to <=2 km pass-through waypoints gives each goto its
# own budget (2 km / 15 m/s = 133 sim-s ~= 90 s wall even at 1.5x) without changing the cited
# route geometry (identical endpoints, altitudes, speeds -- just more intermediate points).
MAX_LEG_M = 2000.0


def densify(x0, y0, x1, y1):
    """Intermediate + end points from (x0,y0) to (x1,y1), each hop <= MAX_LEG_M."""
    length = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, math.ceil(length / MAX_LEG_M))
    return [(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps) for i in range(1, steps + 1)]


def dogleg(x0, y0, x1, y1, offsets):
    """Waypoints from (x0,y0) to (x1,y1) with lateral dogleg offsets at even spacing.

    offsets: perpendicular displacements (m) applied at 1/(n+1), 2/(n+1), ... of the leg,
    modelling street/airspace-corridor routing. Returns intermediate points + endpoint.
    """
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    pts = []
    n = len(offsets)
    for i, off in enumerate(offsets, start=1):
        f = i / (n + 1)
        pts.append((x0 + dx * f + px * off, y0 + dy * f + py * off))
    pts.append((x1, y1))
    return pts


def delivery_maneuver(x, y, cruise_alt, drop_alt, hold_s, jump):
    """Descend to drop_alt at (x,y), hover hold_s, climb back to cruise_alt."""
    return [
        item(16, x, y, drop_alt, jump),
        item(19, x, y, drop_alt, jump + 1, hold_s=hold_s),
        item(16, x, y, cruise_alt, jump + 2),
    ]


# Shape variants: dogleg offsets (m) for the outbound leg(s); return retraces mirrored.
# v0 = straight corridor, v1/v2 = one- and two-bend corridors (magnitudes ~5-10% of leg).
ARCHETYPES = {
    "dlv_a": {  # urban last-mile: 2 km out, stop, 2 km back @ 60 m
        "cruise_alt": 60.0, "drop_alt": 15.0, "hold_s": 30.0,
        "stops": [(2000.0, 0.0)],
        "variants": {"v0": [[]], "v1": [[150.0]], "v2": [[120.0, -120.0]]},
    },
    "dlv_b": {  # suburban: 8 km out, stop, 8 km back @ 100 m
        "cruise_alt": 100.0, "drop_alt": 5.0, "hold_s": 20.0,
        "stops": [(8000.0, 0.0)],
        "variants": {"v0": [[]], "v1": [[500.0]], "v2": [[400.0, -400.0]]},
    },
    "dlv_c": {  # multi-stop (extrapolated): 3 stops, ~10 km total @ 80 m
        "cruise_alt": 80.0, "drop_alt": 15.0, "hold_s": 30.0,
        "stops": [(3000.0, 0.0), (4400.0, 1400.0), (5900.0, 1400.0)],
        "variants": {
            "v0": [[], [], []],
            "v1": [[200.0], [-150.0], []],
            "v2": [[150.0, -150.0], [120.0], [-100.0]],
        },
    },
}


def build_plan(spec, leg_offsets):
    cruise = spec["cruise_alt"]
    items = [item(22, 0.0, 0.0, TAKEOFF_ALT, 1)]
    jump = 2
    items.append(item(16, 0.0, 0.0, cruise, jump))  # climb to cruise over the pad
    jump += 1
    x, y = 0.0, 0.0
    route_len = 0.0
    for (sx, sy), offsets in zip(spec["stops"], leg_offsets):
        for wx, wy in dogleg(x, y, sx, sy, offsets):
            for px, py in densify(x, y, wx, wy):  # <=MAX_LEG_M pass-through waypoints
                items.append(item(16, px, py, cruise, jump))
                jump += 1
                route_len += math.hypot(px - x, py - y)
                x, y = px, py
        for entry in delivery_maneuver(x, y, cruise, spec["drop_alt"], spec["hold_s"], jump):
            items.append(entry)
            jump += 1
        route_len += 2.0 * (cruise - spec["drop_alt"])
    # direct return corridor (mirror of v0), ending over the pad; RTL/land handled by sim.py
    for px, py in densify(x, y, 0.0, 0.0):
        items.append(item(16, px, py, cruise, jump))
        jump += 1
        route_len += math.hypot(px - x, py - y)
        x, y = px, py
    return items, route_len


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in sorted(ARCHETYPES.items()):
        for variant, leg_offsets in sorted(spec["variants"].items()):
            items, route_len = build_plan(spec, leg_offsets)
            plan = {
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
            path = out / f"{name}_{variant}.plan"
            path.write_text(json.dumps(plan, indent=2))
            hover = sum(1 for entry in items if entry["command"] == 19) * spec["hold_s"]
            print(f"{path.name}: {len(items)} items, ~{route_len:.0f} m route, "
                  f"{hover:.0f} s hover, cruise {spec['cruise_alt']:.0f} m")


if __name__ == "__main__":
    main()
