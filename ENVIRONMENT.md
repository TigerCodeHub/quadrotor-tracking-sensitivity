# Environment

Software and model versions behind the simulation campaigns and the analyses.

## Analysis environment (produces every number in the paper)
- Python 3.12 with the versions pinned in `requirements.txt`:
  numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, matplotlib 3.10.8, SALib 1.6.0.

## SITL stack (data generation)
- **PX4-Autopilot v1.16.2** (`git describe` = `v1.16.2-4-g9c2a2985a1`, sha `9c2a2985a1`), SITL target
  `px4_sitl_default`.
- **Gazebo Harmonic: gz-sim 8.14.0** (gz-transport 13.5.0, gz-msgs 10.3.2), lockstep.
  The coordinated point release gz-sim 8.15.0 (gz-transport 13.6.0, gz-msgs 10.4.0) breaks the PX4
  v1.16.2 SITL lockstep handshake (flights stall at PX4 startup); pin the 8.14 packages
  (`apt-mark hold`). After a fresh PX4 build, run PX4's stock sim once
  (`make px4_sitl gz_x500`) so `build/px4_sitl_default/rootfs` is populated before batch runs.
- Flight control through MAVSDK-Python; ULogs parsed with pyulog.
- Custom Dryden-form gust plugin: `gz_plugins/dryden_wind/` (MIL-STD-1797A forming filters).
  **Force model:** `F = mass · forceScaling · (v_wind − v_link)` (`DrydenWind.cc`), i.e.
  proportional to mass and linear in relative wind velocity, applied independently per link; it is
  not a quadratic aerodynamic drag. See the paper's Limitations section.

## Airframe models
- Catalog and upstream commit per airframe: `generated/airframes.csv` (the catalog the generator
  reads). x500 from PX4-gazebo-models at `b6127f4e…`; f330/f450/t650/m690 from CTU-MRS
  `mrs_uav_gazebo_simulator` at `ec4f6f55…`. Licenses: `THIRD_PARTY_NOTICES.md`, `LICENSES/`.
- **As-modified, not as-published:** the generator edits the upstream SDFs (mass, inertia and CoM
  on the first link; rotor-speed limit; wind enabled on all links), so the configurations are
  controlled rigid-body designs, not buildable variants of the named vehicles.
