# Third-Party Notices

This project uses or derives simulation assets from the following upstream
projects. Upstream names identify provenance and do not imply endorsement.

## PX4 X500

- Project: PX4 Gazebo Models
- Source: https://github.com/PX4/PX4-gazebo-models
- Pinned commit: `b6127f4ec20de867e215fb5f78ae88b80f371909`
- Intended assets: `models/x500_base`
- License: BSD 3-Clause; see `LICENSES/PX4-X500-BSD-3-Clause.txt`
- Copyright: 2022 Rudis Laboratories

> Reconciled 2026-10-08 (audit #23). The machine-readable catalog
> `generated/airframes.csv` is the single source of truth for which upstream commit
> each airframe was generated from; its `source_url` for `px4_x500` pins commit
> `b6127f4e…`, so this notice is aligned to it (an earlier `e05f4312…` value here was
> stale). If you have the generation log, verify against the fetched model hash.

## CTU MRS Airframes

- Project: MRS UAV Gazebo Simulator
- Source: https://github.com/ctu-mrs/mrs_uav_gazebo_simulator
- Branch: `ros2`
- Pinned commit: `ec4f6f554dfeaf79e12dd2b76d627adb7a155bac`
- Intended airframes: F330, F450, T650, and M690
- License: BSD 3-Clause; see `LICENSES/MRS-BSD-3-Clause.txt`
- Copyright: 2023 Multi-robot Systems (MRS) group at Czech Technical University in Prague

Some MRS SDF/Jinja component files contain Apache-2.0 headers. Their headers
are retained and the license is preserved at `LICENSES/Apache-2.0.txt`. The
model generator also copies that license beside generated MRS resources. The
importer modifies the rendered SDF motor-plugin interface; it does not edit the
upstream Jinja source files.

The upstream licenses do not grant trademark rights. Product and project names
are used only to identify the simulation models and their provenance.
