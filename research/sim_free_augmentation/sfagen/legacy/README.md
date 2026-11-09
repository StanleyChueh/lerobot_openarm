# Superseded task-specific pipelines (reference only)

These made the **previous** Hub revisions (now under tags `v04-65ep-two-pipelines`, `v01-65ep-two-pipelines`, ...). They are kept
because the cross-pipeline benchmark (`../tools/xbench.py`) renders with them as the "published renderer" baseline.
They are not maintained; use `../sfagen` instead. Original location on the workstation: `~/Stanley_ws/sfa_hf_export/<dir>`.

- `render_v4/`  mesh-guided warping renderer (MGW) v1, Pringles replay checks, calibration attempts
- `3cam_v3/`, `3cam_v4/`  Pringles synthetic-only builds (frozen v3.1 planner trajectories + MGW)
- `battery_bsfa/`  battery v00 reconstruction / planning / rendering
- `target/`  battery v01 additions: camera refit, station proxy, one-arm planner, target-position analysis
- `battery_v01/`  battery v01 build / render / cards
