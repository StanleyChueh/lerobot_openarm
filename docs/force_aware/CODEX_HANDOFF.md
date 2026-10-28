# Handoff: force-aware telemetry, Gate 1

**Branch:** `force-aware/telemetry-gate1`, from `official-lerobot-quest` @ `9abbe12`. **Status:** stopped at
Gate 1. No real-robot run has been made or approved.

## What exists

* `plugins/lerobot_robot_openarm_umeow/lerobot_robot_openarm_umeow/telemetry.py`: a read-only sidecar. A
  receive-only CAN tap per arm plus `Motor` getters, one row per `get_observation()`, written as Parquet to
  `--robot.telemetry_dir`. Off by default.
* Hooks in `openarm_umeow.py` (`_init_umeow`, `_hw_connect`, `get_observation`, `_hw_disconnect`) and one
  config field (`telemetry_dir`). The follower, `can_monitor.py`, gains, guards and the 16-D schema are
  untouched.
* Contract: `docs/force_aware/TELEMETRY_SCHEMA.md` (`openarm_telemetry/1`, 340 columns).
* Evidence: `HARDWARE_AUDIT.md`. Research plan: `CONTACT_FEASIBILITY.md`.

## Tests (offline, no CAN socket; run from the repo root)

```bash
R="env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 HF_HUB_OFFLINE=1 .venv/bin/python"
$R plugins/tests/test_telemetry.py <scratch>/u          # codec vs installed openarm_can, ordering, freshness, faults, schema, overhead, wrapper
$R plugins/tests/test_telemetry_record.py <scratch>/e2e  # official lerobot-record + dataset alignment
```

The existing suites must stay green: `test_robot_guard`, `test_return_lag`, `test_quest_safety`,
`test_teleoperate_mock` (no args) and `test_record_mock`, `test_record_gate`, `test_rerun_status` (scratch
dir arg). Tests patch `telemetry.make_tap`. Never let a test construct a real `WireTap`: can0/can1 are up on
the lab PC.

## Consuming it from the training repo

You need only `pandas` + `pyarrow`. Join by name, never by index (`openarm_umeow` datasets interleave
R/L). `align_to_dataset` is about 15 lines; copy it instead of importing the plugin. Row k+1 holds the
command for action k. Honour NaN = unknown and `stale` = 1 = don't trust.

## Do not

* run anything on the real arm, query motor parameters, re-zero, or change gains without explicit approval;
* add CAN transmissions to `telemetry.py` (check 11 enforces this);
* put telemetry into the policy observation (a later phase);
* claim force in Newtons while `provenance.torque_calibration` is `null`.

## Next (in order, each gated)

1. Gate-1 bench run: `HARDWARE_AUDIT.md` §8, steps 1–2. It answers items 5, 7, 8, 9, 10 and 11 there.
2. An offline residual baseline (CPU) on D2/D3 telemetry: `CONTACT_FEASIBILITY.md` §5.
3. A known gap: an episode where no joint changes cannot be aligned by value. If that matters, add an
   episode marker (e.g. via the Quest record gate's `begin_episode` hook), which changes the recording path
   and needs review.
