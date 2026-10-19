#!/usr/bin/env python3
"""Check OpenArm LeRobot dataset metadata before a comparison or merge.

Accepts local dataset roots or Hugging Face dataset repo IDs. Only meta/info.json is
downloaded for repo IDs. This checks metadata compatibility, not motor calibration,
camera alignment, image content, or episode quality.
"""

import argparse
import json
from pathlib import Path


def load_info(source: str) -> dict:
    local = Path(source).expanduser()
    path = local / "meta" / "info.json"
    if path.is_file():
        return json.loads(path.read_text())
    if local.exists():
        raise ValueError(f"{source}: no meta/info.json")
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(source, "meta/info.json", repo_type="dataset")).read_text())


def signature(feature: dict | None) -> tuple | None:
    if feature is None:
        return None
    return (feature.get("dtype"), tuple(feature.get("shape", ())), tuple(feature.get("names") or ()))


def compare(real: dict, sim: dict) -> list[str]:
    problems = []
    for field in ("codebase_version", "fps", "robot_type"):
        if real.get(field) != sim.get(field):
            problems.append(f"{field}: real={real.get(field)!r}, sim={sim.get(field)!r}")

    required = ("action", "observation.state")
    real_features = real.get("features", {})
    sim_features = sim.get("features", {})
    for key in required:
        a, b = signature(real_features.get(key)), signature(sim_features.get(key))
        if a is None or b is None:
            problems.append(f"{key}: missing from {'real' if a is None else 'sim'} dataset")
        elif a != b:
            problems.append(f"{key}: real={a!r}, sim={b!r}")

    real_cams = {k for k in real_features if k.startswith("observation.images.")}
    sim_cams = {k for k in sim_features if k.startswith("observation.images.")}
    if not real_cams or not sim_cams:
        problems.append("both datasets must have at least one observation.images.* camera")
    if real_cams != sim_cams:
        problems.append(f"camera keys: real={sorted(real_cams)}, sim={sorted(sim_cams)}")
    for key in sorted(real_cams & sim_cams):
        a, b = signature(real_features[key]), signature(sim_features[key])
        if a != b:
            problems.append(f"{key}: real={a!r}, sim={b!r}")
    for label, info in (("real", real), ("sim", sim)):
        if info.get("total_episodes", 0) < 1 or info.get("total_frames", 0) < 1:
            problems.append(f"{label}: no completed episodes/frames in meta/info.json")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="Local dataset root or HF dataset repo ID")
    parser.add_argument("--sim", required=True, help="Local dataset root or HF dataset repo ID")
    args = parser.parse_args()
    try:
        real, sim = load_info(args.real), load_info(args.sim)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Could not load dataset metadata: {exc}\n")

    problems = compare(real, sim)
    if problems:
        print("Schema check failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Schema check passed: LeRobot version, fps, robot type, action/state layout and camera layout match.")
    print("This does not check task identity, calibration, image appearance, contact or episode quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
