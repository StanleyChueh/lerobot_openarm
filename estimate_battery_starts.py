"""Estimate charger and loose-AA start poses in real battery episodes from body frames.

The calibrated body camera projects bright-object masks to the object-height planes. The
charger centre has a bias correction measured by applying the same mask to a known sim pose.
These are centimetre-scale initial estimates for replay, not measured CAD geometry.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation

REPO = "ethanCSL/openarm_insertbattery_lerobot_real_v01"
CAMERA_POS = np.array([0.05621, -0.00305, 0.83062])
CAMERA_ROT = Rotation.from_quat([-0.68244, 0.69543, -0.12490, 0.18720]).as_matrix()
FOCAL_PX = 607.22
OBJECTS = {
    "charger": {"roi": (210, 390, 135, 345), "z": 0.3175, "bias": (0.00986, 0.00107), "min_area": 1500},
    "battery": {"roi": (390, 540, 130, 345), "z": 0.28725, "bias": (-0.00023, -0.00038), "min_area": 400},
}


def pixels_to_xy(u, v, z):
    ray = np.stack([(u - 320) / FOCAL_PX, (v - 240) / FOCAL_PX, np.ones_like(u)], axis=-1) @ CAMERA_ROT.T
    return CAMERA_POS[:2] + ray[..., :2] * ((z - CAMERA_POS[2]) / ray[..., 2:3])


def pose_from_frame(image, name):
    cfg = OBJECTS[name]
    x0, x1, y0, y1 = cfg["roi"]
    bright = ((image[:, :, 0] > 100) & (image[:, :, 1] > 90) & (image[:, :, 2] > 80)).astype(np.uint8)
    cropped = bright[y0:y1, x0:x1]
    n, labels, stats, centres = cv2.connectedComponentsWithStats(cropped)
    if n < 2:
        raise ValueError(f"{name}: no bright component")
    component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < cfg["min_area"]:
        raise ValueError(f"{name}: bright component only {area} px")
    u, v = centres[component] + [x0, y0]
    center = pixels_to_xy(np.array(u), np.array(v), cfg["z"]) + cfg["bias"]
    ys, xs = np.nonzero(labels == component)
    # Fit the long axis in world x/y; a slot and an AA cell both lie along local +x.
    points = pixels_to_xy(xs.astype(float) + x0, ys.astype(float) + y0, cfg["z"])
    covariance = np.cov(points.T)
    axis = np.linalg.eigh(covariance)[1][:, -1]
    yaw = np.degrees(np.arctan2(axis[1], axis[0]))
    yaw = ((yaw + 90) % 180) - 90
    return [float(center[0]), float(center[1]), float(yaw)], {"uv": [float(u), float(v)], "area": area}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    starts, diagnostics = {}, {}
    for file_index in range(2):
        meta = pd.read_parquet(hf_hub_download(REPO, f"meta/episodes/chunk-000/file-{file_index:03d}.parquet",
                                               repo_type="dataset"))
        targets = {3 + sum(int(x) for x in meta.length.iloc[:i]): int(row.episode_index)
                   for i, (_, row) in enumerate(meta.iterrows())}
        video = hf_hub_download(REPO, f"videos/observation.images.body_cam/chunk-000/file-{file_index:03d}.mp4",
                                repo_type="dataset")
        with av.open(video) as container:
            for k, frame in enumerate(container.decode(video=0)):
                if k in targets:
                    ep = targets[k]
                    image = frame.to_ndarray(format="rgb24")
                    starts[str(ep)], diagnostics[str(ep)] = {}, {}
                    for name in OBJECTS:
                        try:
                            starts[str(ep)][name], diagnostics[str(ep)][name] = pose_from_frame(image, name)
                        except ValueError as error:
                            diagnostics[str(ep)][name] = {"error": str(error)}
                    if len(starts[str(ep)]) != len(OBJECTS):
                        del starts[str(ep)]
                if k >= max(targets):
                    break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(starts, indent=2))
    out.with_name(out.stem + "_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Estimated both objects in {len(starts)}/9 episodes -> {out}")
    for ep, poses in starts.items():
        print(ep, poses)
    print("Missing:", sorted(set(map(str, range(9))) - starts.keys()))


if __name__ == "__main__":
    main()
