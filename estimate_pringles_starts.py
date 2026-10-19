"""Estimate can starts from the yellow lid in each real body-camera episode.

Uses the calibrated IsaacLab body camera and the known can top height. It produces
an approximate starting pose for replay; inspect the diagnostics before relying on it.
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

REPO = "ethanCSL/openarm_pringles_lerobot_real_v00"
CAMERA_POS = np.array([0.05621, -0.00305, 0.83062])
CAMERA_ROT = Rotation.from_quat([-0.68244, 0.69543, -0.12490, 0.18720]).as_matrix()
FOCAL_PX = 607.22
CAN_TOP_Z = 0.480


def can_lid(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([10, 35, 90]), np.array([40, 255, 255]))
    mask[:70] = 0
    mask[380:] = 0
    mask[:, :190] = 0
    mask[:, 480:] = 0
    _, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    if len(stats) < 2:
        raise ValueError("no yellow lid candidate")
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 1500:
        raise ValueError(f"yellow lid candidate too small: {area} pixels")
    return centres[i], area


def pixel_to_xy(pixel):
    u, v = pixel
    ray = CAMERA_ROT @ np.array([(u - 320) / FOCAL_PX, (v - 240) / FOCAL_PX, 1.0])
    point = CAMERA_POS + ray * (CAN_TOP_Z - CAMERA_POS[2]) / ray[2]
    return [float(point[0]), float(point[1]), 0.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    meta = pd.read_parquet(hf_hub_download(REPO, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset"))
    videos = hf_hub_download(REPO, "videos/observation.images.body_cam/chunk-000/file-000.mp4", repo_type="dataset")
    targets = {int(row.dataset_from_index) + 3: int(row.episode_index) for _, row in meta.iterrows()}
    starts, diagnostics = {}, {}
    with av.open(videos) as container:
        for k, frame in enumerate(container.decode(video=0)):
            if k in targets:
                ep = targets[k]
                try:
                    pixel, area = can_lid(frame.to_ndarray(format="rgb24"))
                    xy = pixel_to_xy(pixel)
                    starts[str(ep)] = {"can": xy}
                    diagnostics[str(ep)] = {"lid_uv": [float(v) for v in pixel], "mask_area": area, "can_xy": xy[:2]}
                except ValueError as error:
                    diagnostics[str(ep)] = {"error": str(error)}
            if k >= max(targets):
                break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(starts, indent=2))
    out.with_name(out.stem + "_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Estimated {len(starts)}/{len(targets)} can starts -> {out}")
    if len(starts) != len(targets):
        print("Missing episodes:", sorted(set(targets.values()) - {int(x) for x in starts}))
    values = np.array([v["can"][:2] for v in starts.values()])
    print("x/y range:", values.min(axis=0), values.max(axis=0))


if __name__ == "__main__":
    main()
