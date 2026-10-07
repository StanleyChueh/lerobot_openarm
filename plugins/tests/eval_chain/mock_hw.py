"""Mocked follower (perfect tracking + fake camera frames) shared by the chain scripts. No CAN traffic."""
import time
from unittest import mock

import numpy as np

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
import robots.umeow_openarm_follower.openarm_follower as fol

fol.oa.OpenArm = mock.MagicMock()
KEYS = [f"{p}J{i}.pos" for i in range(1, 9) for p in ("R", "L")]
FAKE = {"state": {k: 0.0 for k in KEYS}, "sends": []}
CAMS = ("body_cam", "wrist_cam", "right_wrist_cam")
CAMERAS_ARG = "{" + ", ".join(f"{c}: {{type: opencv, index_or_path: {90 + i}, width: 64, height: 48, fps: 30}}"
                              for i, c in enumerate(CAMS)) + "}"


def _obs(self):
    out = dict(FAKE["state"])
    t = time.perf_counter()
    for i, c in enumerate(self.config.cameras):
        img = np.zeros((48, 64, 3), np.uint8)
        img[..., i % 3] = int(127 + 120 * np.sin(t + i))  # a changing colour, so videos are not constant
        out[c] = img
    return out


def _send(self, action, vel):
    FAKE["sends"].append((time.perf_counter(), {k: float(action[k]) for k in KEYS}))
    FAKE["state"] = {k: float(action[k]) for k in KEYS}
    return action


fol.OpenArmFollower.connect = lambda self, calibrate=False: setattr(self, "_is_connected", True)
fol.OpenArmFollower.disconnect = lambda self: setattr(self, "_is_connected", False)
fol.OpenArmFollower.get_observation = _obs
fol.OpenArmFollower.send_action = _send
