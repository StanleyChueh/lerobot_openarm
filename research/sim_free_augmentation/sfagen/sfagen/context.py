"""Loads a task config (YAML) and builds the shared objects: source dataset, robot, camera (calibrated if available)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, yaml
from .camera import Camera
from .data import Source
from .robot import Robot


class Context:
    def __init__(self, config_path, work_dir=None):
        self.config_path = Path(config_path).resolve(); self.base = self.config_path.parent
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.work = Path(work_dir or self.cfg.get("work_dir", self.base / "work" / self.config_path.stem)).resolve()
        self.work.mkdir(parents=True, exist_ok=True); self.cache_dir = self.work / "cache"; self.cache_dir.mkdir(exist_ok=True)
        self.src = Source(self.cfg["source"]); self.robot = Robot(self.cfg["robot"], self.base)
        self.cam0 = Camera(self.cfg["camera"])
        cal = self.work / "calibration.json"
        self.calib = json.loads(cal.read_text()) if cal.exists() else None
        self.cam = self.cam0.perturbed(self.calib["params7"]) if self.calib and self.calib.get("adopted", True) else self.cam0
        self._table = None

    @property
    def lowest_arm_z(self):
        """lowest arm point over the source demos (cached): an upper bound of the table height."""
        p = self.work / "lowest_arm.json"
        if not p.exists():
            from .checks import Checker
            ck = Checker(self); z = min(float(ck.lowest(self.src.arrays(e)[0]).min()) for e in self.episodes())
            p.write_text(json.dumps({"z": z}))
        return json.loads(p.read_text())["z"]

    @property
    def table_fixed(self):
        return "table_height" in self.cfg or (self.work / "table_height.json").exists()

    @property
    def table_height(self):
        """table plane z of this rig: config value, else `sfagen table` result (table_height.json), else (before that
        step) the lowest arm point - 5 mm as the starting value."""
        if self._table is None:
            if "table_height" in self.cfg:
                self._table = float(self.cfg["table_height"])
            elif (self.work / "table_height.json").exists():
                self._table = json.loads((self.work / "table_height.json").read_text())["z"]
            else:
                return self.lowest_arm_z - 0.005
        return self._table

    def episodes(self):
        e = self.cfg["source"].get("episodes", "all")
        return self.src.episode_ids if e == "all" else list(e)
