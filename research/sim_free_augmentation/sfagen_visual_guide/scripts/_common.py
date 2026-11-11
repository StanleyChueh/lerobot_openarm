"""Shared helpers for the diagnostic-only visual-guide scripts.

Rules these scripts follow (see ../README section "Execution policy"):
- read-only use of the sfagen work directories (caches, candidates, renders, built datasets); nothing is written there;
- sfagen code (commit 50832b3) and the frozen v1 `sfa/` code are imported, never modified;
- every run writes diagnostics/run_<name>.json with command, Git SHA, source revision, device, peak GPU memory, wall time,
  input/output SHA-256 and whether each number came from an existing cache or a new diagnostic computation."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1]
REPO = GUIDE.parents[2]
SFAGEN = Path("/home/stanley/Stanley_ws/sfa_hf_export/sfagen")          # working copy, byte-identical to 50832b3 sfagen/
sys.path.insert(0, str(SFAGEN))
PY = "env -u PYTHONPATH -u LD_LIBRARY_PATH /home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python"

TASKS = {
    "pringles": {"config": SFAGEN / "configs/openarm_pringles_handover.yaml", "work": SFAGEN / "configs/work/openarm_pringles_handover",
                 "source_episode": 0, "export_episode": 6, "variation": "interact_x+5_y+0_z+0",
                 "repo": "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00", "repo_rev": "30533e672dc16c1a35f480cb56beb5882b7e3db9",
                 "source_repo": "ethanCSL/openarm_pringles_lerobot_real_10_v00", "source_rev": "cebdee98115ab6827a4ec2655335c536b5dd085b"},
    "battery": {"config": SFAGEN / "configs/openarm_battery_insertion.yaml", "work": SFAGEN / "configs/work/openarm_battery_insertion",
                "source_episode": 1, "export_episode": 14, "variation": "battery_x+6_y+0",
                "repo": "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00", "repo_rev": "c3c571d0d2474a2b64a4de412209b21d3bfff913",
                "source_repo": "ethanCSL/openarm_insertbattery_lerobot_real_10_v02", "source_rev": "a1c29d77dabcaf5a0a9ff658ba742a2d941e1570"},
}


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


class Run:
    """with Run('name', inputs=[...]) as r: ...; r.out(path); r.note(key=value)"""
    def __init__(self, name, inputs=(), device="cuda:0"):
        self.name, self.inputs, self.device, self.outputs, self.notes = name, [Path(p) for p in inputs], device, [], {}

    def __enter__(self):
        self.t0 = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def out(self, p):
        self.outputs.append(Path(p)); return p

    def note(self, **kw):
        self.notes.update(kw)

    def __exit__(self, *exc):
        import torch
        gpu = {}
        if torch.cuda.is_available():
            gpu = {"device_name": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "peak_torch_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
        rec = {"name": self.name, "command": " ".join([PY.split()[-1]] + sys.argv), "cwd": os.getcwd(), "git_sha": git_sha(),
               "sfagen_code": "sfa_hf_export/sfagen/sfagen (byte-identical to research/sim_free_augmentation/sfagen/sfagen @ 50832b3)",
               "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)), "wall_time_s": round(time.time() - self.t0, 2), "gpu": gpu,
               "inputs": {str(p): sha256(p) for p in self.inputs if p.exists() and p.is_file()},
               "outputs": {str(p.relative_to(GUIDE)) if GUIDE in p.parents else str(p): sha256(p) for p in self.outputs if p.exists()},
               "notes": self.notes, "error": None if exc[0] is None else repr(exc[1])}
        (GUIDE / "diagnostics").mkdir(exist_ok=True)
        (GUIDE / "diagnostics" / f"run_{self.name}.json").write_text(json.dumps(rec, indent=1, default=str))
        return False
