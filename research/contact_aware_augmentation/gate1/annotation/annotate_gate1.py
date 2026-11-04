"""Human keypoint annotation for the Gate 1 pilot (body_cam, Pringles real_10_v00 eps 0, 4, 8). Blind: the tool never
shows tracker output, FK projections or previous annotators' clicks.

    # 1) write the frame manifest (once; deterministic)
    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/contact_aware_augmentation/gate1/annotation/annotate_gate1.py \
        manifest --out research/contact_aware_augmentation/outputs/gate1/annotation/annotation_manifest.csv
    # 2) annotate (GUI; resumable; one CSV per annotator)
    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/contact_aware_augmentation/gate1/annotation/annotate_gate1.py \
        annotate --annotator A1 --manifest research/contact_aware_augmentation/outputs/gate1/annotation/annotation_manifest.csv \
        --out research/contact_aware_augmentation/outputs/gate1/annotation/annotations_A1.csv

Keys (see ANNOTATION_INSTRUCTIONS.md): left click = place the current keypoint (VISIBLE); `u` then click = place as
UNCERTAIN (excluded from ground truth); `o` = OCCLUDED / out of frame (no click); `z` = undo the last keypoint;
`n` or space = next frame (only when every keypoint of the frame is answered); `p` = previous frame; `q` = save and quit.
The CSV is rewritten after every frame, so quitting never loses finished frames.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

KEYPOINTS = [
    ("lid_center", "centre of the can's lid disk (the pale circular top); estimate the centre from the rim"),
    ("right_wrist_tag", "centre of the ORANGE plastic tag on the RIGHT arm (robot's right = image right at start)"),
    ("left_wrist_tag", "centre of the ORANGE plastic tag on the LEFT arm"),
    ("right_jaw_mid", "midpoint between the tips of the RIGHT gripper's two foam pads"),
    ("left_jaw_mid", "midpoint between the tips of the LEFT gripper's two foam pads"),
]
VIS = ("visible", "uncertain", "occluded")
FIELDS = ["annotator", "episode", "frame", "purpose", "repeat", "keypoint", "visibility", "u", "v", "timestamp"]
PILOT = (0, 4, 8)
SEALED = {1, 2, 3, 5, 6, 7, 9}
PRINGLES = Path.home() / ".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/cebdee98115ab6827a4ec2655335c536b5dd085b"


# ----------------------------------------------------------------------------------------------- manifest
def manifest_rows(events: dict[int, dict], lengths: dict[int, int], dup_frac: float = 0.15, seed: int = 0) -> list[dict]:
    """Key frames + dense windows around grasp/lift and take-over (command-defined events), then a seeded duplicate
    subset appended in shuffled order (intra-annotator repeatability)."""
    rows = []
    for e in PILOT:
        assert e not in SEALED
        ev, n = events[e], lengths[e]
        rc, lc, ro = ev["right_close"], ev["left_close"], ev["right_open"]
        key = {0: "key_start", rc - 20: "key_pregrasp", rc: "key_grasp_cmd", rc + 30: "key_carry", (rc + lc) // 2: "key_midcarry",
               lc: "key_take_cmd", ro: "key_release_cmd", min(ro + 30, n - 1): "key_after_release"}
        frames = dict(key)
        for k in range(rc - 6, rc + 25, 2):
            frames.setdefault(k, "dense_grasp_lift")
        for k in range(lc - 10, ro + 21, 2):
            frames.setdefault(k, "dense_takeover")
        for k in sorted(frames):
            if 0 <= k < n:
                rows.append({"episode": e, "frame": int(k), "purpose": frames[k], "repeat": 0})
    rng = random.Random(seed)
    dup = rng.sample(rows, max(1, int(round(dup_frac * len(rows)))))
    rng.shuffle(dup)
    rows += [dict(r, repeat=1) for r in dup]
    return rows


def write_manifest(out: Path):
    sys.path.insert(0, str(REPO / "research/contact_aware_augmentation/scripts"))
    import gate0_diagnostics as g0
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="p", episodes=list(PILOT))
    ev = {r["episode"]: r for r in g0.episode_events(P)}
    rows = manifest_rows(ev, {e: int(P.episodes.loc[e]["length"]) for e in PILOT})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "frame", "purpose", "repeat"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} frames ({sum(r['repeat'] for r in rows)} repeats) to {out}")


# ----------------------------------------------------------------------------------------------- session (GUI-free)
class Session:
    """State machine of one annotator over the manifest; independent of OpenCV so it can be tested headlessly."""

    def __init__(self, annotator: str, manifest: list[dict], existing: list[dict] | None = None):
        self.annotator = annotator
        self.items = manifest
        self.answers: dict[tuple, list[dict]] = {}
        for r in existing or []:
            k = (int(r["episode"]), int(r["frame"]), int(r["repeat"]))
            self.answers.setdefault(k, []).append(r)
        self.i = next((j for j, it in enumerate(self.items) if not self.complete(j)), len(self.items))
        self.pending_uncertain = False

    def key(self, j=None):
        it = self.items[self.i if j is None else j]
        return (int(it["episode"]), int(it["frame"]), int(it["repeat"]))

    def complete(self, j=None) -> bool:
        return len(self.answers.get(self.key(j), [])) == len(KEYPOINTS)

    def current_keypoint(self):
        n = len(self.answers.get(self.key(), []))
        return KEYPOINTS[n] if n < len(KEYPOINTS) else None

    def _add(self, vis, u="", v=""):
        kp = self.current_keypoint()
        if kp is None:
            return False
        it = self.items[self.i]
        self.answers.setdefault(self.key(), []).append(
            {"annotator": self.annotator, "episode": it["episode"], "frame": it["frame"], "purpose": it["purpose"],
             "repeat": it["repeat"], "keypoint": kp[0], "visibility": vis, "u": u, "v": v,
             "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
        return True

    def click(self, u: float, v: float):
        ok = self._add("uncertain" if self.pending_uncertain else "visible", round(u, 1), round(v, 1))
        self.pending_uncertain = False
        return ok

    def mark_uncertain_next(self):
        self.pending_uncertain = True

    def occluded(self):
        self.pending_uncertain = False
        return self._add("occluded")

    def undo(self):
        a = self.answers.get(self.key(), [])
        if a:
            a.pop()

    def next(self) -> bool:
        if not self.complete():
            return False
        self.i = min(self.i + 1, len(self.items))
        return True

    def prev(self):
        self.i = max(self.i - 1, 0)

    def done(self) -> bool:
        return self.i >= len(self.items)

    def rows(self) -> list[dict]:
        return [r for k in sorted(self.answers) for r in self.answers[k]]


def save(rows: list[dict], out: Path):
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


# ----------------------------------------------------------------------------------------------- GUI
def gui(annotator: str, manifest: Path, out: Path, scale: int = 2):
    import cv2
    import numpy as np
    from tools.dataset_diagnostics.loader import load_dataset
    items = list(csv.DictReader(open(manifest)))
    assert not {int(r["episode"]) for r in items} & SEALED, "manifest contains sealed episodes"
    existing = list(csv.DictReader(open(out))) if out.exists() else []
    S = Session(annotator, items, existing)
    P = load_dataset(root=PRINGLES, label="p", episodes=list(PILOT))
    cache = {}
    win = f"Gate1 annotation [{annotator}]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    mouse = {"xy": None}
    cv2.setMouseCallback(win, lambda ev, x, y, *_: mouse.update(xy=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while not S.done():
        it = S.items[S.i]
        k = (int(it["episode"]), int(it["frame"]))
        if k not in cache:
            cache[k] = P.read_frame("observation.images.body_cam", *k)[..., ::-1].copy()
        img = cv2.resize(cache[k], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for r in S.answers.get(S.key(), []):
            if r["u"] != "":
                col = (0, 255, 0) if r["visibility"] == "visible" else (0, 165, 255)
                cv2.drawMarker(img, (int(float(r["u"]) * scale), int(float(r["v"]) * scale)), col, cv2.MARKER_CROSS, 24, 2)
        kp = S.current_keypoint()
        msg = (f"{S.i + 1}/{len(S.items)}  ep{k[0]} f{k[1]}  " +
               (f"CLICK: {kp[0]}{' (UNCERTAIN)' if S.pending_uncertain else ''}" if kp else "frame done: n=next"))
        cv2.rectangle(img, (0, 0), (img.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(img, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if kp:
            cv2.putText(img, kp[1][:110], (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(win, img)
        c = cv2.waitKey(30) & 0xFF
        if mouse["xy"] is not None:
            x, y = mouse["xy"]
            mouse["xy"] = None
            if y > 64:
                S.click(x / scale, y / scale)
        elif c == ord("u"):
            S.mark_uncertain_next()
        elif c == ord("o"):
            S.occluded()
        elif c == ord("z"):
            S.undo()
        elif c in (ord("n"), ord(" ")):
            if S.next():
                save(S.rows(), out)
        elif c == ord("p"):
            S.prev()
        elif c == ord("q"):
            break
    save(S.rows(), out)
    cv2.destroyAllWindows()
    print(f"saved {len(S.rows())} keypoint answers to {out}; {'complete' if S.done() else 'resume later with the same command'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("--annotator", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        write_manifest(Path(args.out))
    else:
        gui(args.annotator, Path(args.manifest), Path(args.out))


if __name__ == "__main__":
    main()
