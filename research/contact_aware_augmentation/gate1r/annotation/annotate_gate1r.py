"""Gate 1R independent human annotation (episodes 0, 4, 8 only; blind to every automatic output).

Two tasks, one CSV each per annotator:

  keypoints  per manifest frame: object and gripper landmarks with visibility and confidence
  events     per episode: frames of the VISIBLE grasp / lift / hand-over / release events, scrubbing the video

    T=research/contact_aware_augmentation/gate1r/annotation/annotate_gate1r.py
    O=research/contact_aware_augmentation/outputs/gate1r/annotation
    PY="env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python"
    $PY $T manifest  --out $O/manifest_keypoints.csv
    $PY $T keypoints --annotator A1 --manifest $O/manifest_keypoints.csv --out $O/keypoints_A1.csv
    $PY $T events    --annotator A1 --out $O/events_A1.csv

Provenance written on every row: annotator, session id, tool file SHA-256, repository commit, dataset revision, video
file SHA-256, SHA-1 of the decoded RGB frame shown, UTC timestamp. Nothing is pre-filled; the tool never displays
tracker, FK, verifier output or another annotator's answers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import random
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO), str(HERE.parent)]
import spec as S  # noqa: E402

DATASET_REV = "cebdee98115ab6827a4ec2655335c536b5dd085b"
PRINGLES = Path.home() / f".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/{DATASET_REV}"
CAM = "observation.images.body_cam"
KEYPOINTS = [  # (key, group, required, description)
    ("obj_lid_center", "object", True, "centre of the can's pale circular lid (estimate from the rim if partly hidden)"),
    ("obj_bottom_center", "object", False, "centre of the can's bottom edge, if you can see it"),
    ("R_pad_tip_1", "right_gripper", True, "tip of the RIGHT gripper's first foam pad (either pad; be consistent: the one nearer the image top)"),
    ("R_pad_tip_2", "right_gripper", True, "tip of the RIGHT gripper's other foam pad"),
    ("R_wrist_tag", "right_gripper", False, "centre of the orange tag on the RIGHT arm's wrist"),
    ("L_pad_tip_1", "left_gripper", True, "tip of the LEFT gripper's first foam pad (the one nearer the image top)"),
    ("L_pad_tip_2", "left_gripper", True, "tip of the LEFT gripper's other foam pad"),
    ("L_wrist_tag", "left_gripper", False, "centre of the orange tag on the LEFT arm's wrist"),
]
EVENTS = [
    ("right_first_contact", "first frame the RIGHT pads visibly touch the can"),
    ("lift_off", "first frame the can is visibly off the table"),
    ("left_first_contact", "first frame the LEFT pads visibly touch the can"),
    ("right_release", "first frame the RIGHT pads are visibly separated from the can"),
]
VIS = ("visible", "occluded", "out_of_frame")
KP_FIELDS = ["annotator", "session", "episode", "frame", "purpose", "repeat", "keypoint", "group", "visibility", "confidence",
             "u", "v", "frame_sha1", "video_sha256", "dataset_rev", "repo_commit", "tool_sha256", "timestamp_utc"]
EV_FIELDS = ["annotator", "session", "episode", "event", "observable", "frame", "confidence", "video_sha256", "dataset_rev",
             "repo_commit", "tool_sha256", "timestamp_utc"]


def provenance() -> dict:
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"session": uuid.uuid4().hex[:12], "tool_sha256": sha, "repo_commit": commit, "dataset_rev": DATASET_REV}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------------------------- manifest
def manifest_rows(events: dict, lengths: dict, dup_frac=0.15, seed=1) -> list[dict]:
    rows = []
    for e in S.PILOT:
        rc, lc, ro = events[e]["right_close"], events[e]["left_close"], events[e]["right_open"]
        n = lengths[e]
        frames = {0: "key_start", rc - 20: "key_pregrasp", rc + 30: "key_carry", (rc + lc) // 2: "key_midcarry",
                  ro + 30: "key_after_release", ro + 45: "key_receiver_hold"}
        for k in range(rc - 8, rc + 26, 2):
            frames.setdefault(k, "dense_grasp_lift")
        for k in range(lc - 12, ro + 22, 2):
            frames.setdefault(k, "dense_transition")
        rows += [{"episode": e, "frame": k, "purpose": p, "repeat": 0} for k, p in sorted(frames.items()) if 0 <= k < n]
    rng = random.Random(seed)
    dup = rng.sample(rows, max(1, round(dup_frac * len(rows))))
    rng.shuffle(dup)
    return rows + [dict(r, repeat=1) for r in dup]


def write_manifest(out: Path):
    sys.path.insert(0, str(REPO / "research/contact_aware_augmentation/scripts"))
    import gate0_diagnostics as g0
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="p", episodes=list(S.PILOT))
    ev = {r["episode"]: r for r in g0.episode_events(P)}
    rows = manifest_rows(ev, {e: int(P.episodes.loc[e]["length"]) for e in S.PILOT})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "frame", "purpose", "repeat"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({sum(r['repeat'] for r in rows)} repeats) -> {out}")


# ----------------------------------------------------------------------------------------------- keypoint session
class KeypointSession:
    """GUI-free state machine (tested headlessly)."""

    def __init__(self, annotator, items, prov, existing=None):
        self.annotator, self.items, self.prov = annotator, items, prov
        self.answers = {}
        for r in existing or []:
            self.answers.setdefault((int(r["episode"]), int(r["frame"]), int(r["repeat"])), []).append(r)
        self.i = next((j for j in range(len(items)) if not self.complete(j)), len(items))
        self.confidence = 3
        self.frame_meta = {}

    def key(self, j=None):
        it = self.items[self.i if j is None else j]
        return (int(it["episode"]), int(it["frame"]), int(it["repeat"]))

    def complete(self, j=None):
        return len(self.answers.get(self.key(j), [])) == len(KEYPOINTS)

    def current(self):
        n = len(self.answers.get(self.key(), []))
        return KEYPOINTS[n] if n < len(KEYPOINTS) else None

    def set_confidence(self, c: int):
        assert c in (1, 2, 3)
        self.confidence = c

    def _add(self, vis, u="", v=""):
        kp = self.current()
        if kp is None:
            return False
        it = self.items[self.i]
        fm = self.frame_meta.get(self.key(), {})
        self.answers.setdefault(self.key(), []).append({
            "annotator": self.annotator, "session": self.prov["session"], "episode": it["episode"], "frame": it["frame"],
            "purpose": it["purpose"], "repeat": it["repeat"], "keypoint": kp[0], "group": kp[1], "visibility": vis,
            "confidence": self.confidence if vis == "visible" else "", "u": u, "v": v,
            "frame_sha1": fm.get("frame_sha1", ""), "video_sha256": fm.get("video_sha256", ""), "dataset_rev": self.prov["dataset_rev"],
            "repo_commit": self.prov["repo_commit"], "tool_sha256": self.prov["tool_sha256"], "timestamp_utc": now()})
        self.confidence = 3
        return True

    def click(self, u, v):
        return self._add("visible", round(float(u), 1), round(float(v), 1))

    def occluded(self):
        return self._add("occluded")

    def out_of_frame(self):
        return self._add("out_of_frame")

    def undo(self):
        a = self.answers.get(self.key(), [])
        if a:
            a.pop()

    def next(self):
        if not self.complete():
            return False
        self.i += 1
        return True

    def prev(self):
        self.i = max(self.i - 1, 0)

    def done(self):
        return self.i >= len(self.items)

    def rows(self):
        return [r for k in sorted(self.answers) for r in self.answers[k]]


# ----------------------------------------------------------------------------------------------- event session
class EventSession:
    def __init__(self, annotator, prov, lengths: dict, existing=None):
        self.annotator, self.prov, self.lengths = annotator, prov, lengths
        self.answers = {(int(r["episode"]), r["event"]): r for r in existing or []}
        self.order = [(e, ev[0]) for e in S.PILOT for ev in EVENTS]
        self.i = next((j for j, k in enumerate(self.order) if k not in self.answers), len(self.order))
        self.frame = 0
        self.confidence = 3

    def current(self):
        return self.order[self.i] if self.i < len(self.order) else None

    def step(self, d):
        e, _ = self.current()
        self.frame = int(min(max(self.frame + d, 0), self.lengths[e] - 1))

    def mark(self, video_sha=""):
        e, name = self.current()
        self.answers[(e, name)] = self._row(e, name, True, self.frame, video_sha)
        self._advance()

    def not_observable(self, video_sha=""):
        e, name = self.current()
        self.answers[(e, name)] = self._row(e, name, False, "", video_sha)
        self._advance()

    def _row(self, e, name, obs, frame, vsha):
        return {"annotator": self.annotator, "session": self.prov["session"], "episode": e, "event": name,
                "observable": int(obs), "frame": frame, "confidence": self.confidence if obs else "", "video_sha256": vsha,
                "dataset_rev": self.prov["dataset_rev"], "repo_commit": self.prov["repo_commit"],
                "tool_sha256": self.prov["tool_sha256"], "timestamp_utc": now()}

    def _advance(self):
        prev = self.current()[0]
        self.i += 1
        self.confidence = 3
        if self.current() and self.current()[0] != prev:
            self.frame = 0

    def done(self):
        return self.i >= len(self.order)

    def rows(self):
        return [self.answers[k] for k in self.order if k in self.answers]


def save(rows, out: Path, fields):
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


# ----------------------------------------------------------------------------------------------- GUI
def _video_sha(P, e):
    rel, _ = P.video_path(CAM, e)
    return hashlib.sha256((P.root / rel).read_bytes()).hexdigest()


def _frames(P, e):
    import av
    import numpy as np
    rel, t0 = P.video_path(CAM, e)
    n = int(P.episodes.loc[e]["length"])
    out = []
    with av.open(str(P.root / rel)) as c:
        st = c.streams.video[0]
        c.seek(int(max(t0 - 0.5, 0) / st.time_base), stream=st)
        for fr in c.decode(st):
            if fr.time is None or fr.time < t0 - 0.5 / 30:
                continue
            k = int(round((fr.time - t0) * 30))
            if k >= n:
                break
            if k == len(out):
                out.append(fr.to_ndarray(format="rgb24"))
    return np.stack(out)


def gui_keypoints(annotator, manifest: Path, out: Path, scale=2):
    import cv2
    from tools.dataset_diagnostics.loader import load_dataset
    items = list(csv.DictReader(open(manifest)))
    assert {int(r["episode"]) for r in items} <= set(S.PILOT), "manifest outside the pilot episodes"
    P = load_dataset(root=PRINGLES, label="p", episodes=list(S.PILOT))
    existing = list(csv.DictReader(open(out))) if out.exists() else []
    K = KeypointSession(annotator, items, provenance(), existing)
    vids, frames = {}, {}
    win = f"Gate1R keypoints [{annotator}]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    mouse = {"xy": None}
    cv2.setMouseCallback(win, lambda ev, x, y, *_: mouse.update(xy=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while not K.done():
        e, f, _ = K.key()
        if e not in frames:
            frames[e], vids[e] = _frames(P, e), _video_sha(P, e)
        rgb = frames[e][f]
        K.frame_meta[K.key()] = {"frame_sha1": hashlib.sha1(rgb.tobytes()).hexdigest(), "video_sha256": vids[e]}
        img = cv2.resize(rgb[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for r in K.answers.get(K.key(), []):
            if r["u"] != "":
                cv2.drawMarker(img, (int(float(r["u"]) * scale), int(float(r["v"]) * scale)), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        kp = K.current()
        bar = f"{K.i + 1}/{len(K.items)} ep{e} f{f}  " + (f"CLICK {kp[0]} [{'req' if kp[2] else 'opt'}] conf={K.confidence}" if kp else "done: n")
        cv2.rectangle(img, (0, 0), (img.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(img, bar, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if kp:
            cv2.putText(img, kp[3][:115], (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win, img)
        c = cv2.waitKey(30) & 0xFF
        if mouse["xy"] is not None:
            x, y = mouse["xy"]; mouse["xy"] = None
            if y > 64:
                K.click(x / scale, y / scale)
        elif c in (ord("1"), ord("2"), ord("3")):
            K.set_confidence(int(chr(c)))
        elif c == ord("o"):
            K.occluded()
        elif c == ord("f"):
            K.out_of_frame()
        elif c == ord("z"):
            K.undo()
        elif c in (ord("n"), ord(" ")):
            if K.next():
                save(K.rows(), out, KP_FIELDS)
        elif c == ord("p"):
            K.prev()
        elif c == ord("q"):
            break
    save(K.rows(), out, KP_FIELDS)
    cv2.destroyAllWindows()


def gui_events(annotator, out: Path, scale=2):
    import cv2
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="p", episodes=list(S.PILOT))
    lengths = {e: int(P.episodes.loc[e]["length"]) for e in S.PILOT}
    existing = list(csv.DictReader(open(out))) if out.exists() else []
    E = EventSession(annotator, provenance(), lengths, existing)
    frames, vids = {}, {}
    win = f"Gate1R events [{annotator}]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while not E.done():
        e, name = E.current()
        if e not in frames:
            frames[e], vids[e] = _frames(P, e), _video_sha(P, e)
        img = cv2.resize(frames[e][E.frame][..., ::-1], None, fx=scale, fy=scale)
        desc = dict(EVENTS)[name]
        cv2.rectangle(img, (0, 0), (img.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(img, f"ep{e} frame {E.frame}/{lengths[e] - 1}  EVENT {name} conf={E.confidence}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, desc + "   [a/d +-1, w/s +-10, m=mark, x=not observable, 1-3 conf, q=quit]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win, img)
        c = cv2.waitKey(30) & 0xFF
        if c == ord("a"):
            E.step(-1)
        elif c == ord("d"):
            E.step(1)
        elif c == ord("w"):
            E.step(10)
        elif c == ord("s"):
            E.step(-10)
        elif c in (ord("1"), ord("2"), ord("3")):
            E.confidence = int(chr(c))
        elif c == ord("m"):
            E.mark(vids[e]); save(E.rows(), out, EV_FIELDS)
        elif c == ord("x"):
            E.not_observable(vids[e]); save(E.rows(), out, EV_FIELDS)
        elif c == ord("q"):
            break
    save(E.rows(), out, EV_FIELDS)
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").add_argument("--out", required=True)
    k = sub.add_parser("keypoints"); k.add_argument("--annotator", required=True); k.add_argument("--manifest", required=True); k.add_argument("--out", required=True)
    v = sub.add_parser("events"); v.add_argument("--annotator", required=True); v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        write_manifest(Path(a.out))
    elif a.cmd == "keypoints":
        gui_keypoints(a.annotator, Path(a.manifest), Path(a.out))
    else:
        gui_events(a.annotator, Path(a.out))


if __name__ == "__main__":
    main()
