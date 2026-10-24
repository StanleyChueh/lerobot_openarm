"""Per-episode initial object position: sources, priority, annotation files and image->workspace transforms.

Priority (best first), as required by the readiness protocol:
  A  recorded_pose        object pose logged by the recorder/simulator, in a declared frame
  B  validated_estimate   camera-based estimate whose transform/detector passed a held-out validation
  C  human_annotation     a person clicked the object centre in the starting RGB frame (image coordinates)
  D  unknown              nothing available -- never filled in from robot joint states

`auto_detect` proposals (colour blob) are kept as a separate, lower tier: they are image-space measurements
that nobody verified, used only for provisional plots and as pre-filled suggestions for annotators.

Image coordinates (u, v in pixels of a named camera) and metric workspace coordinates (x, y in metres of a
named frame) are stored in separate columns and never mixed. A metric position derived from an image needs a
transform whose validation evidence is attached (see ImageToWorkspace).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

RECORDED, VALIDATED, HUMAN, AUTO, UNKNOWN_SRC = (
    "recorded_pose", "validated_estimate", "human_annotation", "auto_detect", "unknown")
PRIORITY = {RECORDED: 0, VALIDATED: 1, HUMAN: 2, AUTO: 3, UNKNOWN_SRC: 4}
ANNOTATION_COLUMNS = ["dataset", "episode_index", "camera", "frame_index", "u", "v", "image_width", "image_height",
                      "source", "verified", "annotator", "timestamp", "notes"]


@dataclass
class Position:
    episode_index: int
    source: str = UNKNOWN_SRC
    # metric workspace coordinates
    x: float | None = None
    y: float | None = None
    z: float | None = None
    frame_id: str | None = None
    # image coordinates
    u: float | None = None
    v: float | None = None
    camera: str | None = None
    image_size: tuple[int, int] | None = None
    verified: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def has_metric(self) -> bool:
        return self.x is not None and self.y is not None and self.frame_id is not None

    @property
    def has_image(self) -> bool:
        return self.u is not None and self.v is not None and self.camera is not None

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d["image_size"] = None if self.image_size is None else f"{self.image_size[0]}x{self.image_size[1]}"
        d.update({f"extra_{k}": v for k, v in self.extra.items()})
        return d


def merge_by_priority(*sources: dict[int, Position], episodes: list[int]) -> dict[int, Position]:
    """Best available position per episode; episodes without any source become explicit UNKNOWN."""
    out = {}
    for e in episodes:
        cands = [s[e] for s in sources if e in s and s[e].source != UNKNOWN_SRC]
        out[e] = min(cands, key=lambda p: PRIORITY[p.source]) if cands else Position(e, UNKNOWN_SRC)
    return out


# ----------------------------------------------------------------------------- annotation files


def read_annotations(path: str | Path, dataset: str | None = None) -> dict[int, Position]:
    """Human (or proposed) image annotations. Rows of other datasets are ignored when `dataset` is given."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if dataset and row.get("dataset") and row["dataset"] != dataset:
                continue
            if row.get("u") in (None, "") or row.get("v") in (None, ""):
                continue
            src = row.get("source") or HUMAN
            verified = str(row.get("verified", "")).lower() in ("1", "true", "yes")
            if src == AUTO and verified:  # a person confirmed the proposal: it is a human annotation now
                src = HUMAN
            w, h = row.get("image_width"), row.get("image_height")
            out[int(row["episode_index"])] = Position(
                episode_index=int(row["episode_index"]), source=src, u=float(row["u"]), v=float(row["v"]),
                camera=row.get("camera"), image_size=(int(w), int(h)) if w and h else None,
                verified=verified or src == HUMAN, notes=row.get("notes", ""),
                extra={"annotator": row.get("annotator", ""), "frame_index": row.get("frame_index", "0")})
    return out


def write_annotations(path: str | Path, dataset: str, positions: dict[int, Position], annotator: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        w.writeheader()
        for e in sorted(positions):
            p = positions[e]
            w.writerow({"dataset": dataset, "episode_index": e, "camera": p.camera or "",
                        "frame_index": p.extra.get("frame_index", 0), "u": "" if p.u is None else round(p.u, 1),
                        "v": "" if p.v is None else round(p.v, 1),
                        "image_width": p.image_size[0] if p.image_size else "",
                        "image_height": p.image_size[1] if p.image_size else "",
                        "source": p.source, "verified": int(bool(p.verified)),
                        "annotator": p.extra.get("annotator", annotator), "timestamp": p.extra.get(
                            "timestamp", dt.datetime.now().isoformat(timespec="seconds")), "notes": p.notes})


def read_recorded_poses(path: str | Path) -> dict[int, Position]:
    """Generic recorded poses: CSV with episode_index,x,y[,z],frame_id (frame_id is mandatory)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("frame_id"):
                raise ValueError(f"{path}: recorded poses need a frame_id column (coordinate frame)")
            out[int(row["episode_index"])] = Position(
                int(row["episode_index"]), RECORDED, x=float(row["x"]), y=float(row["y"]),
                z=float(row["z"]) if row.get("z") not in (None, "") else None, frame_id=row["frame_id"],
                verified=True, notes=row.get("notes", ""))
    return out


# ----------------------------------------------------------------------------- image -> workspace transform


class TransformError(ValueError):
    pass


def fit_homography(uv: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """DLT homography mapping pixels to plane coordinates (normalized for conditioning)."""
    uv, xy = np.asarray(uv, float), np.asarray(xy, float)
    if len(uv) < 4 or len(uv) != len(xy):
        raise TransformError(f"a homography needs >= 4 correspondences, got {len(uv)}")

    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Tu, Tx = norm(uv), norm(xy)
    a = np.c_[uv, np.ones(len(uv))] @ Tu.T
    b = np.c_[xy, np.ones(len(xy))] @ Tx.T
    rows = []
    for (u, v, _), (x, y, _) in zip(a, b):
        rows.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        rows.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    _, s, vt = np.linalg.svd(np.asarray(rows))
    if s[-2] < 1e-9 * s[0]:
        raise TransformError("degenerate correspondences (collinear or repeated points)")
    H = np.linalg.inv(Tx) @ vt[-1].reshape(3, 3) @ Tu
    return H / H[2, 2]


def apply_homography(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(uv, float), np.ones(len(uv))] @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


@dataclass
class ImageToWorkspace:
    """A pixel -> metric-plane map with the evidence that it is right. Without passing evidence it is refused."""

    camera: str
    frame_id: str
    H: np.ndarray
    image_size: tuple[int, int] | None
    n_points: int
    heldout_rmse_m: float
    heldout_max_m: float
    max_allowed_m: float
    source: str = ""

    @property
    def valid(self) -> bool:
        return self.n_points >= 6 and np.isfinite(self.heldout_rmse_m) and self.heldout_max_m <= self.max_allowed_m

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "H"}
        d["H"] = np.asarray(self.H).tolist()
        d["valid"] = self.valid
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ImageToWorkspace":
        for k in ("camera", "frame_id", "H", "n_points", "heldout_rmse_m", "heldout_max_m", "max_allowed_m"):
            if k not in d:
                raise TransformError(f"transform file lacks {k!r}: no validation evidence, refusing to use it")
        H = np.asarray(d["H"], float)
        if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
            raise TransformError("transform matrix is not a finite, invertible 3x3 homography")
        return cls(d["camera"], d["frame_id"], H, tuple(d["image_size"]) if d.get("image_size") else None,
                   int(d["n_points"]), float(d["heldout_rmse_m"]), float(d["heldout_max_m"]),
                   float(d["max_allowed_m"]), d.get("source", ""))

    def map(self, positions: dict[int, Position]) -> dict[int, Position]:
        if not self.valid:
            raise TransformError(f"transform for {self.camera} failed validation (held-out max "
                                 f"{self.heldout_max_m:.3f} m > {self.max_allowed_m} m or < 6 points)")
        out = {}
        for e, p in positions.items():
            if not p.has_image or p.camera != self.camera:
                continue
            if self.image_size and p.image_size and tuple(p.image_size) != tuple(self.image_size):
                raise TransformError(f"episode {e}: annotation image size {p.image_size} != transform's "
                                     f"{self.image_size}")
            x, y = apply_homography(self.H, np.array([[p.u, p.v]]))[0]
            out[e] = Position(e, VALIDATED if p.source in (HUMAN, RECORDED) else AUTO, x=float(x), y=float(y),
                              frame_id=self.frame_id, u=p.u, v=p.v, camera=p.camera, image_size=p.image_size,
                              verified=p.verified, notes=f"mapped by {self.source or 'homography'}",
                              extra={"image_source": p.source})
        return out


def fit_validated_transform(uv: np.ndarray, xy: np.ndarray, *, camera: str, frame_id: str,
                            image_size: tuple[int, int] | None = None, max_allowed_m: float = 0.02,
                            source: str = "") -> ImageToWorkspace:
    """Fit on all points; report leave-one-out held-out error as the validation evidence."""
    uv, xy = np.asarray(uv, float), np.asarray(xy, float)
    H = fit_homography(uv, xy)
    errs = []
    for i in range(len(uv)):
        keep = np.arange(len(uv)) != i
        try:
            Hi = fit_homography(uv[keep], xy[keep])
        except TransformError:
            errs.append(np.inf)
            continue
        errs.append(float(np.linalg.norm(apply_homography(Hi, uv[i:i + 1])[0] - xy[i])))
    errs = np.asarray(errs)
    return ImageToWorkspace(camera, frame_id, H, image_size, len(uv),
                            float(np.sqrt(np.mean(errs**2))) if len(errs) else np.inf,
                            float(errs.max()) if len(errs) else np.inf, max_allowed_m, source)


def load_transform(path: str | Path) -> ImageToWorkspace:
    return ImageToWorkspace.from_json(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------------- colour-blob proposals


def detect_color_blob(rgb: np.ndarray, hsv_ranges: list[dict], roi: list[int] | None = None,
                      min_area: int = 200) -> dict | None:
    """Centroid of the largest blob inside any of the HSV ranges ({"h": [lo, hi], "s": [..], "v": [..]},
    OpenCV scale H 0-179). Returns {"u", "v", "area", "bbox"} or None. Proposals only -- not validated."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in hsv_ranges:
        lo = np.array([r["h"][0], r["s"][0], r["v"][0]], np.uint8)
        hi = np.array([r["h"][1], r["s"][1], r["v"][1]], np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    if roi:
        x0, y0, x1, y1 = roi
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = 1
        mask *= keep
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None
    x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                               cv2.CC_STAT_HEIGHT))
    return {"u": float(cents[i][0]), "v": float(cents[i][1]), "area": int(stats[i, cv2.CC_STAT_AREA]),
            "bbox": [x, y, x + w, y + h]}
