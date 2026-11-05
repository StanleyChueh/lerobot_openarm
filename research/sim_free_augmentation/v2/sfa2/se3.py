"""Small SE(3) helpers (pinocchio log/exp). Homogeneous 4x4 matrices throughout."""

from __future__ import annotations

import numpy as np
import pinocchio as pin


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def log6(T) -> np.ndarray:
    return pin.log6(pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())).vector.copy()


def exp6(xi) -> np.ndarray:
    return pin.exp6(pin.Motion(np.asarray(xi, float))).homogeneous.copy()


def power(D, w: float) -> np.ndarray:
    """screw interpolation D^w (w=0 -> identity, w=1 -> D)."""
    if w == 0.0:
        return np.eye(4)
    if w == 1.0:
        return D.copy()
    return exp6(w * log6(D))


def blend(A, B, w: float) -> np.ndarray:
    """pose between A (w=0) and B (w=1) along the left-invariant screw B A^-1."""
    return power(B @ inv(A), w) @ A


def trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v
    return T


def rot(axis, deg) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    T = np.eye(4)
    T[:3, :3] = pin.exp3(a * np.radians(deg))
    return T


def about(Rot4, pivot) -> np.ndarray:
    """rotation Rot4 applied about a world pivot point."""
    return trans(pivot) @ Rot4 @ trans(-np.asarray(pivot, float))


def angle_deg(Ra, Rb=None) -> float:
    Rr = Ra if Rb is None else Ra.T @ Rb
    return float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))


def axis_tilt_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def spin_deg(Ra, Rb) -> float:
    """rotation about the (shared) z axis between two frames after aligning their z axes (can spin)."""
    za, zb = Ra[:, 2], Rb[:, 2]
    v = np.cross(za, zb)
    s, c = np.linalg.norm(v), za @ zb
    if s < 1e-9:
        Al = np.eye(3)
    else:
        Al = pin.exp3(v / s * np.arctan2(s, c))
    xa = Al @ Ra[:, 0]
    xb = Rb[:, 0]
    return float(np.degrees(np.arctan2(np.cross(xa, xb) @ zb, xa @ xb)))
