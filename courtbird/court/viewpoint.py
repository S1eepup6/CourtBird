"""Where was the video shot from? Two independent estimates.

A. `from_calibration`: the court homography contains the camera pose. With the principal point at the
   image centre and square pixels, the focal length and the camera position follow from the fact that
   the court axes are orthogonal and equally scaled (metric court coordinates).
B. `from_layout`: without any calibration. Standing people are a ruler: person height in pixels
   together with the foot's image row gives the horizon, the camera height and, with a focal-length
   guess, approximate ground-plane coordinates. The principal axis of where people spend their time
   is then the court's long axis, and its angle to the viewing direction is the answer.

Both report `beta`: the angle (0-90 deg) between the viewing direction projected on the floor and the
court's long axis. 0 = looking straight down the court from behind an end line, 90 = looking
across from a sideline, in between = from a corner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..sports.spec import SportSpec
from .calibration import Calibration

PERSON_M = 1.75


@dataclass
class ViewPoint:
    kind: str                    # "end" | "side" | "corner"
    beta: float                  # degrees, see module docstring
    source: str                  # "calibration" | "layout"
    confidence: float            # 0..1, rough
    position: tuple[float, float] | None = None   # camera (x, y) on the court plane, metres (calibration only)
    height: float | None = None  # camera height above the floor, metres
    focal_px: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self, spec: SportSpec | None = None) -> str:
        label = {"end": "끝선 뒤(엔드라인)", "side": "옆선(사이드라인)", "corner": "코너(대각선)"}[self.kind]
        s = f"{label} 시점 (시선과 코트 길이축 사이 {self.beta:.0f}°)"
        if self.position is not None and spec is not None:
            end = "x=0 쪽" if self.position[0] < spec.length / 2 else "x=L 쪽"
            s += f", 카메라 약 ({self.position[0]:.1f}, {self.position[1]:.1f}) m, 높이 {self.height:.1f} m, {end}"
        return s


def classify(beta: float) -> str:
    return "end" if beta < 25 else ("side" if beta > 65 else "corner")


def _solve_focal(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Focal length (px) from the metric constraints on the first two columns of the court->image
    homography: K^-1 h1 and K^-1 h2 are orthogonal and of equal length."""
    h1, h2 = H[:, 0], H[:, 1]

    def n(a, b):
        return (a[0] - cx * a[2]) * (b[0] - cx * b[2]) + (a[1] - cy * a[2]) * (b[1] - cy * b[2])
    n12, n11, n22 = n(h1, h2), n(h1, h1), n(h2, h2)
    w12, w11, w22 = h1[2] * h2[2], h1[2] ** 2, h2[2] ** 2
    # unknown s = 1/f^2:   n12*s + w12 = 0   and   (n11-n22)*s + (w11-w22) = 0  (least squares)
    a = np.array([n12, n11 - n22])
    b = -np.array([w12, w11 - w22])
    den = float(a @ a)
    if den < 1e-18:
        return None
    s = float(a @ b) / den
    return float(1.0 / np.sqrt(s)) if s > 1e-12 else None


def from_calibration(calib: Calibration, spec: SportSpec, focal_fallback: float = 0.9) -> ViewPoint:
    """Camera pose from the court homography (see module docstring)."""
    W, Hh = calib.image_size
    cx, cy = W / 2.0, Hh / 2.0
    Hc2i = np.linalg.inv(calib.H)
    Hc2i = Hc2i / np.linalg.norm(Hc2i)
    f = _solve_focal(Hc2i, cx, cy)
    note = ""
    conf = 0.9
    if f is None or not (0.3 * W <= f <= 4.0 * W):
        f, note, conf = focal_fallback * W, "focal length not recoverable from the calibration; assumed", 0.4
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ Hc2i
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if (lam * B[2, 2]) < 0:  # court must be in front of the camera
        lam = -lam
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    R = U @ Vt
    C = -R.T @ t                      # camera centre in court coordinates
    axis = R.T @ np.array([0.0, 0.0, 1.0])  # optical axis in court coordinates
    horiz = float(np.hypot(axis[0], axis[1]))
    if horiz < 1e-6:
        return ViewPoint("corner", 45.0, "calibration", 0.1, note="camera looks straight down")
    beta = float(np.degrees(np.arccos(min(1.0, abs(axis[0]) / horiz))))
    return ViewPoint(classify(beta), beta, "calibration", conf, (float(C[0]), float(C[1])), float(abs(C[2])),
                     float(f), note)


# --- B: calibration-free ---------------------------------------------------------------------

def _fit_horizon(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None, iters: int = 400, seed: int = 0):
    """Robust fit of  s = a * (y - y_h)  (person height grows linearly with the foot row on a plane).
    Returns (a, y_h, inlier mask) or None. People on a different plane (spectators) are rejected as
    outliers; `w` (e.g. how much each person moves) makes the dominant plane the one most *people
    in motion* stand on, so many stationary spectators cannot win over a few players."""
    rng = np.random.default_rng(seed)
    best = None
    best_score = -1.0
    n = len(y)
    if w is None:
        w = np.ones(n)
    if n < 4:
        return None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(y[i] - y[j]) < 5:
            continue
        a = (s[i] - s[j]) / (y[i] - y[j])
        if a <= 0:
            continue
        resid = np.abs(s - (s[i] + a * (y - y[i]))) / np.maximum(s, 1.0)
        inl = resid < 0.10
        score = float(w[inl].sum())
        if score > best_score:
            best_score, best = score, (a, y[i] - s[i] / a, inl)
    if best is None or best[2].sum() < 4:
        return None
    a, yh, inl = best
    A = np.c_[y[inl], np.ones(inl.sum())]  # refine on the inliers
    (a, b), *_ = np.linalg.lstsq(A, s[inl], rcond=None)
    return float(a), float(-b / a), inl


def ground_coordinates(feet: np.ndarray, heights: np.ndarray, size: tuple[int, int], focal: float,
                       horizon_y: float, cam_height: float) -> np.ndarray | None:
    """Approximate ground coordinates (right, forward) in metres of image foot points, for a camera
    with zero roll whose pitch follows from the horizon row. None if the geometry is impossible."""
    W, H = size
    cx, cy = W / 2.0, H / 2.0
    tan_pitch = (cy - horizon_y) / focal           # > 0: camera looks down
    if not (0.0 < tan_pitch < 6.0):
        return None
    th = np.arctan(tan_pitch)
    x = (feet[:, 0] - cx) / focal
    y = (feet[:, 1] - cy) / focal
    # camera ray in (right, forward, up), pitch down by th
    up = -(y * np.cos(th)) + (-np.sin(th))        # component along world up
    fwd = np.cos(th) - y * np.sin(th)
    ok = up < -1e-6
    t = np.full(len(feet), np.nan)
    t[ok] = cam_height / (-up[ok])
    return np.c_[t * x, t * fwd]


def from_layout(feet: np.ndarray, heights: np.ndarray, size: tuple[int, int],
                focal_guess: float = 0.9, weights: np.ndarray | None = None,
                teams: np.ndarray | None = None) -> ViewPoint | None:
    """Camera orientation from where people stand, with no calibration (see module docstring).

    feet    : (N, 2) pixel foot points of people over time (a track contributes many rows)
    heights : (N,)   box heights in pixels
    weights : (N,)   optional importance, e.g. how much the person moves (stationary spectators ~ 0)
    teams   : (N,)   optional 0/1 team label per point (e.g. from kit colour). When both teams are
              present their centroids' direction is used as the court axis: much steadier than the
              principal axis of 12 wandering people.
    """
    W, Hh = size
    fit = _fit_horizon(feet[:, 1], heights, weights)
    if fit is None:
        return None
    a, y_h, inl = fit
    f = focal_guess * W
    cam_h = float(np.median(PERSON_M * (feet[inl, 1] - y_h) / heights[inl]))
    g = ground_coordinates(feet[inl], heights[inl], size, f, y_h, cam_h)
    if g is None:
        return None
    ok = np.isfinite(g).all(axis=1)
    g = g[ok]
    if len(g) < 20:
        return None
    w = None if weights is None else weights[inl][ok]
    mean = np.average(g, axis=0, weights=w)
    d = g - mean
    cov = (d * (w[:, None] if w is not None else 1.0)).T @ d / (w.sum() if w is not None else len(d))
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]                          # (right, forward)
    beta = float(np.degrees(np.arctan2(abs(major[0]), abs(major[1]))))   # 0: along the viewing direction
    ratio = float(evals[1] / max(evals[0], 1e-9))
    conf = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))  # a 2:1 court gives ~4:1 variance
    how = f"principal axis, variance ratio {ratio:.1f}"
    if teams is not None:
        tm = teams[inl][ok]
        if (tm == 0).sum() >= 10 and (tm == 1).sum() >= 10:
            c0 = np.average(g[tm == 0], axis=0, weights=None if w is None else w[tm == 0])
            c1 = np.average(g[tm == 1], axis=0, weights=None if w is None else w[tm == 1])
            v = c1 - c0
            if np.hypot(*v) > 3.0:               # centroids of the two sides well apart (the court is 18 m)
                beta = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]))))
                conf, how = max(conf, 0.6), f"team centroid axis ({np.hypot(*v):.1f} m apart)"
    return ViewPoint(classify(beta), beta, "layout", conf, None, cam_h, f,
                     f"{int(inl.sum())}/{len(inl)} points on the floor plane, {how}")


# --- from the tracker's output ----------------------------------------------------------------

def track_arrays(tracks: list, min_frames: int = 15):
    """Flatten tracklets into per-point arrays for `from_layout`.

    Returns feet (N,2), heights (N,), weights (N,), teams (N,) and the track index of each point.
    The weight of a track says how much it *moves* (extent in person heights, clipped to 0.02..1):
    players wander across the court, spectators and officials stand still. Teams (0/1, else 2) come
    from clustering the moving tracks' kit-colour histograms in two groups; they are only used when
    the two groups are clearly separated.
    """
    import cv2
    feet, heights, weights, tidx, ok_all = [], [], [], [], []
    kept = []
    for t in tracks:
        h = t.history
        if len(h) < min_frames:
            continue
        a = np.array([[r[1], r[2], r[3], float(r[4])] for r in h])
        a = a[a[:, 3] > 0.5]                       # feet in frame
        if len(a) < min_frames:
            continue
        extent = float(max(np.ptp(a[:, 0]), np.ptp(a[:, 1])) / max(np.median(a[:, 2]), 1.0))
        kept.append((t, a, float(np.clip(extent, 0.02, 1.0))))
    if not kept:
        return None
    teams_of = {}
    moving = [(i, k[0].feat) for i, k in enumerate(kept) if k[2] >= 0.5 and k[0].feat is not None]
    if len(moving) >= 6:
        X = np.sqrt(np.array([f for _, f in moving], np.float32))     # Hellinger embedding
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        _, lab, cen = cv2.kmeans(X, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        spread = np.mean([np.linalg.norm(X[lab == c] - cen[c], axis=1).mean() for c in (0, 1) if (lab == c).any()])
        sep = float(np.linalg.norm(cen[0] - cen[1]))
        if (lab == 0).sum() >= 2 and (lab == 1).sum() >= 2 and sep > 1.5 * spread:
            teams_of = {i: int(l) for (i, _), l in zip(moving, lab)}
    for i, (t, a, w) in enumerate(kept):
        feet.append(a[:, :2]); heights.append(a[:, 2]); weights.append(np.full(len(a), w))
        tidx.append(np.full(len(a), teams_of.get(i, 2)))
    return (np.concatenate(feet), np.concatenate(heights), np.concatenate(weights), np.concatenate(tidx),
            bool(teams_of))


def from_tracks(tracks: list, size: tuple[int, int], focal_guess: float = 0.9) -> ViewPoint | None:
    arrs = track_arrays(tracks)
    if arrs is None:
        return None
    feet, heights, weights, teams, has_teams = arrs
    return from_layout(feet, heights, size, focal_guess, weights, teams if has_teams else None)
