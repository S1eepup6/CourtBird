"""Online multi-person tracker working in IMAGE space.

Why image space: identity should not depend on the court calibration. A calibration that is a
little off (or fixed later by hand) stretches distances in the far half of the court, which wrecks
distance gating there; pixels with a per-person scale do not have that problem, and a track stays
valid when the calibration changes. Distances are measured in "person heights" (the track's
smoothed box height), which makes them roughly metric and independent of depth.

Per track: constant-velocity Kalman filter on the foot point (pixels) + running colour-histogram
appearance. Association is ByteTrack-style two-stage Hungarian matching:
  1. confident detections vs all live tracks (distance + appearance),
  2. weak detections vs the tracks stage 1 left unmatched (distance only),
so an occluded player's weak, partial box keeps the track alive instead of spawning a new id.
Unmatched tracks coast on the Kalman prediction for `max_age_s`. The tracklets are linked into
full tracks offline by `postprocess.stitch`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6
PERSON_M = 1.8  # nominal person height in metres, only used to convert metre thresholds


@dataclass
class Det:
    foot: np.ndarray                 # (x, y) pixel: bottom-centre of the box
    height: float                    # box height, pixels
    score: float
    feet_ok: bool = True             # False when the box is cut off by the image bottom
    feat: np.ndarray | None = None   # appearance histogram (L1 normalised)


def appearance(frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """HSV histogram of the torso/shorts area (shirt colour separates teams and staff).
    Value is kept coarse-but-present so black and grey kits differ."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None
    ya, yb = int(y1 + 0.18 * h), int(y1 + 0.62 * h)
    xa, xb = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    H, W = frame_bgr.shape[:2]
    xa, xb, ya, yb = max(0, xa), min(W, xb), max(0, ya), min(H, yb)
    if xb - xa < 2 or yb - ya < 2:
        return None
    hsv = cv2.cvtColor(frame_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 6], [0, 180, 0, 256, 0, 256]).ravel()
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else None


def app_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Bhattacharyya distance in [0, 1]; 0.5 (uninformative) if either feature is missing."""
    if a is None or b is None:
        return 0.5
    bc = float(np.sum(np.sqrt(a * b)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


@dataclass
class Track:
    id: int
    x: np.ndarray                    # [x, y, vx, vy] foot point, pixels
    P: np.ndarray
    scale: float                     # smoothed box height, pixels
    feat: np.ndarray | None
    start: int
    last: int
    lost: int = 0
    hits: int = 1
    # (pos, foot_x, foot_y, height, feet_ok) for every matched frame
    history: list[tuple[int, float, float, float, bool]] = field(default_factory=list)

    @property
    def xy(self) -> np.ndarray:
        return self.x[:2]


class Tracker:
    def __init__(self, dt: float, hi_score: float = 0.3, max_age_s: float = 2.0, gate_m: float = 1.0,
                 vmax: float = 5.0, meas_sigma_m: float = 0.3, accel_sigma_m: float = 3.0,
                 app_weight: float = 0.6, app_gate: float = 0.75):
        self.dt, self.hi, self.vmax = dt, hi_score, vmax
        self.max_age = max(1, int(round(max_age_s / dt)))
        self.gate_h = gate_m / PERSON_M            # gate, in person heights
        self.vmax_h = vmax / PERSON_M
        self.meas_h = meas_sigma_m / PERSON_M
        self.accel_h = accel_sigma_m / PERSON_M
        self.app_weight, self.app_gate = app_weight, app_gate
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def _Q(self, s: float) -> np.ndarray:
        dt, q = self.dt, (self.accel_h * s) ** 2
        return q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                             [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])

    def _predict(self, t: Track) -> None:
        if t.lost > 0:
            t.x[2:] *= 0.9  # a coasting track must not run away on a stale velocity
        t.x = self.F @ t.x
        t.P = self.F @ t.P @ self.F.T + self._Q(t.scale)

    def _update(self, t: Track, d: Det, pos: int) -> None:
        R = np.eye(2) * (self.meas_h * t.scale) ** 2
        S = self.Hm @ t.P @ self.Hm.T + R
        K = t.P @ self.Hm.T @ np.linalg.inv(S)
        t.x = t.x + K @ (d.foot - self.Hm @ t.x)
        t.P = (np.eye(4) - K @ self.Hm) @ t.P
        if d.feat is not None:
            t.feat = d.feat if t.feat is None else (0.9 * t.feat + 0.1 * d.feat)
            t.feat = t.feat / t.feat.sum()
        t.scale = 0.9 * t.scale + 0.1 * d.height
        t.lost, t.last = 0, pos
        t.hits += 1
        t.history.append((pos, float(t.x[0]), float(t.x[1]), t.scale, d.feet_ok))

    def _new(self, d: Det, pos: int) -> None:
        s = d.height
        P = np.diag([(self.meas_h * s) ** 2] * 2 + [(1.0 * s) ** 2] * 2)
        t = Track(self._next, np.array([d.foot[0], d.foot[1], 0.0, 0.0]), P, s, d.feat, pos, pos)
        t.history.append((pos, float(d.foot[0]), float(d.foot[1]), s, d.feet_ok))
        self._next += 1
        self.tracks.append(t)

    def _match(self, tracks: list[Track], dets: list[Det], use_app: bool):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = np.full((len(tracks), len(dets)), BIG)
        for i, t in enumerate(tracks):
            gate = (self.gate_h + self.vmax_h * self.dt * t.lost) * t.scale
            for j, d in enumerate(dets):
                dist = float(np.hypot(*(t.xy - d.foot)))
                if dist > gate:
                    continue
                c = dist / gate
                if use_app:
                    a = app_distance(t.feat, d.feat)
                    if a > self.app_gate:
                        continue
                    c += self.app_weight * a
                cost[i, j] = c
        rows, cols = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < BIG]
        mt, md = {i for i, _ in pairs}, {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def update(self, pos: int, dets: list[Det]) -> None:
        for t in self.tracks:
            self._predict(t)
        hi = [d for d in dets if d.score >= self.hi]
        lo = [d for d in dets if d.score < self.hi]

        pairs, un_t, un_hi = self._match(self.tracks, hi, use_app=True)
        for i, j in pairs:
            self._update(self.tracks[i], hi[j], pos)
        # stage 2: weak boxes may only continue tracks that were seen last frame
        cand = [self.tracks[i] for i in un_t if self.tracks[i].lost == 0]
        pairs2, _, _ = self._match(cand, lo, use_app=False)
        for i, j in pairs2:
            self._update(cand[i], lo[j], pos)
        for j in un_hi:  # only confident detections may start a track
            d = hi[j]
            if all(float(np.hypot(*(t.xy - d.foot))) > 0.3 * t.scale for t in self.tracks if t.last == pos):
                self._new(d, pos)
        for t in self.tracks:
            if t.last != pos:
                t.lost += 1
        alive = []
        for t in self.tracks:
            (self.finished if t.lost > self.max_age else alive).append(t)
        self.tracks = alive

    def all_tracks(self) -> list[Track]:
        return self.finished + self.tracks
