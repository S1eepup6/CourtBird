"""Classical line-based court calibration refinement.

Idea: the white court lines leave "line evidence" in the image (thin bright ridges). A calibration
is good when the court model's lines, projected into the image, lie on that evidence. We score a
calibration by the mean distance (chamfer, capped) from points sampled along the model lines to the
nearest evidence, and optimise the 4 court corners' pixel positions to minimise it, starting from
a rough calibration (a hand click, the previous frame's result, ...). Pure OpenCV/NumPy/SciPy: no
learned model, a few milliseconds per score evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import Bounds, minimize

from ..sports.spec import SportSpec
from .calibration import Calibration, landmarks

CORNERS = ("corner_near_left", "corner_near_right", "corner_far_right", "corner_far_left")


def line_evidence(frame_bgr: np.ndarray, exclude_boxes: np.ndarray | None = None,
                  pad: int = 6) -> np.ndarray:
    """Binary map (uint8 0/255) of thin bright ridges.

    Two channels are combined: a top-hat on grey (lines on dark/teal floors) and a top-hat on
    min(B, G) (white lines on orange/red floors, where grey contrast is weak). Boxes of people can
    be excluded: their limbs and shoes look like short lines.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mn = np.minimum(frame_bgr[..., 0], frame_bgr[..., 1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    th_g = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    th_m = cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, k)
    ev = (((th_g > 30) & (hsv[..., 1] < 90)) | (th_m > 30)).astype(np.uint8) * 255
    ev = cv2.morphologyEx(ev, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Colour boundaries: many courts are a coloured area (not white lines) next to a different
    # floor colour. Strong edges of the heavily blurred Lab image trace those boundaries.
    lab = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (0, 0), 2.0), cv2.COLOR_BGR2LAB)
    edges = np.zeros_like(gray)
    for c in range(3):
        edges |= cv2.Canny(lab[..., c], 12, 30)
    ev = np.maximum(ev, edges)
    # keep only elongated components: lines, not blobs (jerseys, lights, shoes)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ev, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        fill = area / float(w * h)
        aspect = max(w, h) / max(1, min(w, h))
        # a line is long and either axis-aligned/thin (high aspect) or a sparse diagonal (low fill)
        keep[i] = max(w, h) >= 25 and (aspect >= 3 or fill < 0.45)
    ev = (keep[lab] * 255).astype(np.uint8)
    if exclude_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(exclude_boxes)[:, :4]:
            ev[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad] = 0
    return ev


def model_lines(spec: SportSpec) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Court lines painted on the floor (the net is not one of them)."""
    L, W = spec.length, spec.width
    segs = [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0))]
    return segs + list(spec.lines)


def sample_points(spec: SportSpec, step: float = 0.2) -> np.ndarray:
    pts = []
    for (x1, y1), (x2, y2) in model_lines(spec):
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], 1))
    return np.concatenate(pts)


@dataclass
class Fit:
    calib: Calibration
    cost: float          # mean capped distance (px) of visible model-line points to line evidence
    coverage: float      # fraction of model-line points inside the image
    seed_cost: float


class Refiner:
    def __init__(self, spec: SportSpec, evidence: np.ndarray, step: float = 0.2):
        self.spec = spec
        self.h, self.w = evidence.shape
        self.dt = cv2.distanceTransform(255 - evidence, cv2.DIST_L2, 3).astype(np.float32)
        self.pts = sample_points(spec, step)
        lm = landmarks(spec)
        self.corner_court = np.array([lm[c] for c in CORNERS], np.float32)

    def _H_court_to_img(self, corners_img: np.ndarray) -> np.ndarray | None:
        try:
            return cv2.getPerspectiveTransform(self.corner_court, corners_img.astype(np.float32))
        except cv2.error:
            return None

    def cost(self, corners_img: np.ndarray, cap: float) -> tuple[float, float]:
        """(mean capped distance, coverage). Points outside the image are ignored but reduce
        coverage; a hypothesis seeing too little of the court is penalised."""
        H = self._H_court_to_img(corners_img)
        if H is None:
            return 1e3, 0.0
        p = np.c_[self.pts, np.ones(len(self.pts))] @ H.T
        w = p[:, 2]
        ok = w > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = p[:, :2] / w[:, None]
        inside = ok & (xy[:, 0] >= 0) & (xy[:, 0] < self.w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < self.h - 1)
        cov = float(inside.mean())
        if inside.sum() < 20:
            return 1e3, cov
        d = self.dt[xy[inside, 1].astype(int), xy[inside, 0].astype(int)]
        return float(np.minimum(d, cap).mean()), cov

    def _plausible(self, corners: np.ndarray, seed: np.ndarray) -> bool:
        # convex quad with a consistent winding order, and not wildly different in size
        c = corners
        def cross2(a, b):
            return a[0] * b[1] - a[1] * b[0]
        cross = [cross2(c[(i + 1) % 4] - c[i], c[(i + 2) % 4] - c[(i + 1) % 4]) for i in range(4)]
        if not (all(x > 0 for x in cross) or all(x < 0 for x in cross)):
            return False
        a = abs(cv2.contourArea(c.astype(np.float32))); a0 = abs(cv2.contourArea(seed.astype(np.float32)))
        return 0.4 < a / max(a0, 1.0) < 2.5

    def refine(self, seed_corners: np.ndarray, caps=(30.0, 16.0, 8.0), reg: float = 0.002, max_shift: float = 35.0) -> tuple[np.ndarray, float]:
        seed = np.asarray(seed_corners, np.float64)
        x0 = seed.ravel()
        best = x0
        for cap in caps:
            def f(x):
                c = x.reshape(4, 2)
                if not self._plausible(c, seed):
                    return 1e3
                cost, cov = self.cost(c, cap)
                return cost / cap + reg * float(np.mean((x - x0) ** 2) ** 0.5) / cap + (1.0 - cov) * 0.5
            res = minimize(f, best, method="Powell", bounds=Bounds(x0 - max_shift, x0 + max_shift),
                           options={"xtol": 0.5, "ftol": 1e-4, "maxiter": 40})
            best = res.x
        return best.reshape(4, 2), self.cost(best.reshape(4, 2), caps[-1])[0]


def corners_of(calib: Calibration, spec: SportSpec) -> np.ndarray:
    lm = landmarks(spec)
    return calib.court_to_image(np.array([lm[c] for c in CORNERS]))


def calibration_from_corners(spec: SportSpec, corners: np.ndarray, image_size: tuple[int, int]) -> Calibration:
    return Calibration.from_points(spec, [(n, float(x), float(y)) for n, (x, y) in zip(CORNERS, corners)], image_size)


def refine_calibration(seed: Calibration, frame_bgr: np.ndarray, spec: SportSpec,
                       exclude_boxes: np.ndarray | None = None, max_shift: float = 35.0) -> Fit:
    """Snap a rough calibration onto the line evidence of `frame_bgr`. Each corner may move at
    most `max_shift` pixels, so strong unrelated lines further away (other sports' markings) cannot
    capture the fit: the seed must already be close."""
    ev = line_evidence(frame_bgr, exclude_boxes)
    r = Refiner(spec, ev)
    c0 = corners_of(seed, spec)
    seed_cost = r.cost(c0, 8.0)[0]
    c1, cost = r.refine(c0, max_shift=max_shift)
    h, w = frame_bgr.shape[:2]
    return Fit(calibration_from_corners(spec, c1, (w, h)), cost, r.cost(c1, 8.0)[1], seed_cost)
