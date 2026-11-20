"""Camera motion from the static background.

A tripod/handheld pan, tilt or zoom is (nearly) a pure camera rotation, and for a pure rotation the
image-to-image mapping of *any* static scene point, near or far, is a single homography. So we
track background features between consecutive frames, fit a homography with RANSAC (players and
other movers are outliers) and chain the steps: M_t maps pixels of the reference frame to pixels
of frame t. Positions seen in frame t are brought back to the reference view with M_t^-1, where a
single court calibration applies.

Chaining accumulates a little drift; `register` (SIFT, direct) is used to measure that and to
relate a calibration frame to the first analysed frame.
"""
from __future__ import annotations

import cv2
import numpy as np


def _scale_matrix(s: float) -> np.ndarray:
    return np.diag([s, s, 1.0])


class MotionTracker:
    """Keyframe-based registration: `update(frame)` returns M (reference -> this frame).

    Features are detected on a *keyframe* and tracked straight into each new frame, so the error of
    one frame does not feed into the next (chaining frame-to-frame homographies drifted by hundreds
    of pixels over a few minutes even with a static camera). When the view has changed enough that
    few keyframe features are still tracked, the current frame becomes the new keyframe, so drift
    only accumulates over keyframe switches. People are masked out of the feature detection.
    """

    def __init__(self, scale: float = 0.5, max_corners: int = 900, min_inliers: int = 30,
                 ransac_px: float = 1.5, rekey_inliers: int = 150, fb_check: bool = False,
                 method: int = cv2.RANSAC, grid: tuple[int, int] = (4, 5), quality: float = 0.002):
        self.fb_check, self.method, self.grid, self.quality = fb_check, method, grid, quality
        self.s, self.max_corners, self.min_inliers = scale, max_corners, min_inliers
        self.ransac_px, self.rekey_inliers = ransac_px, rekey_inliers
        self.M = np.eye(3)            # reference -> current
        self.key_M = np.eye(3)        # reference -> keyframe
        self._key_g: np.ndarray | None = None
        self._key_pts: np.ndarray | None = None
        self.inliers = 0
        self.ok = True
        self.keyframes = 0

    def _prepare(self, frame_bgr: np.ndarray, boxes: np.ndarray | None):
        g = cv2.cvtColor(cv2.resize(frame_bgr, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        if boxes is not None and len(boxes):
            for x1, y1, x2, y2 in np.asarray(boxes)[:, :4] * self.s:  # features on players move with them
                mask[max(0, int(y1) - 4):int(y2) + 4, max(0, int(x1) - 4):int(x2) + 4] = 0
        return g, mask

    def _detect(self, g: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Corners spread over the whole frame: detect per grid cell (with a low quality bar) so
        the weakly textured floor, where the court is, also constrains the homography. Detecting
        globally clusters features on the busy stands and the fit then extrapolates badly."""
        h, w = g.shape
        rows, cols = self.grid
        per_cell = max(10, self.max_corners // (rows * cols))
        out = []
        for r in range(rows):
            for c in range(cols):
                y0, y1, x0, x1 = r * h // rows, (r + 1) * h // rows, c * w // cols, (c + 1) * w // cols
                cell_mask = np.zeros_like(mask)
                cell_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
                pts = cv2.goodFeaturesToTrack(g, per_cell, self.quality, 6, mask=cell_mask)
                if pts is not None:
                    out.append(pts)
        return np.concatenate(out) if out else None

    def _set_key(self, g: np.ndarray, mask: np.ndarray, M: np.ndarray) -> None:
        self._key_g, self.key_M = g, M.copy()
        self._key_pts = self._detect(g, mask)
        self._n_key = 0 if self._key_pts is None else len(self._key_pts)
        self.keyframes += 1

    def update(self, frame_bgr: np.ndarray, boxes: np.ndarray | None = None) -> np.ndarray:
        g, mask = self._prepare(frame_bgr, boxes)
        if self._key_g is None or self._key_pts is None or self._n_key < self.min_inliers:
            self._set_key(g, mask, self.M)
            return self.M.copy()
        S = _scale_matrix(self.s)
        # prediction from the previous frame's motion initialises LK, so fast pans still track
        H_prev_half = S @ (self.M @ np.linalg.inv(self.key_M)) @ np.linalg.inv(S)
        init = cv2.perspectiveTransform(self._key_pts, H_prev_half)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._key_g, g, self._key_pts, init, winSize=(21, 21), maxLevel=3,
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        good = st.ravel() == 1
        if self.fb_check and good.any():  # forward-backward consistency: drop unreliable tracks
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, self._key_g, nxt, None, winSize=(21, 21), maxLevel=3)
            err = np.linalg.norm((back - self._key_pts).reshape(-1, 2), axis=1)
            good &= (st2.ravel() == 1) & (err < 0.5)
        self.ok = False
        if good.sum() >= self.min_inliers:
            H, inl = cv2.findHomography(self._key_pts[good], nxt[good], self.method, self.ransac_px)
            if H is not None and inl is not None and int(inl.sum()) >= self.min_inliers:
                self.inliers, self.ok = int(inl.sum()), True
                self.M = (np.linalg.inv(S) @ H @ S) @ self.key_M
                self.M /= self.M[2, 2]
        if not self.ok:
            self._set_key(g, mask, self.M)       # lost the keyframe: restart from the last good estimate
        elif self.inliers < self.rekey_inliers or self.inliers < 0.4 * self._n_key:
            self._set_key(g, mask, self.M)       # view drifted away from the keyframe
        return self.M.copy()


def register(frame_a: np.ndarray, frame_b: np.ndarray, scale: float = 0.5, min_inliers: int = 30,
             mask_a: np.ndarray | None = None, mask_b: np.ndarray | None = None) -> np.ndarray | None:
    """Direct SIFT registration: homography mapping pixels of frame_a to frame_b, or None."""
    def prep(f, m):
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if m is not None:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        return g, m
    ga, ma = prep(frame_a, mask_a)
    gb, mb = prep(frame_b, mask_b)
    sift = cv2.SIFT_create(2000)
    ka, da = sift.detectAndCompute(ga, ma)
    kb, db = sift.detectAndCompute(gb, mb)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    if H is None or int(inl.sum()) < min_inliers:
        return None
    S = _scale_matrix(scale)
    H = np.linalg.inv(S) @ H @ S
    return H / H[2, 2]


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    h = np.c_[pts, np.ones(len(pts))] @ M.T
    return h[:, :2] / h[:, 2:3]


def local_scale(M: np.ndarray, pt: np.ndarray) -> float:
    """Linear size change (sqrt of the Jacobian determinant) of the map M at pixel `pt`."""
    x, y = float(pt[0]), float(pt[1])
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    u = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / w
    v = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / w
    dudx, dudy = (M[0, 0] - u * M[2, 0]) / w, (M[0, 1] - u * M[2, 1]) / w
    dvdx, dvdy = (M[1, 0] - v * M[2, 0]) / w, (M[1, 1] - v * M[2, 1]) / w
    return float(np.sqrt(abs(dudx * dvdy - dudy * dvdx)))


def box_mask(shape: tuple[int, int], boxes: np.ndarray | None, pad: int = 6) -> np.ndarray | None:
    """255 where features may be taken, 0 over people (full-resolution `shape` = (h, w))."""
    if boxes is None or len(boxes) == 0:
        return None
    m = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in np.asarray(boxes)[:, :4]:
        m[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad] = 0
    return m


def snap_static(M: np.ndarray, size: tuple[int, int], tol: float = 1.0) -> np.ndarray:
    """Return the identity when M moves the image corners by less than `tol` px: removes the
    sub-pixel tracking jitter of a camera that is really standing still."""
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    if np.abs(apply(M, corners) - corners).max() < tol:
        return np.eye(3)
    return M


def stabilize_detections(dets: list, Minv: np.ndarray) -> None:
    """In place: bring detections (foot point, box height) from the current frame back to the
    reference view, so the tracker and the single court calibration work in one fixed frame."""
    for d in dets:
        d.height = float(d.height * local_scale(Minv, d.foot))
        d.foot = apply(Minv, d.foot)[0]
