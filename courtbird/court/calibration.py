"""Image <-> court-plane mapping from point correspondences (homography)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..schema import Polygon
from ..sports.spec import SportSpec


def landmarks(spec: SportSpec) -> dict[str, tuple[float, float]]:
    """Named court-space points a user can click in the image.

    Court x runs from the near end line (0) to the far end line (L); y runs across.
    Posts stand `post_offset` outside the sidelines, so they are approximate.
    """
    L, W, n, o = spec.length, spec.width, spec.net_x, spec.post_offset
    pts = {
        "corner_near_left": (0.0, 0.0), "corner_near_right": (0.0, W),
        "corner_far_left": (L, 0.0), "corner_far_right": (L, W),
        "net_left_sideline": (n, 0.0), "net_right_sideline": (n, W),
        "post_left": (n, -o), "post_right": (n, W + o),
    }
    for i, (a, b) in enumerate(spec.lines):
        pts[f"line{i}_left"], pts[f"line{i}_right"] = a, b
    return pts


@dataclass
class Calibration:
    sport: str
    points: list[tuple[str, float, float]]  # (landmark name, pixel x, pixel y)
    image_size: tuple[int, int]  # (width, height) the pixels refer to
    H: np.ndarray  # image -> court, 3x3
    sec: float | None = None  # video time of the frame the points were clicked in (None = unknown)

    @classmethod
    def from_points(cls, spec: SportSpec, points: list[tuple[str, float, float]],
                    image_size: tuple[int, int]) -> "Calibration":
        if len(points) < 4:
            raise ValueError("need at least 4 point correspondences")
        lm = landmarks(spec)
        img = np.array([[px, py] for _, px, py in points], np.float64)
        court = np.array([lm[name] for name, _, _ in points], np.float64)
        H, _ = cv2.findHomography(img, court, 0)
        if H is None:
            raise ValueError("homography estimation failed (degenerate points?)")
        return cls(spec.name, points, image_size, H)

    # --- mapping -------------------------------------------------------------
    def image_to_court(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """image -> court like image_to_court, but also returns a validity mask. Points on the
        far side of the horizon line (stands, ceiling) project to garbage; they are marked
        invalid (NaN) instead of being mirrored onto the court."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        ref = np.c_[[[x, y] for _, x, y in self.points], np.ones(len(self.points))] @ self.H.T
        sign = np.sign(np.median(ref[:, 2])) or 1.0
        h = np.c_[pts, np.ones(len(pts))] @ self.H.T
        w = h[:, 2] * sign
        valid = w > 1e-9
        xy = np.full((len(pts), 2), np.nan)
        xy[valid] = h[valid, :2] / h[valid, 2:3]
        valid &= np.all(np.abs(np.nan_to_num(xy, nan=1e9)) < 80.0, axis=1)
        xy[~valid] = np.nan
        return xy, valid

    def court_to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def reprojection_error(self, spec: SportSpec) -> float:
        """Mean court-space error (metres) over the calibration points."""
        lm = landmarks(spec)
        got = self.image_to_court(np.array([[x, y] for _, x, y in self.points]))
        want = np.array([lm[n] for n, _, _ in self.points])
        return float(np.linalg.norm(got - want, axis=1).mean())

    # --- visibility ----------------------------------------------------------
    def visibility(self, spec: SportSpec, step: float = 0.25, pad_px: float = 0.0
                   ) -> tuple[list[Polygon], list[Polygon]]:
        """Split the court + free zone into (visible, hidden) polygons by checking
        which cells project inside the image."""
        m = spec.free_zone
        xs = np.arange(-m, spec.length + m + step / 2, step)
        ys = np.arange(-m, spec.width + m + step / 2, step)
        gx, gy = np.meshgrid(xs, ys)
        court = np.stack([gx.ravel(), gy.ravel()], 1)
        Hinv = np.linalg.inv(self.H)
        h = np.c_[court, np.ones(len(court))] @ Hinv.T
        w = h[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = h[:, 0] / w
            py = h[:, 1] / w
        W_, H_ = self.image_size
        inside = (w > 0) & (px >= pad_px) & (px < W_ - pad_px) & (py >= pad_px) & (py < H_ - pad_px)
        mask = inside.reshape(gx.shape).astype(np.uint8)
        return _mask_polys(mask, xs[0], ys[0], step), _mask_polys(1 - mask, xs[0], ys[0], step)

    # --- io ------------------------------------------------------------------
    def moved(self, M: np.ndarray) -> "Calibration":
        """The same calibration seen through a camera motion M (reference-frame pixels -> pixels of
        another frame): every calibration point is carried along, so the court projects correctly
        into that frame."""
        pts = np.array([[x, y] for _, x, y in self.points], np.float64).reshape(-1, 1, 2)
        new = cv2.perspectiveTransform(pts, np.asarray(M, np.float64)).reshape(-1, 2)
        points = [(n, float(x), float(y)) for (n, _, _), (x, y) in zip(self.points, new)]
        H, _ = cv2.findHomography(np.array([[x, y] for _, x, y in points]),
                                  np.array([[*self._court_of(n)] for n, _, _ in points]), 0)
        return Calibration(self.sport, points, self.image_size, H, self.sec)

    def _court_of(self, name: str) -> tuple[float, float]:
        from ..sports import SPORTS
        return landmarks(SPORTS[self.sport])[name]

    def to_dict(self) -> dict:
        d = {"sport": self.sport, "image_size": list(self.image_size),
             "points": [[n, x, y] for n, x, y in self.points]}
        if self.sec is not None:
            d["sec"] = self.sec
        return d

    @classmethod
    def from_dict(cls, d: dict, spec: SportSpec) -> "Calibration":
        if d["sport"] != spec.name:
            raise ValueError(f"calibration is for {d['sport']}, not {spec.name}")
        calib = cls.from_points(spec, [tuple(p) for p in d["points"]], tuple(d["image_size"]))
        calib.sec = d.get("sec")
        return calib

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, spec: SportSpec) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), spec)


def _mask_polys(mask: np.ndarray, x0: float, y0: float, step: float) -> list[Polygon]:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cs, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 4:
            continue
        # contour pixel (col,row) in padded mask -> court metres (cell centres, +/- half a cell)
        out.append([(float(x0 + (px - 1) * step), float(y0 + (py - 1) * step)) for px, py in c.reshape(-1, 2)])
    return out
