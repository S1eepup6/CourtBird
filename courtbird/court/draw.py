"""Debug drawing of a calibrated court over a camera frame."""
from __future__ import annotations

import cv2
import numpy as np

from ..sports.spec import SportSpec
from .calibration import Calibration


def draw_court(frame: np.ndarray, calib: Calibration, spec: SportSpec, color=(0, 255, 255),
               show_points: bool = True) -> np.ndarray:
    def poly(pts, c, t=2):
        p = calib.court_to_image(np.array(pts)).astype(np.int32)
        cv2.polylines(frame, [p], True, c, t, cv2.LINE_AA)
    L, W, m = spec.length, spec.width, spec.free_zone
    poly([(0, 0), (L, 0), (L, W), (0, W)], color)
    for a, b in spec.lines:
        poly([a, b], color, 1)
    poly([(L / 2, -spec.post_offset), (L / 2, W + spec.post_offset)], (0, 0, 255), 2)
    poly([(-m, -m), (L + m, -m), (L + m, W + m), (-m, W + m)], (255, 0, 255), 1)
    if show_points:
        for _, x, y in calib.points:
            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), 2)
    return frame
