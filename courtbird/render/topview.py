"""Top-view renderer. Knows nothing about video or detection: it only draws a
FrameState (court-space positions) onto a court diagram."""
from __future__ import annotations

import cv2
import numpy as np

from ..schema import FrameState, Polygon
from ..sports.spec import SportSpec

# Colours are BGR (OpenCV).
BG = (46, 38, 34)
FLOOR = (96, 122, 92)
LINE = (235, 235, 235)
NET = (60, 60, 230)
TEAM_COLORS = {0: (230, 120, 40), 1: (50, 50, 220)}  # blue, red
UNKNOWN_TEAM = (160, 160, 160)
BALL = (255, 255, 255)
BALL_EDGE = (30, 30, 30)
SHADE = (0, 0, 0)
SHADE_ALPHA = 0.55


class TopViewRenderer:
    def __init__(self, spec: SportSpec, px_per_m: float = 40.0, show_ids: bool = True):
        self.spec = spec
        self.s = px_per_m
        self.show_ids = show_ids
        m = spec.free_zone
        self.w = int(round((spec.length + 2 * m) * self.s))
        self.h = int(round((spec.width + 2 * m) * self.s))
        self._base = self._draw_court()

    # court metres -> pixel
    def to_px(self, x: float, y: float) -> tuple[int, int]:
        m = self.spec.free_zone
        return int(round((x + m) * self.s)), int(round((y + m) * self.s))

    def to_court(self, px: float, py: float) -> tuple[float, float]:
        """Inverse of to_px (pixel in the rendered image -> court metres)."""
        m = self.spec.free_zone
        return px / self.s - m, py / self.s - m

    def _draw_court(self) -> np.ndarray:
        sp, img = self.spec, np.full((self.h, self.w, 3), BG, np.uint8)
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), FLOOR, -1)
        # court area slightly lighter than free zone
        p0, p1 = self.to_px(0, 0), self.to_px(sp.length, sp.width)
        img[p0[1]:p1[1], p0[0]:p1[0]] = (118, 148, 110)
        t = max(2, int(0.05 * self.s))
        cv2.rectangle(img, p0, p1, LINE, t)
        for a, b in sp.lines:
            cv2.line(img, self.to_px(*a), self.to_px(*b), LINE, t)
        cv2.line(img, self.to_px(sp.net_x, -0.5), self.to_px(sp.net_x, sp.width + 0.5), NET, t + 2)
        return img

    def _fill_polys(self, img: np.ndarray, polys: list[Polygon]) -> None:
        overlay = img.copy()
        for poly in polys:
            pts = np.array([self.to_px(x, y) for x, y in poly], np.int32)
            cv2.fillPoly(overlay, [pts], SHADE)
        cv2.addWeighted(overlay, SHADE_ALPHA, img, 1 - SHADE_ALPHA, 0, img)

    def render(self, state: FrameState) -> np.ndarray:
        img = self._base.copy()
        if state.hidden_regions:
            self._fill_polys(img, state.hidden_regions)

        r = max(5, int(0.28 * self.s))
        for p in state.people:
            c = TEAM_COLORS.get(p.team, UNKNOWN_TEAM)
            pt = self.to_px(p.x, p.y)
            cv2.circle(img, pt, r, c, -1, cv2.LINE_AA)
            cv2.circle(img, pt, r, (255, 255, 255), 1, cv2.LINE_AA)
            if self.show_ids:
                label = str(p.id)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.putText(img, label, (pt[0] - tw // 2, pt[1] + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        b = state.ball
        if b is not None:
            # Ball: larger than a person marker's inner area, white, fainter when unsure.
            br = int(r * 0.8)
            pt = self.to_px(b.x, b.y)
            alpha = float(np.clip(b.confidence, 0.25, 1.0))
            overlay = img.copy()
            cv2.circle(overlay, pt, br + 2, BALL_EDGE, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, br, BALL, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
