from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..render import TopViewRenderer
from ..schema import FrameState
from .model import EditSession

PICK_RADIUS_M = 0.6


def to_qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    return QImage(np.ascontiguousarray(bgr).data, w, h, 3 * w, QImage.Format_BGR888).copy()


class ImageView(QWidget):
    """Draws an image scaled to fit while keeping its aspect ratio."""

    def __init__(self):
        super().__init__()
        self._img: QImage | None = None
        self.setMinimumSize(240, 160)

    def set_image(self, bgr: np.ndarray | None) -> None:
        self._img = to_qimage(bgr) if bgr is not None else None
        self.update()

    def _fit(self) -> tuple[float, float, float]:
        if self._img is None:
            return 1.0, 0.0, 0.0
        s = min(self.width() / self._img.width(), self.height() / self._img.height())
        return s, (self.width() - self._img.width() * s) / 2, (self.height() - self._img.height() * s) / 2

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        if self._img is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            s, ox, oy = self._fit()
            p.drawImage(QRectF(ox, oy, self._img.width() * s, self._img.height() * s), self._img)


class TopViewWidget(ImageView):
    """Top-view court. Click selects, drag moves, double-click adds a person,
    right-click places (or removes, when on the ball) the ball."""

    changed = Signal()            # an edit was made
    selection_changed = Signal()  # selected object changed

    def __init__(self, renderer: TopViewRenderer):
        super().__init__()
        self.renderer = renderer
        self.session: EditSession | None = None
        self.pos = 0
        self.selected: tuple[str, int] | None = None  # ("person", id) | ("ball", 0)
        self._dragging = False
        self._checkpointed = False
        self.cam_pos: tuple[float, float] | None = None  # estimated camera position, court metres
        self.cam_label = "CAM"
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def refresh(self) -> None:
        if self.session is None or len(self.session) == 0:
            self.set_image(self.renderer.render(FrameState()))
            return
        self.set_image(self.renderer.render(self.session.frame(self.pos)))

    # --- coordinates -----------------------------------------------------------
    def _to_court(self, pt: QPointF) -> tuple[float, float]:
        s, ox, oy = self._fit()
        return self.renderer.to_court((pt.x() - ox) / s, (pt.y() - oy) / s)

    def _to_widget(self, x: float, y: float) -> QPointF:
        s, ox, oy = self._fit()
        px, py = self.renderer.to_px(x, y)
        return QPointF(ox + px * s, oy + py * s)

    def _pick(self, x: float, y: float):
        f = self.session.frame(self.pos)
        best, best_d = None, PICK_RADIUS_M
        if f.ball is not None:
            d = float(np.hypot(f.ball.x - x, f.ball.y - y))
            if d < best_d:
                best, best_d = ("ball", 0), d
        for p in f.people:
            d = float(np.hypot(p.x - x, p.y - y))
            if d < best_d:
                best, best_d = ("person", p.id), d
        return best

    # --- painting ----------------------------------------------------------------
    def _draw_camera(self) -> None:
        """Camera glyph where the video was shot from. Cameras are usually outside the drawn area,
        so the glyph sits on its border in the camera's direction, pointing at the court."""
        if self.cam_pos is None or self._img is None:
            return
        s, ox, oy = self._fit()
        spec = self.renderer.spec
        c = self._to_widget(spec.length / 2, spec.width / 2)
        q = self._to_widget(*self.cam_pos)
        x0, y0, x1, y1 = ox + 16, oy + 16, ox + self._img.width() * s - 16, oy + self._img.height() * s - 16
        dx, dy = q.x() - c.x(), q.y() - c.y()
        k = 1.0
        if not (x0 <= q.x() <= x1 and y0 <= q.y() <= y1):
            ks = []
            if dx:
                ks += [(x0 - c.x()) / dx, (x1 - c.x()) / dx]
            if dy:
                ks += [(y0 - c.y()) / dy, (y1 - c.y()) / dy]
            k = min([v for v in ks if v > 0] or [1.0])
        pos = QPointF(c.x() + dx * k, c.y() + dy * k)
        n = max(float(np.hypot(dx, dy)), 1e-6)
        ux, uy = -dx / n, -dy / n  # towards the court
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.setBrush(QColor(30, 30, 30, 230))
        p.drawEllipse(pos, 10, 10)
        tip = QPointF(pos.x() + ux * 26, pos.y() + uy * 26)
        left = QPointF(pos.x() + ux * 11 - uy * 7, pos.y() + uy * 11 + ux * 7)
        right = QPointF(pos.x() + ux * 11 + uy * 7, pos.y() + uy * 11 - ux * 7)
        p.setBrush(QColor(255, 220, 0))
        p.drawPolygon([tip, left, right])
        p.setPen(QColor(255, 255, 255))
        p.drawText(QPointF(pos.x() - 14, pos.y() - 14), self.cam_label or "CAM")

    def paintEvent(self, e):
        super().paintEvent(e)
        self._draw_camera()
        if self.session is None or not self.selected or not len(self.session):
            return
        f = self.session.frame(self.pos)
        kind, ident = self.selected
        obj = f.ball if kind == "ball" else next((q for q in f.people if q.id == ident), None)
        if obj is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 220, 0), 2))
        p.setBrush(Qt.NoBrush)
        r = 0.45 * self.renderer.s * self._fit()[0]
        p.drawEllipse(self._to_widget(obj.x, obj.y), r, r)

    # --- mouse -------------------------------------------------------------------
    def _select(self, sel) -> None:
        if sel != self.selected:
            self.selected = sel
            self.selection_changed.emit()
        self.update()

    def mousePressEvent(self, e):
        if self.session is None or not len(self.session):
            return
        self.setFocus()
        x, y = self._to_court(e.position())
        hit = self._pick(x, y)
        if e.button() == Qt.LeftButton:
            self._select(hit)
            if hit:
                self._dragging, self._checkpointed = True, False
        elif e.button() == Qt.RightButton:
            if hit == ("ball", 0):
                self.session.delete_ball(self.pos)
                self._select(None)
            else:
                self.session.set_ball(self.pos, x, y)
                self._select(("ball", 0))
            self.changed.emit(); self.refresh()

    def mouseMoveEvent(self, e):
        if not self._dragging or not self.selected:
            return
        x, y = self._to_court(e.position())
        kind, ident = self.selected
        if not self._checkpointed:  # one undo step per drag, taken on the first movement
            self.session.checkpoint([self.pos])
            self._checkpointed = True
        if kind == "ball":
            self.session.set_ball(self.pos, x, y, snapshot=False)
        else:
            self.session.move_person(self.pos, ident, x, y, snapshot=False)
        self.refresh()

    def mouseReleaseEvent(self, e):
        if self._dragging:
            self._dragging = False
            if self._checkpointed:
                self.changed.emit()

    def mouseDoubleClickEvent(self, e):
        if self.session is None or not len(self.session) or e.button() != Qt.LeftButton:
            return
        x, y = self._to_court(e.position())
        if self._pick(x, y) is None:
            p = self.session.add_person(self.pos, x, y)
            self._dragging = False
            self._select(("person", p.id))
            self.changed.emit(); self.refresh()
