from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from .. import record as rec_io
from ..court import Calibration
from ..court.draw import draw_court
from ..render import TopViewRenderer
from ..render.topview import TEAM_COLORS, UNKNOWN_TEAM, half_teams
from ..sports import SPORTS
from .analyze_dialog import AnalyzeDialog
from .model import EditSession
from .video import VideoSource
from .widgets import ImageView, TopViewWidget

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.spec = SPORTS["volleyball"]
        self.session: EditSession | None = None
        self.path: Path | None = None
        self.video: VideoSource | None = None
        self.calib: Calibration | None = None
        self.pos = 0
        self._acc = 0.0

        self.top = TopViewWidget(TopViewRenderer(self.spec, team_zones=True))
        self.video_view = ImageView()
        self.video_view.setVisible(False)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.video_view)
        split.addWidget(self.top)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([700, 700])
        self.splitter = split

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(44)
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶|")
        self.prev_btn.setFixedWidth(36)
        self.next_btn.setFixedWidth(36)
        self.slider = QSlider(Qt.Horizontal)
        self.speed = QComboBox()
        self.speed.addItems([f"{s}x" for s in SPEEDS])
        self.speed.setCurrentIndex(2)
        self.ids = QCheckBox("ID 표시")
        self.ids.setChecked(True)
        self.zones = QCheckBox("팀 영역 표시")
        self.zones.setChecked(True)
        self.time_label = QLabel("0:00.0")
        self.time_label.setMinimumWidth(110)
        for w in (self.play_btn, self.prev_btn, self.next_btn, self.slider, self.speed, self.ids, self.zones):
            w.setFocusPolicy(Qt.NoFocus)
        bar = QHBoxLayout()
        for w in (self.play_btn, self.prev_btn, self.next_btn):
            bar.addWidget(w)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.time_label)
        bar.addWidget(self.speed)
        bar.addWidget(self.ids)
        bar.addWidget(self.zones)
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(split, 1)
        lay.addLayout(bar)
        self.setCentralWidget(root)
        self.resize(1400, 640)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(1))
        self.slider.valueChanged.connect(self._slider)
        self.ids.toggled.connect(self._ids_toggled)
        self.zones.toggled.connect(self._zones_toggled)
        self.top.changed.connect(self._edited)
        self.top.selection_changed.connect(self._status)
        self._build_menu()
        self._update_title()
        self._status()

    # --- menu ------------------------------------------------------------------------
    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_menu(self):
        f = self.menuBar().addMenu("파일")
        self._act(f, "기록 열기…", self.open_record, "Ctrl+O")
        self._act(f, "영상 분석…", self.analyze_dialog, "Ctrl+N")
        f.addSeparator()
        self._act(f, "영상 연결…", self.link_video)
        self._act(f, "코트 보정 파일 연결…", self.link_calibration)
        f.addSeparator()
        self._act(f, "저장", self.save, "Ctrl+S")
        self._act(f, "다른 이름으로 저장…", self.save_as, "Ctrl+Shift+S")
        f.addSeparator()
        self._act(f, "종료", self.close, "Ctrl+Q")
        e = self.menuBar().addMenu("편집")
        self._act(e, "실행 취소", self.undo, "Ctrl+Z")
        self._act(e, "선택 삭제", self.delete_selected, "Delete")
        self._act(e, "팀 전환 (이 프레임)", lambda: self.toggle_team(False), "T")
        self._act(e, "팀 전환 (이후 모든 프레임)", lambda: self.toggle_team(True), "Shift+T")
        self._act(e, "ID 변경…", self.rename_selected, "I")
        self._act(e, "파랑/빨강 팀 서로 바꾸기 (전체)", self.swap_teams, "Ctrl+T")
        p = self.menuBar().addMenu("재생")
        self._act(p, "재생/일시정지", self.toggle_play, "Space")
        self._act(p, "이전 프레임", lambda: self.step(-1), "Left")
        self._act(p, "다음 프레임", lambda: self.step(1), "Right")
        self._act(p, "1초 뒤로", lambda: self.step(-self._fps()), "Ctrl+Left")
        self._act(p, "1초 앞으로", lambda: self.step(self._fps()), "Ctrl+Right")
        h = self.menuBar().addMenu("도움말")
        self._act(h, "사용법", self.show_help)

    def show_help(self):
        QMessageBox.information(self, "사용법", (
            "탑뷰에서:\n"
            "• 클릭: 선택 / 드래그: 위치 이동\n• 빈 곳 더블클릭: 선수 추가\n"
            "• 우클릭: 공 위치 지정 (공 위 우클릭: 공 삭제)\n"
            "• Delete: 삭제 / T: 팀 전환 (Shift+T: 이후 프레임 모두) / I: ID 변경\n"
            "• Ctrl+Z: 실행 취소\n\n"
            "재생: Space, ←/→ 한 프레임, Ctrl+←/→ 1초"))

    # --- loading / saving -----------------------------------------------------------------
    def open_record(self):
        path, _ = QFileDialog.getOpenFileName(self, "기록 열기", "", "CourtBird 기록 (*.cbrec);;All (*)")
        if path:
            self.load_record_file(Path(path))

    def load_record_file(self, path: Path):
        if not self._confirm_discard():
            return
        try:
            record = rec_io.load(path)
        except Exception as e:
            QMessageBox.critical(self, "기록 열기 실패", str(e))
            return
        self.path = path
        self._set_record(record)
        for cand in (path.parent / record.source, path.parent.parent / record.source):
            if record.source and cand.is_file():
                self.attach_video(str(cand))
                break

    def _set_record(self, record: rec_io.Record):
        self.timer.stop()
        self.play_btn.setText("▶")
        self.spec = SPORTS[record.sport]
        self.top.renderer = TopViewRenderer(self.spec, show_ids=self.ids.isChecked(), team_zones=self.zones.isChecked())
        self.session = EditSession(record, self.spec)
        self.top.session = self.session
        self.top.selected = None
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.session) - 1))
        self.slider.blockSignals(False)
        if record.calibration:
            try:
                self.calib = Calibration.from_dict(record.calibration, self.spec)
            except Exception:
                self.calib = None
        self.pos = 0
        self.show_frame(0)
        self._update_title()

    def analyze_dialog(self):
        if not self._confirm_discard():
            return
        dlg = AnalyzeDialog(self.spec, self, video=self.video.path if self.video else "")
        if dlg.exec() and dlg.record is not None:
            self.path = None
            self._set_record(dlg.record)
            self.session.dirty = True
            self.attach_video(dlg.video_path)
            # the record carries the calibration in the analysis start view (already set by _set_record)
            self.show_frame(self.pos)
            self._update_title()

    def link_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 연결", "", "Video (*.mp4 *.mov *.avi *.mkv);;All (*)")
        if path:
            self.attach_video(path)

    def attach_video(self, path: str):
        try:
            v = VideoSource(path)
        except OSError as e:
            QMessageBox.warning(self, "영상 열기 실패", str(e))
            return
        if self.video:
            self.video.close()
        self.video = v
        self.video_view.setVisible(True)
        self.splitter.setSizes([700, 700])
        self.show_frame(self.pos)

    def link_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "보정 파일 연결", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.calib = Calibration.load(path, self.spec)
        except Exception as e:
            QMessageBox.warning(self, "보정 파일 오류", str(e))
            return
        self.show_frame(self.pos)

    def save(self):
        if self.session is None:
            return
        if self.path is None:
            return self.save_as()
        rec_io.save(self.session.record, self.path)
        self.session.dirty = False
        self._update_title()

    def save_as(self):
        if self.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "기록 저장", str(self.path or ""), "CourtBird 기록 (*.cbrec)")
        if path:
            self.path = Path(path if path.endswith(".cbrec") else path + ".cbrec")
            self.save()

    def _confirm_discard(self) -> bool:
        if self.session is None or not self.session.dirty:
            return True
        r = QMessageBox.question(self, "저장하지 않은 변경", "변경 내용을 저장할까요?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            self.save()
            return not self.session.dirty
        return r == QMessageBox.Discard

    def closeEvent(self, e):
        if self._confirm_discard():
            e.accept()
        else:
            e.ignore()

    # --- playback ---------------------------------------------------------------------------
    def _fps(self) -> int:
        return max(1, int(round(self.session.record.fps))) if self.session else 30

    def toggle_play(self):
        if self.session is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶")
        else:
            self._acc = 0.0
            self.timer.start()
            self.play_btn.setText("⏸")

    def _tick(self):
        self._acc += self.session.record.fps * SPEEDS[self.speed.currentIndex()] * 0.030
        n = int(self._acc)
        self._acc -= n
        if n and not self.step(n):
            self.toggle_play()

    def step(self, n: int) -> bool:
        """Move n frames; returns False when the end was hit."""
        if self.session is None or not len(self.session):
            return False
        new = max(0, min(len(self.session) - 1, self.pos + int(n)))
        self.show_frame(new)
        return new < len(self.session) - 1 or n < 0

    def _slider(self, v):
        if self.session is not None and v != self.pos:
            self.show_frame(v)

    def show_frame(self, pos: int):
        self.pos = pos
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.top.pos = pos
        sel = self.top.selected
        if self.session is not None and sel is not None and sel[0] == "person" \
                and self.session.person(pos, sel[1]) is None:
            self.top.selected = None
        self.top.refresh()
        self._draw_video()
        self._status()

    def _draw_video(self):
        if self.video is None or self.session is None or not len(self.session):
            return
        f = self.session.frame(self.pos)
        img = self.video.frame_at(f.frame_idx)
        if img is None:
            self.video_view.set_image(None)
            return
        if self.calib is None:
            cv2.putText(img, "link calibration (File menu) to see team zones", (20, 40), 0, 0.9, (0, 255, 255), 2)
        if self.calib is not None:
            # the calibration is for the reference view; carry it through this frame's camera motion
            calib = self.calib.moved(np.array(f.camera, float).reshape(3, 3)) if f.camera else self.calib
            if self.zones.isChecked():
                self._draw_team_zones(img, f, calib)
            draw_court(img, calib, self.spec, show_points=False)
            for p in f.people:
                u, v = calib.court_to_image(np.array([[p.x, p.y]]))[0]
                c = TEAM_COLORS.get(p.team, UNKNOWN_TEAM)
                cv2.circle(img, (int(u), int(v)), 9, c, -1, cv2.LINE_AA)
                cv2.circle(img, (int(u), int(v)), 9, (255, 255, 255), 1, cv2.LINE_AA)
                if self.ids.isChecked():
                    cv2.putText(img, str(p.id), (int(u) + 10, int(v) - 8), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if f.ball is not None:
                u, v = calib.court_to_image(np.array([[f.ball.x, f.ball.y]]))[0]
                cv2.circle(img, (int(u), int(v)), 7, (255, 255, 255), -1, cv2.LINE_AA)
        self.video_view.set_image(img)

    def _draw_team_zones(self, img, frame_state, calib):
        """Tint each half of the court in its team colour and label it, so the video can be
        compared with the top view (which side is blue?)."""
        sp = self.spec
        overlay = img.copy()
        labels = []
        for team, (x0, x1) in zip(half_teams(sp, frame_state), ((0.0, sp.net_x), (sp.net_x, sp.length))):
            if team is None:
                continue
            quad = calib.court_to_image(np.array([(x0, 0), (x1, 0), (x1, sp.width), (x0, sp.width)]))
            cv2.fillPoly(overlay, [quad.astype(np.int32)], TEAM_COLORS[team])
            cx, cy = calib.court_to_image(np.array([[(x0 + x1) / 2, sp.width / 2]]))[0]
            labels.append((("BLUE", "RED")[team], int(cx), int(cy), TEAM_COLORS[team]))
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
        for text, cx, cy, col in labels:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
            cv2.putText(img, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 7, cv2.LINE_AA)
            cv2.putText(img, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4, cv2.LINE_AA)

    def swap_teams(self):
        if not self.session:
            return
        self.session.swap_teams()
        self.top.refresh()
        self._edited()

    def _zones_toggled(self, on: bool):
        self.top.renderer.team_zones = on
        self.top.refresh()
        self._draw_video()

    def _ids_toggled(self, on: bool):
        self.top.renderer.show_ids = on
        self.top.refresh()
        self._draw_video()

    # --- editing ------------------------------------------------------------------------------
    def _edited(self):
        self._update_title()
        self._status()
        self._draw_video()

    def delete_selected(self):
        sel = self.top.selected
        if not self.session or not sel:
            return
        if sel[0] == "ball":
            self.session.delete_ball(self.pos)
        else:
            self.session.delete_person(self.pos, sel[1])
        self.top.selected = None
        self.top.refresh()
        self._edited()

    def toggle_team(self, propagate: bool):
        sel = self.top.selected
        if not self.session or not sel or sel[0] != "person":
            return
        p = self.session.person(self.pos, sel[1])
        self.session.set_team(self.pos, sel[1], 1 if p.team == 0 else 0, propagate)
        self.top.refresh()
        self._edited()

    def rename_selected(self):
        sel = self.top.selected
        if not self.session or not sel or sel[0] != "person":
            return
        new, ok = QInputDialog.getInt(self, "ID 변경", f"ID {sel[1]} → 새 ID:", sel[1], 1, 9999)
        if not ok:
            return
        r = QMessageBox.question(self, "ID 변경", "이 프레임 이후의 모든 프레임에도 적용할까요?",
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return
        if not self.session.rename(self.pos, sel[1], new, propagate=(r == QMessageBox.Yes)):
            QMessageBox.warning(self, "ID 변경 실패", f"ID {new}가 이미 사용 중입니다.")
            return
        self.top.selected = ("person", new)
        self.top.refresh()
        self._edited()

    def undo(self):
        if not self.session:
            return
        pos = self.session.undo()
        if pos is not None:
            self.top.selected = None
            self.show_frame(pos)
            self._update_title()

    # --- chrome -------------------------------------------------------------------------------
    def _update_title(self):
        name = self.path.name if self.path else ("(새 분석)" if self.session else "")
        mark = "*" if self.session and self.session.dirty else ""
        self.setWindowTitle(f"CourtBird — {name}{mark}" if name else "CourtBird")

    def _status(self):
        if not self.session or not len(self.session):
            self.statusBar().showMessage("기록을 열거나 영상을 분석하세요 (파일 메뉴)")
            self.time_label.setText("0:00.0")
            return
        f = self.session.frame(self.pos)
        t = f.frame_idx / self._source_fps()
        self.time_label.setText(f"{int(t // 60)}:{t % 60:04.1f}")
        sel = self.top.selected
        s = ""
        if sel and sel[0] == "person":
            p = self.session.person(self.pos, sel[1])
            if p:
                s = f"  |  선택: ID {p.id}  팀 {p.team}  ({p.x:.1f}, {p.y:.1f}) m"
        elif sel:
            s = "  |  선택: 공"
        self.statusBar().showMessage(f"프레임 {f.frame_idx}  ({self.pos + 1}/{len(self.session)})  선수 {len(f.people)}명{s}")

    def _source_fps(self) -> float:
        """fps of the original video (the record's fps is divided by the analysis stride)."""
        if self.video is not None:
            return self.video.fps
        fr = self.session.record.frames
        stride = (fr[1].frame_idx - fr[0].frame_idx) if len(fr) > 1 else 1
        return self.session.record.fps * max(1, stride)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    w = MainWindow()
    w.show()
    if len(argv) > 1 and Path(argv[1]).is_file():
        w.load_record_file(Path(argv[1]))
    return app.exec()
