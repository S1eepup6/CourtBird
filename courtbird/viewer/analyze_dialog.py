from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..court import Calibration
from ..pipeline import analyze
from ..record import Record
from ..sports.spec import SportSpec


class _Cancelled(Exception):
    pass


class Worker(QThread):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, video, calib, spec, model, stride, start, end, camera=True):
        super().__init__()
        self.args = (video, calib, spec, model, stride, start, end, camera)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        video, calib, spec, model, stride, start, end, camera = self.args

        def prog(i, n):
            if self._cancel:
                raise _Cancelled
            self.progress.emit(i, n)
        try:
            self.done.emit(analyze(video, calib, spec, model, stride, start, end, prog, camera_motion=camera))
        except _Cancelled:
            self.failed.emit("")
        except Exception as e:  # surface any pipeline error in the dialog
            self.failed.emit(str(e))


def _file_row(edit: QLineEdit, title: str, flt: str, parent: QWidget) -> QWidget:
    w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
    btn = QPushButton("찾아보기…"); lay.addWidget(edit); lay.addWidget(btn)

    def pick():
        path, _ = QFileDialog.getOpenFileName(parent, title, edit.text(), flt)
        if path:
            edit.setText(path)
    btn.clicked.connect(pick)
    return w


class AnalyzeDialog(QDialog):
    """Runs video analysis in a background thread; `self.record` is set on success."""

    def __init__(self, spec: SportSpec, parent=None, video: str = "", calib: str = ""):
        super().__init__(parent)
        self.setWindowTitle("영상 분석")
        self.setMinimumWidth(520)
        self.spec = spec
        self.record: Record | None = None
        self.video_path = ""
        self.worker: Worker | None = None
        self.video = QLineEdit(video); self.calib = QLineEdit(calib)
        self.model = QComboBox(); self.model.addItems(["rtmdet-m-person (정확)", "rtmdet-nano-person (빠름)"])
        self.stride = QSpinBox(); self.stride.setRange(1, 10); self.stride.setValue(1)
        self.start = QDoubleSpinBox(); self.start.setRange(0, 1e5); self.start.setSuffix(" 초")
        self.end = QDoubleSpinBox(); self.end.setRange(0, 1e5); self.end.setSuffix(" 초 (0 = 끝까지)")
        form = QFormLayout()
        form.addRow("영상", _file_row(self.video, "영상 선택", "Video (*.mp4 *.mov *.avi *.mkv);;All (*)", self))
        form.addRow("코트 보정 파일", _file_row(self.calib, "보정 파일 선택", "JSON (*.json)", self))
        form.addRow("모델", self.model)
        form.addRow("프레임 간격", self.stride)
        form.addRow("시작", self.start)
        form.addRow("끝", self.end)
        self.camera = QCheckBox("카메라 움직임 보정 (팬/줌, 고정 카메라면 꺼도 됨)")
        self.camera.setChecked(True)
        form.addRow("", self.camera)
        self.bar = QProgressBar(); self.bar.setVisible(False)
        self.run_btn = QPushButton("분석 시작"); self.cancel_btn = QPushButton("닫기")
        row = QHBoxLayout(); row.addStretch(); row.addWidget(self.run_btn); row.addWidget(self.cancel_btn)
        lay = QVBoxLayout(self); lay.addLayout(form); lay.addWidget(self.bar); lay.addLayout(row)
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._close)

    def _start(self):
        try:
            calib = Calibration.load(self.calib.text(), self.spec)
        except Exception as e:
            QMessageBox.warning(self, "보정 파일 오류", str(e)); return
        if not Path(self.video.text()).is_file():
            QMessageBox.warning(self, "영상 오류", "영상 파일을 찾을 수 없습니다."); return
        model = "rtmdet-m-person" if self.model.currentIndex() == 0 else "rtmdet-nano-person"
        self.video_path = self.video.text()
        end = self.end.value() or None
        self.worker = Worker(self.video_path, calib, self.spec, model, self.stride.value(), self.start.value(), end,
                             self.camera.isChecked())
        self.worker.progress.connect(lambda i, n: (self.bar.setMaximum(max(n, 1)), self.bar.setValue(i)))
        self.worker.done.connect(self._done); self.worker.failed.connect(self._failed)
        self.bar.setVisible(True); self.bar.setValue(0); self.run_btn.setEnabled(False)
        self.cancel_btn.setText("취소")
        self.worker.start()

    def _done(self, rec):
        self.record = rec; self.accept()

    def _failed(self, msg):
        self.run_btn.setEnabled(True); self.bar.setVisible(False); self.cancel_btn.setText("닫기")
        if msg:
            QMessageBox.critical(self, "분석 실패", msg)

    def _close(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel(); self.worker.wait()
        else:
            self.reject()

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel(); self.worker.wait()
        super().reject()
