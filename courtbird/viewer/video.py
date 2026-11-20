from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class VideoSource:
    """Random-access frame reader that avoids seeking during sequential playback."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next = 0

    def frame_at(self, idx: int) -> Optional[np.ndarray]:
        if idx != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        self._next = idx + 1 if ok else -1
        return f if ok else None

    def close(self) -> None:
        self.cap.release()
