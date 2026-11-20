"""Video -> Record: detect people, project to the court, filter, assign teams and ids."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2

from .court import Calibration
from .court.players import boxes_to_people
from .perception.detector import PersonDetector
from .perception.tracker import GreedyTracker
from .record import Record
from .schema import FrameState
from .sports.spec import SportSpec


def analyze(video: str | Path, calib: Calibration, spec: SportSpec, model: str = "rtmdet-m-person",
            stride: int = 1, start_sec: float = 0.0, end_sec: Optional[float] = None,
            progress: Optional[Callable[[int, int], None]] = None) -> Record:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_sec * fps)
    last = min(total, int(end_sec * fps)) if end_sec else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    det = PersonDetector(model)
    tracker = GreedyTracker(max_dist=1.5 * stride)
    visible, hidden = calib.visibility(spec)
    rec = Record(sport=spec.name, fps=fps / stride, source=Path(video).name)
    idx = first
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - first) % stride == 0:
            people = tracker.update(boxes_to_people(det(frame), calib, spec))
            rec.frames.append(FrameState(frame_idx=idx, people=people,
                                         visible_region=visible[0] if visible else None,
                                         hidden_regions=hidden))
            if progress:
                progress(idx - first, last - first)
        idx += 1
    cap.release()
    return rec
