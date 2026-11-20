"""Video -> Record: detect people, track them, project to the court, clean up, assign teams.

Three stages run in their own threads so CPU and GPU work overlap:
  reader   : decode + preprocess            (CPU)
  main     : detector inference             (GPU, releases the GIL)
  tracker  : appearance + online tracking   (CPU)
Tracklets are then stitched/filtered/projected offline (perception.postprocess.finalize).
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2

from .court import Calibration
from .court.players import detections
from .perception.detector import PersonDetector
from .perception.postprocess import finalize
from .perception.tracking import Tracker
from .record import Record
from .schema import FrameState
from .sports.spec import SportSpec


def _put(q: queue.Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _get(q: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            pass
    return None


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
    dt = stride / fps
    tracker = Tracker(dt)
    frame_idxs: list[int] = []
    q_in: queue.Queue = queue.Queue(maxsize=8)
    q_det: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    errors: list[BaseException] = []

    def reader():
        try:
            idx = first
            while idx < last and not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if (idx - first) % stride == 0:
                    tensor, ratio = det.preprocess(frame)
                    if not _put(q_in, (idx, frame, tensor, ratio), stop):
                        return
                idx += 1
        except BaseException as e:  # surfaced in the caller's thread below
            errors.append(e)
            stop.set()
        finally:
            _put(q_in, None, stop)

    def track():
        try:
            while True:
                item = _get(q_det, stop)
                if item is None:
                    break
                idx, frame, boxes = item
                tracker.update(len(frame_idxs), detections(frame, boxes))
                frame_idxs.append(idx)
                if progress:
                    progress(idx - first, last - first)
        except BaseException as e:  # includes a cancel raised by the progress callback
            errors.append(e)
            stop.set()

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=track, daemon=True)]
    for t in threads:
        t.start()
    try:
        while True:
            item = _get(q_in, stop)
            if item is None:
                break
            idx, frame, tensor, ratio = item
            _put(q_det, (idx, frame, det.postprocess(det.run(tensor), ratio)), stop)
    except BaseException as e:
        errors.append(e)
        stop.set()
    finally:
        _put(q_det, None, stop)
    for t in threads:
        t.join()
    cap.release()
    if errors:
        raise errors[0]

    visible, hidden = calib.visibility(spec)
    people = finalize(tracker.all_tracks(), len(frame_idxs), dt, spec, calib)
    rec = Record(sport=spec.name, fps=fps / stride, source=Path(video).name, calibration=calib.to_dict())
    for pos, fi in enumerate(frame_idxs):
        rec.frames.append(FrameState(frame_idx=fi, people=people[pos],
                                     visible_region=visible[0] if visible else None, hidden_regions=hidden))
    return rec
