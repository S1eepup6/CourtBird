"""Video -> Record: detect people, track them, project to the court, clean up, assign teams.

Three stages run in their own threads so CPU and GPU work overlap:
  reader   : decode + preprocess                              (CPU)
  main     : detector inference                               (GPU, releases the GIL)
  tracker  : camera motion + appearance + online tracking     (CPU)
Tracklets are then stitched/filtered/projected offline (perception.postprocess.finalize).

Camera motion: the calibration refers to one view (the first analysed frame). For every frame the
camera motion M (view -> this frame) is estimated from the static background, detections are mapped
back to the view with M^-1 before tracking, and M is stored per frame so the viewer can draw the
court correctly. A calibration clicked at another time is first carried to the first frame.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .court import Calibration
from .court.camera_motion import (MotionTracker, apply, box_mask, register, snap_static,
                                  stabilize_detections)
from .court.players import detections
from .court import viewpoint as viewpoint_mod
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


def _calibration_at_start(cap, det, calib: Calibration, fps: float, first: int) -> Calibration:
    """Express `calib` in the view of the first analysed frame. The calibration remembers the video
    time it was clicked at; if that differs from the start, the two frames are registered directly
    (SIFT) and the calibration points are carried across."""
    if calib.sec is None or abs(calib.sec * fps - first) < 0.1 * fps:
        return calib

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            raise ValueError(f"cannot read video frame {int(i)} to relate the calibration to the start")
        b = det(f)
        return f, box_mask(f.shape[:2], b[b[:, 4] > 0.3])

    fa, ma = grab(calib.sec * fps)
    fb, mb = grab(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    H = register(fa, fb, mask_a=ma, mask_b=mb)
    if H is None:
        raise ValueError(f"could not relate the calibration frame ({calib.sec:.1f}s) to the analysis start "
                         f"({first / fps:.1f}s); re-run the calibration tool at the start time")
    moved = calib.moved(H)
    moved.sec = first / fps
    return moved


def analyze(video: str | Path, calib: Calibration, spec: SportSpec, model: str = "rtmdet-m-person",
            stride: int = 1, start_sec: float = 0.0, end_sec: Optional[float] = None,
            progress: Optional[Callable[[int, int], None]] = None, camera_motion: bool = True) -> Record:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_sec * fps)
    last = min(total, int(end_sec * fps)) if end_sec else total

    det = PersonDetector(model)
    if camera_motion:
        calib = _calibration_at_start(cap, det, calib, fps, first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    dt = stride / fps
    tracker = Tracker(dt)
    motion = MotionTracker() if camera_motion else None
    frame_idxs: list[int] = []
    motions: list[np.ndarray] = []
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
                dets = detections(frame, boxes)
                M = np.eye(3)
                if motion is not None:
                    M = snap_static(motion.update(frame, boxes[boxes[:, 4] > 0.3]),
                                    (frame.shape[1], frame.shape[0]))
                    stabilize_detections(dets, np.linalg.inv(M))
                tracker.update(len(frame_idxs), dets)
                frame_idxs.append(idx)
                motions.append(M)
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

    people = finalize(tracker.all_tracks(), len(frame_idxs), dt, spec, calib)
    calib.sec = first / fps  # the reference view is the first analysed frame
    rec = Record(sport=spec.name, fps=fps / stride, source=Path(video).name, calibration=calib.to_dict())
    rec.viewpoint = _viewpoints(tracker.all_tracks(), calib, spec)
    w, h = calib.image_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    vis_cache: tuple[np.ndarray, list, list] | None = None
    for pos, fi in enumerate(frame_idxs):
        M = motions[pos]
        moved = not np.allclose(M, np.eye(3))
        # visible/hidden court regions depend on the camera view: recompute when it moved > 4 px
        if vis_cache is None or np.abs(apply(M, corners) - apply(vis_cache[0], corners)).max() > 4.0:
            v, hd = (calib.moved(M) if moved else calib).visibility(spec)
            vis_cache = (M, v, hd)
        rec.frames.append(FrameState(frame_idx=fi, people=people[pos],
                                     visible_region=vis_cache[1][0] if vis_cache[1] else None,
                                     hidden_regions=vis_cache[2],
                                     camera=[float(x) for x in M.ravel()] if moved else None))
    return rec


def _viewpoints(tracks, calib: Calibration, spec: SportSpec) -> dict:
    """Both camera-position estimates: from the calibration, and (independently) from where the
    tracked people stand. Either may be missing if it cannot be computed."""
    out = {}
    try:
        out["calibration"] = viewpoint_mod.from_calibration(calib, spec).to_dict()
    except Exception:
        pass
    try:
        vp = viewpoint_mod.from_tracks(tracks, calib.image_size)
        if vp is not None:
            out["layout"] = vp.to_dict()
    except Exception:
        pass
    return out
