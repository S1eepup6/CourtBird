"""Turn detector boxes into court-space people, dropping non-players."""
from __future__ import annotations

import numpy as np

from ..schema import Person
from ..sports.spec import SportSpec
from .calibration import Calibration


def foot_points(boxes: np.ndarray, image_h: int, edge_tol: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-centre of each box and a mask of boxes whose feet are actually in frame."""
    if len(boxes) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
    return feet, boxes[:, 3] < image_h - edge_tol


def boxes_to_people(boxes: np.ndarray, calib: Calibration, spec: SportSpec,
                    side_margin: float = 2.0, end_margin: float = 8.0) -> list[Person]:
    """Project boxes to the court. Boxes whose feet are cut off or land far from the court
    (spectators, staff, cameramen) are dropped; players beside/behind the lines (servers,
    retrieving a ball) are kept. Ids are assigned later by the tracker."""
    feet, ok = foot_points(boxes, calib.image_size[1])
    if not ok.any():
        return []
    court = calib.image_to_court(feet[ok])
    people = []
    for x, y in court:
        if -end_margin <= x <= spec.length + end_margin and -side_margin <= y <= spec.width + side_margin:
            people.append(Person(id=-1, x=float(x), y=float(y), team=spec.team_of(x)))
    return people


def detections(frame_bgr: np.ndarray, boxes: np.ndarray, min_height: float = 20.0, feet_margin: float = 0.03):
    """Detector boxes -> tracker detections (image space: foot point, box height, score, appearance).

    Nothing is filtered by court position here: spectators and staff are removed per *track*
    afterwards (perception.postprocess.finalize), because a single frame's foot point is noisy.
    A box ending within `feet_margin` (fraction of the image height) of the bottom edge is
    probably cut off: its bottom is then not where the feet are, so its foot point is not used.
    """
    from ..perception.tracking import Det, appearance

    H = frame_bgr.shape[0]
    out = []
    for b in boxes:
        h = float(b[3] - b[1])
        if h < min_height:
            continue
        out.append(Det(np.array([(b[0] + b[2]) / 2, b[3]], float), h, float(b[4]) if len(b) > 4 else 1.0,
                       bool(b[3] < H * (1.0 - feet_margin)), appearance(frame_bgr, b)))
    return out
