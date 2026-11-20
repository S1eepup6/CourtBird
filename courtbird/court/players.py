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
