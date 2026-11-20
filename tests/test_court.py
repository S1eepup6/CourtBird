import numpy as np
import pytest

from courtbird.court import Calibration, landmarks
from courtbird.court.players import boxes_to_people, foot_points
from courtbird.sports import VOLLEYBALL as V

# A known court->image homography (camera looking at the court from one end).
H_COURT_TO_IMG = np.array([[40.0, 12.0, 300.0], [-6.0, 14.0, 250.0], [0.0, 0.0012, 1.0]])


def _img(pts):
    import cv2
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H_COURT_TO_IMG).reshape(-1, 2)


def _calib(names=("corner_near_left", "corner_near_right", "corner_far_left", "corner_far_right")):
    lm = landmarks(V)
    px = _img([lm[n] for n in names])
    return Calibration.from_points(V, [(n, float(x), float(y)) for n, (x, y) in zip(names, px)], (1280, 720))


def test_calibration_recovers_court_points():
    c = _calib()
    pts = np.array([[3.0, 2.0], [9.0, 4.5], [15.0, 8.0]])
    np.testing.assert_allclose(c.image_to_court(_img(pts)), pts, atol=1e-6)
    np.testing.assert_allclose(c.court_to_image(pts), _img(pts), atol=1e-4)
    assert c.reprojection_error(V) < 1e-6


def test_needs_four_points():
    with pytest.raises(ValueError):
        Calibration.from_points(V, [("corner_near_left", 0, 0)] * 3, (10, 10))


def test_save_load_roundtrip(tmp_path):
    c = _calib(); p = tmp_path / "c.json"; c.save(p)
    np.testing.assert_allclose(Calibration.load(p, V).H, c.H, atol=1e-9)


def test_foot_points_drop_cut_off_boxes():
    boxes = np.array([[0, 0, 10, 100], [0, 0, 10, 720]], np.float32)
    feet, ok = foot_points(boxes, 720)
    assert ok.tolist() == [True, False] and feet[0].tolist() == [5, 100]


def test_filter_and_team():
    c = _calib()
    def box_at(x, y):  # box whose bottom-centre projects to court (x, y)
        fx, fy = _img([(x, y)])[0]
        return [fx - 10, fy - 80, fx + 10, fy]
    boxes = np.array([box_at(4, 3), box_at(14, 5), box_at(9, 4.5 + 20), box_at(-5, 4.5)], np.float32)
    people = boxes_to_people(boxes, c, V)
    assert [p.team for p in people] == [0, 1, 0]  # the y=24.5 person (spectator) is gone, server kept


def test_visibility_marks_outside_image_hidden():
    c = _calib()
    vis, hid = c.visibility(V)
    assert vis and isinstance(vis[0][0], tuple)
