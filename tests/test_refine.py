import cv2
import numpy as np

from courtbird.court import Calibration, landmarks
from courtbird.court.refine import (CORNERS, Refiner, calibration_from_corners, corners_of, line_evidence,
                                    refine_calibration, sample_points)
from courtbird.sports import VOLLEYBALL as V

H_COURT_TO_IMG = np.array([[40.0, 12.0, 300.0], [-6.0, 14.0, 250.0], [0.0, 0.0012, 1.0]])
SIZE = (1280, 720)


def to_img(pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H_COURT_TO_IMG).reshape(-1, 2)


def true_corners():
    lm = landmarks(V)
    return to_img([lm[c] for c in CORNERS])


def synthetic_frame(decoys=True):
    img = np.full((SIZE[1], SIZE[0], 3), (120, 110, 40), np.uint8)  # teal-ish floor (BGR)
    L, W = V.length, V.width
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0))] + list(V.lines):
        p = to_img([a, b]).astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (240, 240, 240), 3, cv2.LINE_AA)
    if decoys:  # another sport's markings 3 m outside the court
        m = 3.0
        quad = to_img([(-m, -m), (L + m, -m), (L + m, W + m), (-m, W + m)]).astype(np.int32)
        cv2.polylines(img, [quad], True, (240, 240, 240), 3, cv2.LINE_AA)
    return img


def seed_from(corners):
    return calibration_from_corners(V, corners, SIZE)


def corner_error(calib, ref):
    return float(np.linalg.norm(corners_of(calib, V) - ref, axis=1).mean())


def test_evidence_finds_lines_not_flat_floor():
    ev = line_evidence(synthetic_frame(decoys=False))
    assert ev.max() == 255
    on_line = corners_of(seed_from(true_corners()), V)[0].astype(int)
    assert ev[on_line[1] - 2:on_line[1] + 3, on_line[0] - 2:on_line[0] + 3].max() == 255
    assert (ev > 0).mean() < 0.05  # sparse: lines only


def test_cost_is_lowest_at_the_true_calibration():
    frame = synthetic_frame()
    r = Refiner(V, line_evidence(frame))
    truth = true_corners()
    off = truth + np.array([[10, -8], [-9, 12], [11, 9], [-10, -10]], float)
    assert r.cost(truth, 8.0)[0] < r.cost(off, 8.0)[0]


def test_refinement_pulls_a_perturbed_seed_back_without_latching_onto_decoys():
    frame = synthetic_frame(decoys=True)
    truth = true_corners()
    rng = np.random.default_rng(3)
    seed_corners = truth + rng.normal(0, 8, truth.shape)
    seed = seed_from(seed_corners)
    before = corner_error(seed, truth)
    fit = refine_calibration(seed, frame, V)
    after = corner_error(fit.calib, truth)
    assert after < before and after < 5.0
    assert fit.cost < fit.seed_cost


def test_max_shift_bounds_how_far_corners_move():
    frame = synthetic_frame(decoys=True)
    seed = seed_from(true_corners() + 25.0)  # whole court slid 25 px
    fit = refine_calibration(seed, frame, V, max_shift=10.0)
    moved = np.abs(corners_of(fit.calib, V) - corners_of(seed, V)).max()
    assert moved <= 10.5


def test_sample_points_cover_all_model_lines():
    pts = sample_points(V, 0.5)
    assert pts[:, 0].min() == 0.0 and pts[:, 0].max() == V.length
    assert any(abs(x - 6.0) < 1e-6 for x in pts[:, 0])  # attack line present
