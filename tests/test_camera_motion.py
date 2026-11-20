import cv2
import numpy as np

from courtbird import record
from courtbird.court import Calibration, landmarks
from courtbird.court.camera_motion import (MotionTracker, apply, box_mask, local_scale, register, snap_static,
                                           stabilize_detections)
from courtbird.perception.tracking import Det
from courtbird.schema import FrameState
from courtbird.sports import VOLLEYBALL as V

W, H = 640, 360


def texture(seed=0):
    """Busy static scene: random blobs and lines on a blurred noisy background."""
    rng = np.random.default_rng(seed)
    img = (rng.random((H, W, 3)) * 80 + 60).astype(np.uint8)
    for _ in range(250):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(3, 14)), c, -1)
        else:
            q = (p[0] + int(rng.integers(-40, 40)), p[1] + int(rng.integers(-40, 40)))
            cv2.line(img, p, q, c, int(rng.integers(1, 4)))
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def motion(i, n, pan=30.0, z0=1.15, z1=1.35):
    a = 2 * np.pi * i / n
    z = z0 + (z1 - z0) * 0.5 * (1 - np.cos(a))
    c = np.array([W / 2, H / 2])
    return (np.array([[1, 0, c[0] + pan * np.sin(a)], [0, 1, c[1] + 0.3 * pan * np.sin(2 * a)], [0, 0, 1]])
            @ np.diag([z, z, 1.0]) @ np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]]))


def view(img, Hm):
    return cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def test_apply_and_local_scale():
    M = np.diag([2.0, 2.0, 1.0])
    np.testing.assert_allclose(apply(M, [[1, 2], [3, 4]]), [[2, 4], [6, 8]])
    assert abs(local_scale(M, np.array([10.0, 10.0])) - 2.0) < 1e-9
    assert abs(local_scale(np.eye(3), np.array([5.0, 5.0])) - 1.0) < 1e-9


def test_motion_tracker_follows_pan_and_zoom():
    img, n = texture(), 80
    mt = MotionTracker(max_corners=400, grid=(3, 4), rekey_inliers=60)
    H0 = motion(0, n)
    probes = np.array([[100.0, 100.0], [500.0, 120.0], [320.0, 300.0], [80.0, 320.0]])
    worst = 0.0
    for i in range(n):
        M = mt.update(view(img, motion(i, n)))
        want = apply(motion(i, n) @ np.linalg.inv(H0), probes)  # reference view -> frame i
        worst = max(worst, float(np.linalg.norm(apply(M, probes) - want, axis=1).max()))
    assert worst < 3.0, worst


def test_static_camera_snaps_to_identity():
    M = np.array([[1, 0, 0.3], [0, 1, -0.2], [0, 0, 1]])
    assert np.array_equal(snap_static(M, (W, H)), np.eye(3))
    far = np.array([[1, 0, 25.0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(snap_static(far, (W, H)), far)


def test_register_recovers_known_homography():
    img = texture(1)
    Ha, Hb = motion(10, 80), motion(55, 80)
    got = register(view(img, Ha), view(img, Hb), scale=1.0)
    assert got is not None
    probes = np.array([[100.0, 100.0], [500.0, 120.0], [320.0, 300.0]])
    want = apply(Hb @ np.linalg.inv(Ha), probes)
    assert np.linalg.norm(apply(got, probes) - want, axis=1).max() < 2.0


def test_stabilize_detections_inverts_the_camera_motion():
    M = motion(30, 80)
    foot_ref, h_ref = np.array([300.0, 200.0]), 100.0
    d = Det(apply(M, foot_ref)[0], h_ref * local_scale(M, foot_ref), 0.9)
    stabilize_detections([d], np.linalg.inv(M))
    np.testing.assert_allclose(d.foot, foot_ref, atol=1e-6)
    assert abs(d.height - h_ref) < 0.5


def test_calibration_moved_projects_court_through_camera_motion():
    lm = landmarks(V)
    names = ("corner_near_left", "corner_near_right", "corner_far_left", "corner_far_right")
    px = {"corner_near_left": (100, 300), "corner_near_right": (500, 320),
          "corner_far_left": (200, 120), "corner_far_right": (450, 125)}
    c = Calibration.from_points(V, [(n, *px[n]) for n in names], (W, H))
    M = motion(40, 80)
    m = c.moved(M)
    court = np.array([[3.0, 2.0], [12.0, 7.0], [9.0, 4.5]])
    np.testing.assert_allclose(m.court_to_image(court), apply(M, c.court_to_image(court)), atol=1e-3)
    np.testing.assert_allclose(m.image_to_court(apply(M, c.court_to_image(court))), court, atol=1e-3)
    assert lm["corner_near_left"] == (0.0, 0.0)


def test_calibration_time_roundtrips(tmp_path):
    c = Calibration.from_points(V, [("corner_near_left", 100, 300), ("corner_near_right", 500, 320),
                                    ("corner_far_left", 200, 120), ("corner_far_right", 450, 125)], (W, H))
    c.sec = 12.5
    p = tmp_path / "c.json"
    c.save(p)
    assert Calibration.load(p, V).sec == 12.5


def test_record_keeps_camera_motion(tmp_path):
    r = record.Record("volleyball", 30.0, "x.mp4", [FrameState(frame_idx=0), FrameState(frame_idx=1, camera=[float(x) for x in np.eye(3).ravel()])])
    p = tmp_path / "r.cbrec"
    record.save(r, p)
    back = record.load(p)
    assert back.frames[0].camera is None and back.frames[1].camera == list(np.eye(3).ravel())


def test_box_mask_blanks_people():
    m = box_mask((100, 100), np.array([[10, 10, 30, 40, 0.9]]), pad=0)
    assert m[20, 20] == 0 and m[60, 60] == 255 and box_mask((10, 10), None) is None
