import os

import numpy as np

from courtbird import record
from courtbird.court import Calibration, landmarks
from courtbird.court.viewpoint import (ViewPoint, classify, from_calibration, from_layout, from_tracks,
                                       ground_coordinates, track_arrays)
from courtbird.perception.tracking import Track
from courtbird.sports import VOLLEYBALL as V

W, H = 1280, 720
CORNERS = ("corner_near_left", "corner_near_right", "corner_far_left", "corner_far_right")


def make_camera(rng, dist=(12, 35), height=(2.5, 12), focal=(0.6, 1.5)):
    phi = rng.uniform(0, 2 * np.pi)
    D = rng.uniform(*dist)
    C = np.array([V.length / 2 + D * np.cos(phi), V.width / 2 + D * np.sin(phi), rng.uniform(*height)])
    T = np.array([V.length / 2 + rng.uniform(-3, 3), V.width / 2 + rng.uniform(-3, 3), 0.0])
    f = rng.uniform(*focal) * W
    fwd = (T - C) / np.linalg.norm(T - C)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), R, -R @ C, C, f


def project(K, R, t, P):
    p = (K @ (R @ np.asarray(P, float).T + t[:, None])).T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def true_beta(R):
    a = R[2]
    return float(np.degrees(np.arccos(abs(a[0]) / np.hypot(a[0], a[1]))))


def calib_of(K, R, t):
    lm = landmarks(V)
    px, z = project(K, R, t, [(*lm[n], 0.0) for n in CORNERS])
    return None if (z <= 0).any() else Calibration.from_points(V, [(n, float(x), float(y)) for n, (x, y) in zip(CORNERS, px)], (W, H))


def test_classify_thresholds():
    assert classify(10) == "end" and classify(45) == "corner" and classify(80) == "side"


def test_pose_from_calibration_is_exact_for_perfect_corners():
    rng = np.random.default_rng(5)
    done = 0
    while done < 25:
        K, R, t, C, f = make_camera(rng)
        c = calib_of(K, R, t)
        if c is None:
            continue
        vp = from_calibration(c, V)
        assert abs(vp.focal_px - f) / f < 1e-3
        assert np.hypot(vp.position[0] - C[0], vp.position[1] - C[1]) < 0.05
        assert abs(vp.height - C[2]) < 0.05
        assert abs(vp.beta - true_beta(R)) < 0.1
        done += 1


def test_pose_from_calibration_survives_corner_noise():
    rng = np.random.default_rng(6)
    errs, ok = [], 0
    for _ in range(80):
        K, R, t, C, f = make_camera(rng)
        c = calib_of(K, R, t)
        if c is None:
            continue
        pts = [(n, x + rng.normal(0, 2), y + rng.normal(0, 2)) for n, x, y in c.points]
        vp = from_calibration(Calibration.from_points(V, pts, (W, H)), V)
        errs.append(abs(vp.beta - true_beta(R)))
        ok += vp.kind == classify(true_beta(R))
    assert np.median(errs) < 3.0 and ok / len(errs) > 0.85


def test_ground_coordinates_recover_distances_between_people():
    rng = np.random.default_rng(7)
    K, R, t, C, f = make_camera(rng, height=(5, 8), dist=(15, 25))
    pts = np.c_[rng.uniform(2, 16, 12), rng.uniform(1, 8, 12)]
    feet, _ = project(K, R, t, np.c_[pts, np.zeros(12)])
    # true horizon row: the vanishing line of the floor, from the camera pitch
    pitch = np.arcsin(-R[2][2]) if False else np.arctan2(-R[2][2], np.hypot(R[2][0], R[2][1]))
    y_h = H / 2 - f * np.tan(pitch)
    g = ground_coordinates(feet, np.full(12, 100.0), (W, H), f, y_h, float(C[2]))
    assert g is not None
    d_true = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    d_est = np.linalg.norm(g[:, None] - g[None], axis=2)
    assert np.abs(d_true - d_est).max() < 0.05 * d_true.max()


def _people_study(rng, n, use_teams=True):
    errs = []
    while len(errs) < n:
        K, R, t, C, f = make_camera(rng)
        pos = np.c_[rng.uniform(0.5, 8.5, 12), rng.uniform(0.5, 8.5, 12)]
        pos[6:, 0] += 9.0
        ph = rng.normal(1.75, 0.08, 12)
        feet_all, h_all, ok = [], [], True
        for _ in range(100):
            pos += rng.normal(0, 0.25, pos.shape)
            pos[:6, 0] = np.clip(pos[:6, 0], -1.0, 8.8)
            pos[6:, 0] = np.clip(pos[6:, 0], 9.2, 19.0)
            pos[:, 1] = np.clip(pos[:, 1], -1.0, 10.0)
            fp, z = project(K, R, t, np.c_[pos, np.zeros(12)])
            hp, _ = project(K, R, t, np.c_[pos, ph])
            if (z <= 0).any():
                ok = False
                break
            feet_all.append(fp + rng.normal(0, 2.0, fp.shape))
            h_all.append(np.abs(fp[:, 1] - hp[:, 1]))
        if not ok:
            continue
        feet, hs = np.concatenate(feet_all), np.concatenate(h_all)
        # spectators on a raised stand, standing still
        sp = np.c_[rng.uniform(-3, 21, 20), rng.uniform(-8, -4, 20), np.full(20, 2.5)]
        fs, zs = project(K, R, t, sp)
        hs_, _ = project(K, R, t, sp + [0, 0, 1.7])
        if (zs <= 0).any():
            continue
        rep = np.repeat(np.arange(20), 100)
        feet = np.concatenate([feet, fs[rep]])
        hs = np.concatenate([hs, np.abs(fs[rep, 1] - hs_[rep, 1])])
        w = np.r_[np.ones(len(feet_all) * 12), np.full(len(rep), 0.05)]
        tm = np.r_[np.tile(np.r_[np.zeros(6), np.ones(6)], 100).astype(int), np.full(len(rep), 2)]
        inside = (feet[:, 0] > 0) & (feet[:, 0] < W) & (feet[:, 1] > 0) & (feet[:, 1] < H) & (hs > 15)
        vp = from_layout(feet[inside], hs[inside], (W, H), 0.9, w[inside], tm[inside] if use_teams else None)
        if vp is not None:
            errs.append(abs(vp.beta - true_beta(R)))
    return np.array(errs)


def test_layout_estimate_is_usable_with_weights_and_teams():
    errs = _people_study(np.random.default_rng(8), 60)
    assert np.median(errs) < 12.0 and np.percentile(errs, 90) < 30.0


def _track(tid, rng, kit, still):
    h = []
    for i in range(60):
        x = 600 + (0 if still else 150 * np.sin(i / 8.0 + tid))
        y = 500 + (0 if still else 40 * np.cos(i / 9.0))
        h.append((i, x, y, 100.0, True))
    feat = np.zeros(192, np.float32)
    feat[kit] = 0.8
    feat[(kit + 5) % 192] = 0.2
    return Track(tid, np.zeros(4), np.eye(4), 100.0, feat, 0, 59, history=h)


def test_track_arrays_weights_and_teams():
    rng = np.random.default_rng(0)
    tracks = [_track(i, rng, kit=10 if i % 2 else 90, still=False) for i in range(8)] + [_track(20, rng, 50, still=True)]
    feet, heights, weights, teams, has_teams = track_arrays(tracks)
    assert has_teams and len(feet) == 9 * 60
    assert weights[-1] < 0.1 and weights[0] > 0.9          # the still one barely counts
    assert set(teams[:-60]) == {0, 1} and teams[-1] == 2   # two kits found, the still one unlabeled


def test_viewpoint_describe_and_record_roundtrip(tmp_path):
    vp = ViewPoint("end", 12.0, "calibration", 0.9, (-8.0, 12.0), 6.0, 900.0)
    assert "끝선" in vp.describe(V)
    r = record.Record("volleyball", 30.0, "x.mp4", [], viewpoint={"calibration": vp.to_dict()})
    p = tmp_path / "r.cbrec"
    record.save(r, p)
    back = record.load(p).viewpoint["calibration"]
    assert back["kind"] == "end" and back["position"] == [-8.0, 12.0]


def test_viewer_shows_camera_marker(tmp_path):
    from PySide6.QtWidgets import QApplication
    from courtbird.schema import FrameState, Person
    from courtbird.viewer.app import MainWindow
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    vp = ViewPoint("end", 20.0, "calibration", 0.9, (-8.0, 12.0), 6.0, 900.0)
    r = record.Record("volleyball", 30.0, "x.mp4", [FrameState(frame_idx=0, people=[Person(1, 3.0, 4.0, 0)])],
                      viewpoint={"calibration": vp.to_dict(), "layout": {**vp.to_dict(), "source": "layout", "position": None}})
    p = tmp_path / "r.cbrec"
    record.save(r, p)
    w = MainWindow()
    w.load_record_file(p)
    assert w.top.cam_pos == (-8.0, 12.0) and w.vp_layout is not None and w.vp_layout.position is None
    w.top.grab()  # paints the camera glyph without raising
    w.session.dirty = False
    w.close()
    app.processEvents()
