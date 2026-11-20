import cv2
import numpy as np

from courtbird.court import Calibration, landmarks
from courtbird.perception.postprocess import finalize
from courtbird.perception.tracking import Det, Tracker
from courtbird.sports import VOLLEYBALL as V

H_COURT_TO_IMG = np.array([[40.0, 12.0, 300.0], [-6.0, 14.0, 250.0], [0.0, 0.0012, 1.0]])
DT = 1 / 30
HEIGHT = 120.0


def to_img(x, y):
    return cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), H_COURT_TO_IMG)[0, 0]


def calib():
    names = ("corner_near_left", "corner_near_right", "corner_far_left", "corner_far_right")
    lm = landmarks(V)
    pts = [(n, *to_img(*lm[n])) for n in names]
    return Calibration.from_points(V, pts, (1280, 720))


def det(x, y, score=0.9):
    return Det(to_img(x, y).astype(float), HEIGHT, score)


def run(path_fn, n, tracker=None, dt=DT):
    """path_fn(i) -> list of court (x, y) visible at frame i. Returns the tracker."""
    tr = tracker or Tracker(dt)
    for i in range(n):
        tr.update(i, [det(x, y) for x, y in path_fn(i)])
    return tr


def test_ids_survive_two_players_passing_each_other():
    # two people walk towards each other in the same row and swap places
    def path(i):
        t = i * DT
        return [(4.0 + 1.5 * t, 4.0), (10.0 - 1.5 * t, 4.8)]
    tr = run(path, 120)
    ids = {t.id for t in tr.all_tracks() if len(t.history) > 100}
    assert len(ids) == 2 and len(tr.all_tracks()) == 2


def test_short_occlusion_keeps_id():
    def path(i):
        return [] if 50 <= i < 65 else [(5.0 + 0.02 * i, 4.0)]
    tr = run(path, 120)
    assert len(tr.all_tracks()) == 1


def test_weak_detection_continues_a_track_but_cannot_start_one():
    tr = Tracker(DT)
    tr.update(0, [det(5, 4, 0.2)])           # weak only: no track
    assert not tr.all_tracks()
    tr.update(1, [det(5, 4, 0.9)])
    tr.update(2, [det(5, 4, 0.2)])           # weak continues it
    assert len(tr.all_tracks()) == 1 and len(tr.all_tracks()[0].history) == 2


def test_long_gap_is_relinked_by_stitching():
    # hidden for 3 s: the tracker gives up (max age 2 s) and starts a new id; stitching joins them
    def path(i):
        return [] if 60 <= i < 150 else [(5.0 + 0.01 * i, 4.0), (14.0, 6.0)]
    n = 240
    tr = run(path, n)
    assert len(tr.all_tracks()) >= 3
    people = finalize(tr.all_tracks(), n, DT, V, calib())
    assert len({p.id for f in people for p in f}) == 2


def test_finalize_drops_static_outside_and_assigns_team_by_track():
    n = 300  # 10 s
    def path(i):
        t = i * DT
        player = (8.0 + 1.0 * np.sin(t), 4.0)   # hovers around the net, mostly on side 0 (x < 9)
        judge = (3.0, -1.5)                       # stands outside the sideline, motionless
        return [player, judge]
    tr = run(path, n)
    people = finalize(tr.all_tracks(), n, DT, V, calib())
    ids = {p.id for f in people for p in f}
    assert len(ids) == 1                           # judge dropped
    teams = {p.team for f in people for p in f}
    assert teams == {0}                            # one team for the whole track, despite crossing x=9
    assert any(p.x > V.net_x for f in people for p in f)


def test_players_cut_off_by_image_bottom_are_not_projected():
    tr = Tracker(DT)
    for i in range(40):
        d = det(5, 4)
        d.feet_ok = False
        tr.update(i, [d])
    assert finalize(tr.all_tracks(), 40, DT, V, calib()) == [[] for _ in range(40)]
