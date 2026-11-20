import os

import pytest

from courtbird import record
from courtbird.schema import Ball, FrameState, Person
from courtbird.sports import VOLLEYBALL as V
from courtbird.viewer.model import EditSession


def _session(n=4):
    frames = [FrameState(frame_idx=i, people=[Person(1, 2.0, 3.0, 0), Person(2, 12.0, 3.0, 1)]) for i in range(n)]
    return EditSession(record.Record("volleyball", 30.0, "x.mp4", frames), V)


def test_move_and_undo():
    s = _session()
    s.move_person(1, 1, 5.0, 6.0)
    assert (s.person(1, 1).x, s.person(1, 1).y) == (5.0, 6.0) and s.dirty
    assert s.undo() == 1
    assert s.person(1, 1).x == 2.0 and s.undo() is None


def test_drag_is_one_undo_step():
    s = _session()
    s.checkpoint([0])
    for x in (3.0, 4.0, 5.0):
        s.move_person(0, 1, x, 3.0, snapshot=False)
    s.undo()
    assert s.person(0, 1).x == 2.0


def test_add_delete_person():
    s = _session()
    p = s.add_person(0, 14.0, 2.0)
    assert p.id == 3 and p.team == 1
    s.delete_person(0, 3)
    assert s.person(0, 3) is None
    s.undo()
    assert s.person(0, 3) is not None


def test_rename_propagates_and_refuses_conflict():
    s = _session()
    assert s.rename(1, 1, 9)
    assert s.person(0, 1) and s.person(0, 9) is None  # earlier frame untouched
    assert all(s.person(i, 9) for i in (1, 2, 3))
    assert not s.rename(1, 9, 2)  # 2 already exists
    s.undo()  # reverts the successful rename: the refused one pushed no snapshot
    assert s.person(1, 1) is not None and s.person(1, 9) is None


def test_team_propagate():
    s = _session()
    s.set_team(1, 1, 1, propagate=True)
    assert [s.person(i, 1).team for i in range(4)] == [0, 1, 1, 1]


def test_ball_edit():
    s = _session()
    s.set_ball(0, 9.0, 4.0)
    assert s.frame(0).ball == Ball(9.0, 4.0, None, 1.0)
    s.delete_ball(0)
    assert s.frame(0).ball is None


def test_viewer_smoke(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from courtbird.viewer.app import MainWindow
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "r.cbrec"
    record.save(_session(10).record, path)
    w = MainWindow()
    w.load_record_file(path)
    w.step(3)
    assert w.pos == 3 and len(w.session) == 10
    w.step(100)
    assert w.pos == 9
    w.session.dirty = False
    w.close()
    app.processEvents()


def test_swap_teams_and_half_teams():
    from courtbird.render.topview import half_teams
    s = _session()
    assert half_teams(V, s.frame(0)) == (0, 1)
    s.swap_teams()
    assert half_teams(V, s.frame(0)) == (1, 0)
    assert s.person(2, 1).team == 1
    s.undo()
    assert s.person(2, 1).team == 0
    assert half_teams(V, FrameState()) == (None, None)


def test_record_keeps_calibration(tmp_path):
    r = _session().record
    r.calibration = {"sport": "volleyball", "image_size": [10, 10], "points": []}
    p = tmp_path / "c.cbrec"
    record.save(r, p)
    assert record.load(p).calibration == r.calibration
