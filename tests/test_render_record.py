import numpy as np

from courtbird import record
from courtbird.render import TopViewRenderer
from courtbird.schema import Ball, FrameState, Person
from courtbird.sports import VOLLEYBALL


def _state():
    people = [Person(i, 2 + i, 3.0, VOLLEYBALL.team_of(2 + i)) for i in range(1, 6)]
    people += [Person(10 + i, 10 + i, 5.0, VOLLEYBALL.team_of(10 + i)) for i in range(1, 6)]
    people.append(Person(99, 5.0, -1.5, 0))  # server behind the line
    return FrameState(frame_idx=7, people=people, ball=Ball(9.5, 4.0, confidence=0.5),
                      hidden_regions=[[(0, 0), (3, 0), (3, 9), (0, 9)]])


def test_render_shape_and_content():
    r = TopViewRenderer(VOLLEYBALL)
    img = r.render(_state())
    assert img.shape == (r.h, r.w, 3)
    assert not np.array_equal(img, r._base)  # something was drawn
    assert np.array_equal(r.render(FrameState()), r._base)  # empty state == bare court


def test_out_of_court_person_is_drawn():
    r = TopViewRenderer(VOLLEYBALL)
    img = r.render(FrameState(people=[Person(1, 5.0, -1.5, 0)]))
    x, y = r.to_px(5.0, -1.5)
    assert 0 <= y < r.h
    assert not np.array_equal(img[y, x], r._base[y, x])


def test_record_roundtrip(tmp_path):
    rec = record.Record("volleyball", 30.0, "x.mp4", [_state(), FrameState(frame_idx=8)])
    p = tmp_path / "r.cbrec"
    record.save(rec, p)
    back = record.load(p)
    assert back.frames[0] == rec.frames[0]
    assert back.frames[1].ball is None
