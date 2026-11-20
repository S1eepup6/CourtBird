"""Editable session over a Record: Qt-free so it can be unit-tested.

Every mutating call pushes an undo snapshot of the frames it touches, except
`move_person`/`move_ball` with snapshot=False (used during a drag, after one checkpoint).
"""
from __future__ import annotations

import copy
from typing import Optional

from ..record import Record
from ..schema import Ball, FrameState, Person
from ..sports.spec import SportSpec


class EditSession:
    def __init__(self, record: Record, spec: SportSpec):
        self.record, self.spec = record, spec
        self.dirty = False
        self._undo: list[list[tuple[int, FrameState]]] = []

    def __len__(self) -> int:
        return len(self.record.frames)

    def frame(self, pos: int) -> FrameState:
        return self.record.frames[pos]

    def person(self, pos: int, pid: int) -> Optional[Person]:
        return next((p for p in self.frame(pos).people if p.id == pid), None)

    # --- undo ----------------------------------------------------------------
    def checkpoint(self, positions: list[int]) -> None:
        self._undo.append([(i, copy.deepcopy(self.record.frames[i])) for i in positions])
        if len(self._undo) > 200:
            self._undo.pop(0)

    def undo(self) -> Optional[int]:
        """Revert the last edit; returns the first frame position it touched."""
        if not self._undo:
            return None
        snap = self._undo.pop()
        for i, f in snap:
            self.record.frames[i] = f
        self.dirty = True
        return snap[0][0]

    # --- edits ---------------------------------------------------------------
    def move_person(self, pos: int, pid: int, x: float, y: float, snapshot: bool = True) -> None:
        p = self.person(pos, pid)
        if p is None:
            return
        if snapshot:
            self.checkpoint([pos])
        p.x, p.y = x, y
        self.dirty = True

    def add_person(self, pos: int, x: float, y: float) -> Person:
        self.checkpoint([pos])
        pid = 1 + max((q.id for f in self.record.frames for q in f.people), default=0)
        p = Person(pid, x, y, self.spec.team_of(x))
        self.frame(pos).people.append(p)
        self.dirty = True
        return p

    def delete_person(self, pos: int, pid: int) -> None:
        self.checkpoint([pos])
        f = self.frame(pos)
        f.people = [p for p in f.people if p.id != pid]
        self.dirty = True

    def set_team(self, pos: int, pid: int, team: Optional[int], propagate: bool = False) -> None:
        rng = range(pos, len(self)) if propagate else [pos]
        self.checkpoint(list(rng))
        for i in rng:
            for p in self.frame(i).people:
                if p.id == pid:
                    p.team = team
        self.dirty = True

    def rename(self, pos: int, old: int, new: int, propagate: bool = True) -> bool:
        """Change an id from this frame on. Refuses if `new` already exists in an affected frame."""
        rng = range(pos, len(self)) if propagate else [pos]
        if old == new or any(p.id == new for i in rng for p in self.frame(i).people):
            return False
        self.checkpoint(list(rng))
        for i in rng:
            for p in self.frame(i).people:
                if p.id == old:
                    p.id = new
        self.dirty = True
        return True

    def set_ball(self, pos: int, x: float, y: float, snapshot: bool = True) -> None:
        if snapshot:
            self.checkpoint([pos])
        f = self.frame(pos)
        if f.ball is None:
            f.ball = Ball(x, y, None, 1.0)
        else:
            f.ball.x, f.ball.y, f.ball.confidence = x, y, 1.0  # hand-placed => fully confident
        self.dirty = True

    def delete_ball(self, pos: int) -> None:
        self.checkpoint([pos])
        self.frame(pos).ball = None
        self.dirty = True
