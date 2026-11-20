"""Placeholder tracker: greedy nearest-neighbour matching in court space.

Good enough to give stable ids for the first end-to-end runs. To be replaced by a
proper tracker (appearance + motion, occlusion handling) in a later milestone.
"""
from __future__ import annotations

import numpy as np

from ..schema import Person


class GreedyTracker:
    def __init__(self, max_dist: float = 1.5, max_lost: int = 15):
        self.max_dist, self.max_lost = max_dist, max_lost
        self._tracks: dict[int, tuple[float, float, int]] = {}  # id -> (x, y, frames since seen)
        self._next = 1

    def update(self, people: list[Person]) -> list[Person]:
        pairs = []
        for i, p in enumerate(people):
            for tid, (x, y, _) in self._tracks.items():
                d = float(np.hypot(p.x - x, p.y - y))
                if d <= self.max_dist:
                    pairs.append((d, i, tid))
        pairs.sort()
        used_p, used_t = set(), set()
        for d, i, tid in pairs:
            if i in used_p or tid in used_t:
                continue
            used_p.add(i); used_t.add(tid)
            people[i].id = tid
            self._tracks[tid] = (people[i].x, people[i].y, 0)
        for i, p in enumerate(people):
            if i not in used_p:
                p.id = self._next
                self._tracks[self._next] = (p.x, p.y, 0)
                self._next += 1
        for tid in list(self._tracks):
            if tid not in used_t and tid not in {p.id for p in people}:
                x, y, lost = self._tracks[tid]
                if lost + 1 > self.max_lost:
                    del self._tracks[tid]
                else:
                    self._tracks[tid] = (x, y, lost + 1)
        return people
