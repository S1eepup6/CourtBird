"""Sport descriptions. Adding a sport = adding a SportSpec."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Point


@dataclass(frozen=True)
class SportSpec:
    name: str
    length: float  # x extent, metres (net sits at length / 2)
    width: float  # y extent, metres
    free_zone: float  # margin drawn around the court, metres
    post_offset: float = 1.0  # net posts stand this far outside each sideline, metres
    # Extra court lines as ((x1, y1), (x2, y2)) segments, e.g. attack lines.
    lines: tuple[tuple[Point, Point], ...] = field(default_factory=tuple)

    @property
    def net_x(self) -> float:
        return self.length / 2

    def team_of(self, x: float) -> int:
        """Team side by position relative to the net: 0 left of it, 1 right."""
        return 0 if x < self.net_x else 1
