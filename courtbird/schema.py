"""Shared data contract between perception and rendering.

All coordinates are in court space, in metres. For a court of length L and
width W: x in [0, L] runs along the long axis (the net is at x = L/2) and
y in [0, W] runs across it. Positions outside that rectangle are allowed
(servers and players behind or beside the lines).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Point = tuple[float, float]
Polygon = list[Point]


@dataclass
class Person:
    id: int
    x: float
    y: float
    team: Optional[int] = None  # 0 / 1, or None when unknown


@dataclass
class Ball:
    x: float
    y: float
    z: Optional[float] = None  # height in metres; None until height estimation exists
    confidence: float = 1.0  # 0..1, low values are drawn fainter


@dataclass
class FrameState:
    """Everything the renderer needs for one frame."""

    frame_idx: int = 0
    people: list[Person] = field(default_factory=list)
    ball: Optional[Ball] = None
    # Region of the court the camera currently sees (informational / viewer use).
    visible_region: Optional[Polygon] = None
    # Regions known NOT to be visible; the renderer shades these. Court areas that
    # are neither visible nor confidently hidden are simply left unshaded.
    hidden_regions: list[Polygon] = field(default_factory=list)
