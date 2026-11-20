"""Track records: the saved output of video analysis, loaded later by the viewer.

Format: gzip-compressed JSON, so a record can be reloaded without re-running
(slow) video analysis, and hand-corrected frames can be written back.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import Ball, FrameState, Person

FORMAT_VERSION = 1


@dataclass
class Record:
    sport: str
    fps: float
    source: str = ""
    frames: list[FrameState] = field(default_factory=list)
    calibration: dict | None = None  # Calibration.to_dict(), so a record can be re-projected/overlaid


def save(record: Record, path: str | Path) -> None:
    data = {
        "version": FORMAT_VERSION,
        "sport": record.sport,
        "fps": record.fps,
        "source": record.source,
        "calibration": record.calibration,
        "frames": [asdict(f) for f in record.frames],
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))


def load(path: str | Path) -> Record:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported record version: {data.get('version')}")
    frames = []
    for f in data["frames"]:
        frames.append(FrameState(
            frame_idx=f["frame_idx"],
            people=[Person(**p) for p in f["people"]],
            ball=Ball(**f["ball"]) if f["ball"] else None,
            visible_region=[tuple(p) for p in f["visible_region"]] if f["visible_region"] else None,
            hidden_regions=[[tuple(p) for p in poly] for poly in f["hidden_regions"]],
            camera=f.get("camera"),
        ))
    return Record(sport=data["sport"], fps=data["fps"], source=data["source"], frames=frames,
                  calibration=data.get("calibration"))
