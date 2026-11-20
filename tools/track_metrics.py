"""Tracking-quality proxies for a record (no ground truth needed).

    python -m tools.track_metrics rec.cbrec

A well-tracked volleyball clip has roughly 12-14 people on court and few distinct ids, long
tracks, and few ids born per second. ID churn (the old greedy tracker) shows up as many ids,
short tracks and a high birth rate.
"""
import sys
from collections import defaultdict

import numpy as np

from courtbird import record


def metrics(rec: record.Record) -> dict:
    frames = rec.frames
    seen: dict[int, list[int]] = defaultdict(list)
    for pos, f in enumerate(frames):
        for p in f.people:
            seen[p.id].append(pos)
    dur = len(frames) / rec.fps if frames else 0.0
    lens = np.array([(v[-1] - v[0] + 1) / rec.fps for v in seen.values()]) if seen else np.zeros(1)
    gaps = sum(int(np.sum(np.diff(v) > 1)) for v in seen.values())
    counts = [len(f.people) for f in frames]
    # jitter: mean frame-to-frame step of tracked people (m); noisy tracks/positions inflate it
    steps = []
    for pos in range(1, len(frames)):
        prev = {p.id: p for p in frames[pos - 1].people}
        for p in frames[pos].people:
            q = prev.get(p.id)
            if q:
                steps.append(np.hypot(p.x - q.x, p.y - q.y))
    return {
        "seconds": round(dur, 1),
        "unique_ids": len(seen),
        "ids_born_per_sec": round(len(seen) / dur, 2) if dur else 0,
        "people_per_frame": round(float(np.mean(counts)), 1) if counts else 0,
        "median_track_s": round(float(np.median(lens)), 2),
        "tracks_under_1s": int(np.sum(lens < 1.0)),
        "gaps_within_tracks": gaps,
        "mean_step_m": round(float(np.mean(steps)), 3) if steps else 0,
        "p95_step_m": round(float(np.percentile(steps, 95)), 3) if steps else 0,
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        for k, v in metrics(record.load(path)).items():
            print(f"  {k:20s} {v}")
