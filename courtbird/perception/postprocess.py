"""Offline clean-up of the tracker's tracklets: stitch -> project -> filter non-players -> fill/smooth -> team.

Everything here works on whole tracks, which the online tracker cannot see: a player hidden for a
few seconds comes back as a new tracklet and is re-linked, staff standing still beside the court
are dropped, and a player's team is the side of the net they spend most of their time on, so
players jumping at the net no longer flip colour from frame to frame.

Stitching happens in image space (like the tracker); only `finalize` touches the calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..court.calibration import Calibration
from ..schema import Person
from ..sports.spec import SportSpec
from .tracking import BIG, PERSON_M, Track, app_distance


@dataclass
class Group:
    pos: np.ndarray     # processed-frame positions, ascending
    foot: np.ndarray    # (n, 2) pixel foot points
    scale: np.ndarray   # (n,) box heights, pixels
    ok: np.ndarray      # (n,) foot point is trustworthy (box not cut by the image bottom)
    feat: np.ndarray | None
    n: int


def _group(t: Track) -> Group:
    h = t.history
    return Group(np.array([r[0] for r in h], int), np.array([[r[1], r[2]] for r in h], float),
                 np.array([r[3] for r in h], float), np.array([r[4] for r in h], bool), t.feat, len(h))


def _end_velocity(g: Group, dt: float, k: int = 6) -> np.ndarray:
    if len(g.pos) < 2:
        return np.zeros(2)
    j = max(0, len(g.pos) - k)
    span = (g.pos[-1] - g.pos[j]) * dt
    return (g.foot[-1] - g.foot[j]) / span if span > 0 else np.zeros(2)


def stitch(tracks: list[Track], dt: float, max_gap_s: float = 4.0, base_m: float = 1.0,
           speed_m: float = 3.0, app_gate: float = 0.6, rounds: int = 3) -> list[Group]:
    """Link a tracklet that ended to one that started later, nearby and looking alike."""
    groups = [_group(t) for t in tracks if t.history]
    max_gap = int(round(max_gap_s / dt))
    for _ in range(rounds):
        n = len(groups)
        cost = np.full((n, n), BIG)
        for i, a in enumerate(groups):
            s = float(np.mean(a.scale[-5:]))
            v = np.clip(_end_velocity(a, dt), -5 * s / PERSON_M, 5 * s / PERSON_M)
            for j, b in enumerate(groups):
                gap = b.pos[0] - a.pos[-1]
                if i == j or gap <= 0 or gap > max_gap:
                    continue
                gs = gap * dt
                d = min(float(np.hypot(*(a.foot[-1] + v * gs - b.foot[0]))),
                        float(np.hypot(*(a.foot[-1] - b.foot[0]))))
                limit = min(10.0, base_m + speed_m * gs) / PERSON_M * s
                if d > limit:
                    continue
                app = app_distance(a.feat, b.feat)
                if app > app_gate:
                    continue
                cost[i, j] = d / limit + 0.6 * app
        rows, cols = linear_sum_assignment(cost)
        links = {i: j for i, j in zip(rows, cols) if cost[i, j] < BIG}
        if not links:
            break
        has_pred = set(links.values())
        merged, used = [], set()
        for i in range(n):
            if i in has_pred or i in used:
                continue
            chain, k = [i], i
            while k in links:  # follow a -> b -> c so each chain becomes one group
                k = links[k]
                chain.append(k)
            used.update(chain)
            gs_ = [groups[c] for c in chain]
            feats = [g.feat for g in gs_ if g.feat is not None]
            feat = None
            if feats:
                feat = np.average(feats, axis=0, weights=[g.n for g in gs_ if g.feat is not None])
                feat = feat / feat.sum()
            merged.append(Group(np.concatenate([g.pos for g in gs_]), np.concatenate([g.foot for g in gs_]),
                                np.concatenate([g.scale for g in gs_]), np.concatenate([g.ok for g in gs_]),
                                feat, sum(g.n for g in gs_)))
        merged.extend(g for i, g in enumerate(groups) if i not in used)
        groups = merged
    return groups


def _fill_and_smooth(pos: np.ndarray, xy: np.ndarray, max_fill: int, window: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate gaps up to max_fill frames, then moving-average inside each run."""
    order = np.argsort(pos)
    pos, xy = pos[order], xy[order]
    out_pos, out_xy = [pos[0]], [xy[0]]
    for k in range(1, len(pos)):
        gap = pos[k] - pos[k - 1]
        if 1 < gap <= max_fill:
            for s in range(1, gap):
                a = s / gap
                out_pos.append(pos[k - 1] + s)
                out_xy.append((1 - a) * xy[k - 1] + a * xy[k])
        out_pos.append(pos[k])
        out_xy.append(xy[k])
    p, q = np.array(out_pos), np.array(out_xy)
    cuts = np.where(np.diff(p) > 1)[0] + 1
    sm = np.empty_like(q)
    for idx in np.split(np.arange(len(p)), cuts):
        seg = q[idx]
        k = min(window, len(seg) if len(seg) % 2 else len(seg) - 1)
        if k < 3:
            sm[idx] = seg
            continue
        ker, pad = np.ones(k) / k, k // 2
        padded = np.pad(seg, ((pad, pad), (0, 0)), mode="edge")
        sm[idx] = np.stack([np.convolve(padded[:, c], ker, mode="valid") for c in range(2)], 1)
    return p, sm


def finalize(tracks: list[Track], n_frames: int, dt: float, spec: SportSpec, calib: Calibration, *,
             min_len_s: float = 0.5, side_margin: float = 2.0, end_margin: float = 8.0,
             min_in_play: float = 0.5, static_extent_m: float = 3.0, static_min_s: float = 6.0,
             outside_m: float = 0.3, max_fill_s: float = 1.0, smooth_window: int = 5
             ) -> list[list[Person]]:
    """Raw tracklets -> per-frame Person lists with stable ids and per-track teams."""
    kept = []
    for g in stitch(tracks, dt):
        dur = (g.pos.max() - g.pos.min() + 1) * dt
        if dur < min_len_s:
            continue
        xy, valid = calib.project(g.foot)
        valid &= g.ok
        if valid.sum() < 2:
            continue
        x, y = xy[valid, 0], xy[valid, 1]
        # horizon-invalid / cut-off points count as "not in play" so background people drop out
        in_play = np.sum((x >= -end_margin) & (x <= spec.length + end_margin)
                         & (y >= -side_margin) & (y <= spec.width + side_margin)) / len(g.pos)
        if in_play < min_in_play:
            continue
        mx, my = float(np.median(x)), float(np.median(y))
        outside = (mx < -outside_m or mx > spec.length + outside_m or my < -outside_m
                   or my > spec.width + outside_m)
        if outside and max(np.ptp(x), np.ptp(y)) < static_extent_m and dur > static_min_s:
            continue  # line judge / referee / bench: stands outside the court and barely moves
        p, sm = _fill_and_smooth(g.pos[valid], xy[valid], max(1, int(round(max_fill_s / dt))), smooth_window)
        team = 0 if np.mean(sm[:, 0] < spec.net_x) >= 0.5 else 1
        kept.append((p[0], p, sm, team))
    kept.sort(key=lambda k: k[0])
    people: list[list[Person]] = [[] for _ in range(n_frames)]
    for new_id, (_, p, sm, team) in enumerate(kept, start=1):
        for pos, (x, y) in zip(p, sm):
            if 0 <= pos < n_frames:
                people[pos].append(Person(new_id, float(x), float(y), team))
    return people
