"""Perception-level emulator: vehicles on a road grid observed by node-mounted cameras.

The emulator generates what a localisation + tracking + cue-extraction stack would output
(noisy ground-plane positions, detector class scores, light-feature time series) together
with exact ground truth. It does **not** render images; its distributions are explicit,
documented assumptions (see README) and are the main external-validity limitation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from evperc.mapmatch import RoadGraph
from evperc.types import TrackObservation


def build_grid(n: int = 4, block_m: float = 120.0, node_box_m: float = 10.0) -> RoadGraph:
    """Bidirectional n x n street grid; opposite directions are offset 1.75 m from the axis."""
    nodes = {(i, j): np.array([i * block_m, j * block_m]) for i in range(n) for j in range(n)}
    edges: dict[str, list[list[float]]] = {}
    succ: dict[str, list[str]] = {}
    for (i, j), u in nodes.items():
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k, ell = i + di, j + dj
            if (k, ell) not in nodes:
                continue
            d = (nodes[(k, ell)] - u) / block_m
            right = np.array([d[1], -d[0]]) * 1.75
            a, b = u + d * node_box_m + right, nodes[(k, ell)] - d * node_box_m + right
            edges[f"{i}{j}>{k}{ell}"] = [a.tolist(), b.tolist()]
    for eid in edges:
        head = eid.split(">")[1]
        succ[eid] = [o for o in edges if o.startswith(head + ">") and o.split(">")[1] != eid[:2]]
    return RoadGraph(edges, succ)


@dataclass
class VehicleMeta:
    track_id: int
    is_ev: bool
    siren: bool
    confuser: bool


@dataclass(frozen=True)
class TruthRow:
    track_id: int
    t: float
    true_edge: str | None
    dist_to_end_m: float
    speed_mps: float
    heading_rad: float
    is_ev: bool


@dataclass
class SynthConfig:
    fps: float = 15.0
    cam_radius_m: float = 90.0
    n_normal: int = 6
    class_conc: float = 12.0
    siren_prob: float = 0.85
    confuser_prob: float = 0.08
    hazard_prob: float = 0.4
    slow_flicker_prob: float = 0.10


def _route(graph: RoadGraph, rng: np.random.Generator, length: int = 5) -> list[str]:
    ids = list(graph.polylines)
    cur = ids[rng.integers(len(ids))]
    route = [cur]
    while len(route) < length:
        nxt = graph.successors.get(cur, [])
        if not nxt:
            break
        cur = nxt[rng.integers(len(nxt))]
        route.append(cur)
    return route


def _light_series(t: np.ndarray, meta: VehicleMeta, rng: np.random.Generator,
                  cfg: SynthConfig) -> np.ndarray:
    base = 0.05 if meta.is_ev else 0.03
    x = np.full_like(t, base)
    if meta.is_ev and meta.siren:
        f, ph, amp = rng.uniform(2, 4), rng.uniform(0, 2 * math.pi), rng.uniform(0.15, 0.4)
        x = x + amp * (np.sin(2 * math.pi * f * t + ph) > 0)
    elif meta.confuser and rng.random() < cfg.hazard_prob:
        x = x + 0.1 * (np.sin(2 * math.pi * rng.uniform(1.2, 1.8) * t) > 0)
    elif (not meta.is_ev) and rng.random() < cfg.slow_flicker_prob:
        x = x + 0.1 * (0.5 + 0.5 * np.sin(2 * math.pi * 0.3 * t + rng.uniform(0, 6)))
    return np.clip(x + rng.normal(0, 0.03, len(t)), 0.0, 1.0)


def generate_scenario(
    graph: RoadGraph, rng: np.random.Generator, cfg: SynthConfig | None = None
) -> tuple[dict[float, list[TrackObservation]], list[TruthRow], list[VehicleMeta]]:
    """One scenario: one emergency vehicle and ``n_normal`` other vehicles on random routes."""
    cfg = cfg or SynthConfig()
    centers = np.array([pl[0] for pl in graph.polylines.values()])  # unused anchor list
    nodes = np.unique(np.round(np.vstack([
        [pl[0] for pl in graph.polylines.values()], [pl[-1] for pl in graph.polylines.values()]
    ]) / 10.0) * 10.0, axis=0)
    del centers
    frames: dict[float, list[TrackObservation]] = {}
    truth: list[TruthRow] = []
    metas: list[VehicleMeta] = []
    for k in range(cfg.n_normal + 1):
        is_ev = k == 0
        meta = VehicleMeta(k + 1, is_ev, is_ev and rng.random() < cfg.siren_prob,
                           (not is_ev) and rng.random() < cfg.confuser_prob)
        metas.append(meta)
        route = _route(graph, rng)
        pts_l: list[Any] = []
        ranges: list[tuple[float, float, str]] = []
        arc = 0.0
        for e in route:
            pl = graph.polylines[e]
            if pts_l:
                arc += float(np.linalg.norm(pl[0] - pts_l[-1]))
            start = arc
            for v in pl:
                if pts_l:
                    arc += float(np.linalg.norm(v - pts_l[-1])) if v is not pl[0] else 0.0
                pts_l.append(v)
            ranges.append((start, start + graph.length(e), e))
            arc = ranges[-1][1]
        pts = np.array(pts_l)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        speed = rng.uniform(14, 20) if is_ev else rng.uniform(8, 14)
        t0 = rng.uniform(0, 6)
        n_frames = int(cum[-1] / speed * cfg.fps)
        t = t0 + np.arange(n_frames) / cfg.fps
        s = (t - t0) * speed
        xy = np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])
        light = _light_series(t, meta, rng, cfg)
        mu = (rng.uniform(0.55, 0.9) if is_ev else
              rng.uniform(0.4, 0.7) if meta.confuser else rng.uniform(0.02, 0.2))
        for i in range(n_frames):
            d_nodes = np.linalg.norm(nodes - xy[i], axis=1)
            rng_m = float(d_nodes.min())
            p_det = 0.97 - 0.3 * (rng_m / cfg.cam_radius_m) ** 2
            if rng_m > cfg.cam_radius_m or rng.random() > p_det:
                continue
            sig = 0.8 + 0.04 * rng_m
            cs = float(rng.beta(mu * cfg.class_conc, (1 - mu) * cfg.class_conc))
            tt = round(float(t[i]), 6)
            frames.setdefault(tt, []).append(TrackObservation(
                meta.track_id, tt, (float(xy[i, 0] + rng.normal(0, sig)),
                                    float(xy[i, 1] + rng.normal(0, sig))), cs, float(light[i])))
            si = min(int(np.searchsorted(cum, s[i], side="right")) - 1, len(seg) - 1)
            hd = math.atan2(pts[si + 1, 1] - pts[si, 1], pts[si + 1, 0] - pts[si, 0])
            on = next(((a, b, e) for a, b, e in ranges if a <= s[i] <= b), None)
            truth.append(TruthRow(meta.track_id, tt, on[2] if on else None,
                                  (on[1] - s[i]) if on else float("nan"), speed, hd, is_ev))
    return dict(sorted(frames.items())), truth, metas
