"""Road graph and online HMM map matching with heading consistency."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class RoadGraph:
    """Directed edges as polylines (direction of travel = vertex order) plus successor lists.

    The end of an edge is taken to be its stop line / junction entry.
    """

    def __init__(
        self,
        edges: Mapping[str, Sequence[Sequence[float]]],
        successors: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.polylines = {k: np.asarray(v, dtype=float) for k, v in edges.items()}
        self.successors = {k: list(v) for k, v in (successors or {}).items()}
        self._seg: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for k, pl in self.polylines.items():
            a, b = pl[:-1], pl[1:]
            vec = b - a
            ln = np.linalg.norm(vec, axis=1)
            self._seg[k] = (a, vec, ln, np.concatenate([[0.0], np.cumsum(ln)[:-1]]))

    @classmethod
    def from_sumo_net(cls, path: str) -> RoadGraph:
        """Load non-internal edges (first lane's shape) and connections from a SUMO ``.net.xml``."""
        root = ET.parse(path).getroot()
        edges: dict[str, list[list[float]]] = {}
        for e in root.iter("edge"):
            if e.get("function") == "internal":
                continue
            lane = e.find("lane")
            if lane is None:
                continue
            shape = lane.get("shape") or ""
            pts = [[float(v) for v in p.split(",")] for p in shape.split()]
            edges[str(e.get("id"))] = pts
        succ: dict[str, list[str]] = {k: [] for k in edges}
        for c in root.iter("connection"):
            f, t = c.get("from"), c.get("to")
            if f in edges and t in edges and t not in succ[f]:
                succ[f].append(t)
        return cls(edges, succ)

    def length(self, edge_id: str) -> float:
        _, _, ln, _ = self._seg[edge_id]
        return float(ln.sum())

    def project(self, edge_id: str, xy: Sequence[float]) -> tuple[float, float, float]:
        """Return ``(distance_m, arc_position_m, tangent_heading_rad)`` of the nearest point."""
        a, vec, ln, cum = self._seg[edge_id]
        p = np.asarray(xy, dtype=float)
        t = np.clip(((p - a) * vec).sum(axis=1) / np.maximum(ln**2, 1e-12), 0.0, 1.0)
        q = a + vec * t[:, None]
        d = np.linalg.norm(p - q, axis=1)
        i = int(np.argmin(d))
        return float(d[i]), float(cum[i] + t[i] * ln[i]), float(math.atan2(vec[i, 1], vec[i, 0]))

    def candidates(self, xy: Sequence[float], radius_m: float) -> list[str]:
        """Edges whose nearest point lies within ``radius_m`` of ``xy``."""
        return [k for k in self.polylines if self.project(k, xy)[0] <= radius_m]

    def nearest_edge(self, xy: Sequence[float]) -> str:
        """Baseline: purely geometric nearest edge."""
        return min(self.polylines, key=lambda k: self.project(k, xy)[0])


@dataclass(frozen=True)
class MatchResult:
    """Posterior map-matching result for one track and frame."""

    edge_id: str
    prob: float
    s_m: float
    lateral_m: float
    edge_length_m: float

    @property
    def dist_to_end_m(self) -> float:
        return max(self.edge_length_m - self.s_m, 0.0)


class MapMatcher:
    """Online HMM filter over candidate edges per track (emission: distance and heading)."""

    def __init__(
        self,
        graph: RoadGraph,
        radius_m: float = 25.0,
        sigma_pos_m: float = 5.0,
        sigma_heading_rad: float = 0.5,
        p_stay: float = 0.97,
        p_next: float = 0.025,
        p_other: float = 1e-4,
    ) -> None:
        self.graph, self.radius = graph, radius_m
        self.sp, self.sh = sigma_pos_m, sigma_heading_rad
        self.lt = (math.log(p_stay), math.log(p_next), math.log(p_other))
        self._state: dict[int, dict[str, float]] = {}

    def drop(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def update(
        self, track_id: int, xy: Sequence[float], heading_rad: float | None
    ) -> MatchResult | None:
        """Advance the filter of ``track_id`` with a smoothed position and heading."""
        cands = self.graph.candidates(xy, self.radius)
        if not cands:
            return None
        prev = self._state.get(track_id)
        scored: dict[str, float] = {}
        info: dict[str, tuple[float, float]] = {}
        for e in cands:
            d, s, th = self.graph.project(e, xy)
            em = -0.5 * (d / self.sp) ** 2
            if heading_rad is not None:
                em += -0.5 * (_wrap(heading_rad - th) / self.sh) ** 2
            if prev:
                terms = []
                for pe, plp in prev.items():
                    if pe == e:
                        lt = self.lt[0]
                    elif e in self.graph.successors.get(pe, ()):
                        lt = self.lt[1]
                    else:
                        lt = self.lt[2]
                    terms.append(plp + lt)
                m = max(terms)
                prior = m + math.log(sum(math.exp(x - m) for x in terms))
            else:
                prior = 0.0
            scored[e] = prior + em
            info[e] = (d, s)
        m = max(scored.values())
        z = m + math.log(sum(math.exp(v - m) for v in scored.values()))
        self._state[track_id] = {e: v - z for e, v in scored.items()}
        best = max(scored, key=lambda k: scored[k])
        d, s = info[best]
        return MatchResult(best, math.exp(scored[best] - z), s, d, self.graph.length(best))
