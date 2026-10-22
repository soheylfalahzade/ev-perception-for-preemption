"""Frame-level pipeline: observations -> emergency-vehicle events on the road graph."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from evperc.events import EVEvent
from evperc.flash import FlashDetector
from evperc.fusion import ConfirmConfig, Confirmer, FusionModel, features
from evperc.mapmatch import MapMatcher, MatchResult, RoadGraph
from evperc.motion import MotionFilter
from evperc.types import TrackObservation


class LatencyProfiler:
    """Accumulates per-stage wall-clock times (seconds)."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add(self, stage: str, dt: float) -> None:
        self.samples[stage].append(dt)

    def summary_ms(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v) * 1e3
            out[k] = {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                      "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                      "n": float(len(a))}
        return out


class EVPipeline:
    """Online pipeline. Call :meth:`process` once per frame with all tracked observations."""

    def __init__(
        self,
        graph: RoadGraph,
        fusion: FusionModel | None = None,
        confirm: ConfirmConfig | None = None,
        stale_s: float = 5.0,
    ) -> None:
        self.graph = graph
        self.fusion = fusion or FusionModel()
        self.flash = FlashDetector()
        self.matcher = MapMatcher(graph)
        self.confirmer = Confirmer(confirm or ConfirmConfig())
        self.profiler = LatencyProfiler()
        self.stale_s = stale_s
        self._motion: dict[int, MotionFilter] = {}
        self._last_seen: dict[int, float] = {}
        self._emitted: set[tuple[int, str]] = set()
        self.last_features: dict[int, tuple[float, float | None, float]] = {}
        self.last_match: dict[int, MatchResult | None] = {}
        self.last_heading: dict[int, float | None] = {}

    def process(self, t: float, observations: Sequence[TrackObservation]) -> list[EVEvent]:
        """Process one frame; returns newly confirmed events (at most one per track and edge)."""
        t0 = time.perf_counter()
        events: list[EVEvent] = []
        self.last_features, self.last_match, self.last_heading = {}, {}, {}
        for o in observations:
            ts = time.perf_counter()
            mf = self._motion.setdefault(o.track_id, MotionFilter())
            mf.update(o.t, o.xy_m)
            self._last_seen[o.track_id] = t
            tm = time.perf_counter()
            self.flash.add(o.track_id, o.t, o.light_feature)
            fs = self.flash.score(o.track_id)
            tf = time.perf_counter()
            post = self.fusion.predict(features(o.class_score, fs))
            smooth, confirmed = self.confirmer.update(o.track_id, post)
            self.last_features[o.track_id] = (o.class_score, fs, smooth)
            tc = time.perf_counter()
            match = self.matcher.update(o.track_id, mf.position, mf.heading_rad)
            te = time.perf_counter()
            self.last_match[o.track_id] = match
            self.last_heading[o.track_id] = mf.heading_rad
            for stage, dt in (("motion", tm - ts), ("flash", tf - tm), ("fusion", tc - tf),
                              ("map_match", te - tc)):
                self.profiler.add(stage, dt)
            if confirmed and match is not None and (o.track_id, match.edge_id) not in self._emitted:
                self._emitted.add((o.track_id, match.edge_id))
                h = mf.heading_rad
                events.append(EVEvent(
                    timestamp_s=o.t, track_id=o.track_id, edge_id=match.edge_id, s_m=match.s_m,
                    dist_to_stop_line_m=match.dist_to_end_m,
                    eta_s=match.dist_to_end_m / max(mf.speed, 1.0),
                    heading_deg=None if h is None else math.degrees(h), speed_mps=mf.speed,
                    confidence=smooth, match_prob=match.prob,
                ))
        self._expire(t)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        self.profiler.add("frame_total", elapsed_ms / 1e3)
        return [EVEvent(**{**e.__dict__, "latency_ms": elapsed_ms}) for e in events]

    def _expire(self, t: float) -> None:
        for tid in [k for k, v in self._last_seen.items() if t - v > self.stale_s]:
            for store in (self._motion, self._last_seen):
                store.pop(tid, None)
            self.flash.drop(tid)
            self.matcher.drop(tid)
            self.confirmer.drop(tid)
