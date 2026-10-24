import math

import numpy as np

from evperc.mapmatch import MapMatcher, RoadGraph
from evperc.motion import MotionFilter
from evperc.synth import build_grid


def _parallel_roads():
    edges = {"A": [[0, 0], [200, 0]], "B": [[0, 12], [200, 12]], "C": [[200, 0], [200, 100]]}
    return RoadGraph(edges, {"A": ["C"], "B": [], "C": []})


def test_projection_and_edge_end_distance():
    g = _parallel_roads()
    d, s, th = g.project("A", (50.0, 3.0))
    assert abs(d - 3.0) < 1e-9 and abs(s - 50.0) < 1e-9 and abs(th) < 1e-9
    assert abs(g.length("C") - 100.0) < 1e-9


def test_heading_disambiguates_opposite_directions():
    g = build_grid()
    a = next(k for k in g.polylines if k == "01>11")
    b = "11>01"
    mid = 0.5 * (np.array(g.polylines[a][0]) + np.array(g.polylines[a][1]))
    m = MapMatcher(g)
    res = m.update(1, mid, 0.0)  # heading east -> the eastbound edge
    assert res is not None and res.edge_id == a
    m2 = MapMatcher(g)
    res2 = m2.update(2, mid, math.pi)
    assert res2 is not None and res2.edge_id == b


def test_hmm_follows_successor_edge():
    g = _parallel_roads()
    m = MapMatcher(g)
    last = None
    for x in np.linspace(120, 200, 20):
        last = m.update(1, (x, 1.0), 0.0)
    for y in np.linspace(5, 60, 20):
        last = m.update(1, (200.0, y), math.pi / 2)
    assert last is not None and last.edge_id == "C"
    assert last.dist_to_end_m < 60.0


def test_motion_filter_recovers_velocity_and_heading():
    rng = np.random.default_rng(0)
    mf = MotionFilter(meas_sigma_m=1.0)
    for k in range(60):
        t = k / 15.0
        mf.update(t, (10.0 * t + rng.normal(0, 1.0), 4.0 * t + rng.normal(0, 1.0)))
    assert abs(mf.speed - math.hypot(10, 4)) < 1.0
    assert abs(mf.heading_rad - math.atan2(4, 10)) < 0.1
    still = MotionFilter()
    for k in range(30):
        still.update(k / 15.0, (5.0, 5.0))
    assert still.heading_rad is None
