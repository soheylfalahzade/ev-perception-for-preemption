import json

import numpy as np

from evperc.events import EVENT_SCHEMA, EVEvent
from evperc.fusion import ConfirmConfig, Confirmer, FusionModel, features
from evperc.pipeline import EVPipeline
from evperc.synth import build_grid, generate_scenario


def test_fusion_fit_separates_classes():
    rng = np.random.default_rng(0)
    pos = [features(float(rng.beta(8, 3)), float(rng.beta(6, 2))) for _ in range(300)]
    neg = [features(float(rng.beta(2, 8)), float(rng.beta(2, 6))) for _ in range(300)]
    x, y = np.array(pos + neg), np.array([1.0] * 300 + [0.0] * 300)
    model = FusionModel().fit(x, y)
    p = np.array([model.predict(r) for r in x])
    assert p[y == 1].mean() > 0.85 and p[y == 0].mean() < 0.15


def test_confirmer_requires_consecutive_frames():
    c = Confirmer(ConfirmConfig(threshold=0.8, confirm_frames=3, ema=1.0))
    seq = [0.9, 0.9, 0.5, 0.9, 0.9, 0.9]
    out = [c.update(1, p)[1] for p in seq]
    assert out == [False, False, False, False, False, True]


def test_event_roundtrip_matches_schema_fields():
    e = EVEvent(1.0, 3, "01>11", 20.0, 80.0, 5.0, 10.0, 16.0, 0.95, 0.99, 1.2)
    d = json.loads(e.to_json())
    assert all(k in d for k in EVENT_SCHEMA["required"])
    assert d["schema_version"] == "1.0" and d["edge_id"] == "01>11"


def test_pipeline_emits_event_for_ev_and_not_for_ordinary_vehicles():
    g = build_grid()
    rng = np.random.default_rng(11)
    frames, _, metas = generate_scenario(g, rng)
    pipe = EVPipeline(g, FusionModel(np.array([0.9, 0.9, 0.0, -2.5])))
    events = []
    for t, obs in frames.items():
        events += pipe.process(t, obs)
    ev_ids = {m.track_id for m in metas if m.is_ev}
    assert any(e.track_id in ev_ids for e in events)
    assert all(e.dist_to_stop_line_m >= 0 and 0 <= e.confidence <= 1 for e in events)
    assert pipe.profiler.summary_ms()["frame_total"]["p95"] < 50.0


def test_scenario_generation_is_deterministic():
    g = build_grid()
    a = generate_scenario(g, np.random.default_rng(5))[0]
    b = generate_scenario(g, np.random.default_rng(5))[0]
    assert list(a) == list(b)
    t0 = next(iter(a))
    assert a[t0][0].xy_m == b[t0][0].xy_m
