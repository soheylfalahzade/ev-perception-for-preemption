import math

import numpy as np

from evperc.evaluate import first_alarm, wilson_ci
from evperc.viz import render_scenario_gif


def test_wilson_interval_properties():
    lo, hi = wilson_ci(77, 100)
    assert 0.67 < lo < 0.77 < hi < 0.86
    lo0, hi0 = wilson_ci(0, 600)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.01
    assert math.isnan(wilson_ci(0, 0)[0])
    lo1, hi1 = wilson_ci(10, 10)
    assert hi1 == 1.0 and lo1 > 0.6


def test_first_alarm_requires_consecutive_frames():
    s = np.array([0.1, 0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9])
    assert first_alarm(s, 0.8, 3) == 6
    assert first_alarm(s, 0.8, 5) is None


def test_gif_smoke(tmp_path):
    out = tmp_path / "s.gif"
    info = render_scenario_gif(out, seed=11, max_frames=3)
    assert out.exists() and out.stat().st_size > 1000 and info["frames"] == 3
