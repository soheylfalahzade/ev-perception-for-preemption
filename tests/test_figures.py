import json

from evperc.figures import RANGES, plot_demand


def test_plot_demand_writes_png(tmp_path):
    lvl = {"none": {"ev_time_mean_s": 30.0, "delta_vs_none_s": 0.0, "delta_ci95": [0, 0]}}
    for r in RANGES:
        lvl[f"range_{r}m"] = {"ev_time_mean_s": 20.0, "delta_vs_none_s": -10.0,
                              "delta_ci95": [-15.0, -5.0]}
    out = tmp_path / "f.png"
    plot_demand({"x1": lvl}, out)
    assert out.stat().st_size > 5000
    json.dumps(lvl)
