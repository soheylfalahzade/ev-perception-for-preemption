from pathlib import Path

import numpy as np

from evperc.adapters import observations_from_detections
from evperc.mapmatch import RoadGraph


def test_adapter_builds_observations_with_light_feature_and_class_score():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[10:30, 60:100] = (20, 20, 250)  # red roof patch of the second vehicle
    dets = [(1, "car", (10.0, 40.0, 50.0, 100.0), 0.9),
            (2, "ambulance", (60.0, 10.0, 100.0, 100.0), 0.8)]
    obs = observations_from_detections(1.5, frame, dets, lambda p: p * 0.1)
    assert [o.track_id for o in obs] == [1, 2]
    assert obs[0].class_score == 0.0 and abs(obs[1].class_score - 0.8) < 1e-12
    assert obs[0].light_feature == 0.0 and obs[1].light_feature > 0.1
    assert np.allclose(obs[1].xy_m, (8.0, 10.0))


def test_sumo_net_loader_reads_edges_and_successors():
    net = Path(__file__).parents[1] / "examples" / "sumo" / "demo.net.xml"
    g = RoadGraph.from_sumo_net(str(net))
    assert {"app_E", "in_E", "out_W", "exit_W"} <= set(g.polylines)
    assert "in_E" in g.successors["app_E"]
    assert abs(g.length("app_E") - 200.0) < 1e-6
