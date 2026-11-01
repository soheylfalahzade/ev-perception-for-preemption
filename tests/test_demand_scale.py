from pathlib import Path

from evperc.sumo_preemption import scaled_sumocfg


def test_scaled_sumocfg_multiplies_every_flow(tmp_path):
    cfg = Path(__file__).parents[1] / "examples" / "sumo" / "demo.sumocfg"
    out = scaled_sumocfg(str(cfg), 2.0, tmp_path)
    txt = (tmp_path / "demo.rou.xml").read_text(encoding="utf-8")
    assert 'vehsPerHour="1800"' in txt and 'vehsPerHour="500"' in txt
    assert Path(out).exists()
