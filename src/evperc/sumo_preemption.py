"""Closed-loop preemption study in SUMO: benefit of preempting as a function of detection lead.

An emergency vehicle is inserted into a queued approach controlled by a fixed-time light. A
perception model (detection range, latency, miss probability, false-event rate) decides *when* the
light is forced green. Every condition replays the same seed, so traffic is identical until the
preemption acts (paired design). Perception is abstracted to those four parameters; the lead
distance they should be set to is what ``evperc-eval`` measures.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

APPROACH_EDGE, LANE_LEN_M, TL_ID, EXIT_EDGE = "app_E", 200.0, "M_E", "out_W"


@dataclass(frozen=True)
class Condition:
    """One experimental condition."""

    name: str
    detect_range_m: float | None  # None = no preemption
    latency_s: float = 0.5
    miss_prob: float = 0.0
    false_events_per_h: float = 0.0


def run_one(args: tuple[str, int, Condition, float]) -> dict:
    """Simulate one (seed, condition) pair and return the EV passage time and cost proxies."""
    import traci

    sumocfg, seed, cond, max_t = args
    label = f"s{seed}_{cond.name}"
    traci.start(["sumo", "-c", sumocfg, "--seed", str(seed), "--no-step-log", "true",
                 "--no-warnings", "true"], label=label)
    conn = traci.getConnection(label)
    rng = np.random.default_rng(seed)
    t_ev = float(rng.uniform(150.0, 330.0))
    t_false = list(np.cumsum(rng.exponential(3600.0 / cond.false_events_per_h, 200))
                   ) if cond.false_events_per_h > 0 else []
    result: dict[str, Any] = {"seed": seed, "condition": cond.name, "ev_time_s": float("nan"),
              "forced_green_s": 0.0, "false_events": 0, "false_during_red": 0}
    inserted, t_in, scheduled, fire_at = False, None, False, None
    preempt_start, false_until = None, None
    try:
        while conn.simulation.getTime() < max_t:
            conn.simulationStep()
            t = conn.simulation.getTime()
            if not inserted and t >= t_ev:
                conn.vehicle.add("ev", "r_w", typeID="ev", departSpeed="max")
                inserted = True
            if inserted and t_in is None and "ev" in conn.simulation.getDepartedIDList():
                t_in = t
            road = conn.vehicle.getRoadID("ev") if "ev" in conn.vehicle.getIDList() else None
            if road == EXIT_EDGE and t_in is not None:
                result["ev_time_s"] = t - t_in
                break
            if cond.detect_range_m is not None and road == APPROACH_EDGE and not scheduled:
                dist = LANE_LEN_M - conn.vehicle.getLanePosition("ev")
                if dist <= cond.detect_range_m:
                    scheduled = True
                    if rng.random() >= cond.miss_prob:
                        fire_at = t + cond.latency_s
            if fire_at is not None and t >= fire_at and preempt_start is None:
                conn.trafficlight.setRedYellowGreenState(TL_ID, "GG")
                preempt_start = t
            if preempt_start is not None and (road not in (APPROACH_EDGE, None)
                                              or t - preempt_start > 60.0):
                conn.trafficlight.setProgram(TL_ID, "0")
                result["forced_green_s"] += t - preempt_start
                preempt_start, fire_at = None, None
            while t_false and t >= t_false[0]:
                t_false.pop(0)
                if preempt_start is None and false_until is None:
                    result["false_events"] += 1
                    result["false_during_red"] += int("r" in
                                                      conn.trafficlight.getRedYellowGreenState(TL_ID))
                    conn.trafficlight.setRedYellowGreenState(TL_ID, "GG")
                    false_until = t + 10.0
            if false_until is not None and t >= false_until and preempt_start is None:
                conn.trafficlight.setProgram(TL_ID, "0")
                result["forced_green_s"] += 10.0
                false_until = None
    finally:
        conn.close()
    return result


def default_conditions(latency_s: float, miss: float, false_per_h: float) -> list[Condition]:
    """Baseline, detection-range sweep without misses, and one realistic imperfect condition."""
    conds = [Condition("none", None)]
    conds += [Condition(f"range_{int(r)}m", float(r), latency_s) for r in (30, 60, 90, 120, 150)]
    conds.append(Condition("realistic_90m", 90.0, latency_s, miss, false_per_h))
    return conds


def summarise(rows: list[dict]) -> dict:
    """Per-condition mean passage time and paired difference vs. the no-preemption baseline."""
    import pandas as pd

    df = pd.DataFrame(rows)
    piv = df.pivot(index="seed", columns="condition", values="ev_time_s")
    out: dict = {"n_seeds": int(len(piv)), "conditions": {}}
    rng = np.random.default_rng(0)
    for c in piv.columns:
        d = (piv[c] - piv["none"]).dropna().to_numpy()
        boots = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) else [np.nan]
        sub = df[df.condition == c]
        out["conditions"][c] = {
            "ev_time_mean_s": float(piv[c].mean()),
            "delta_vs_none_s": float(d.mean()) if len(d) else float("nan"),
            "delta_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "forced_green_s_mean": float(sub.forced_green_s.mean()),
            "false_events_mean": float(sub.false_events.mean()),
            "false_during_red_mean": float(sub.false_during_red.mean()),
        }
    return out


def scaled_sumocfg(sumocfg: str, scale: float, workdir: str | Path) -> str:
    """Copy the scenario to ``workdir`` with every ``vehsPerHour`` multiplied by ``scale``."""
    src = Path(sumocfg).parent
    dst = Path(workdir)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, dst / f.name)
    for rou in dst.glob("*.rou.xml"):
        txt = rou.read_text(encoding="utf-8")
        txt = re.sub(r'vehsPerHour="([0-9.]+)"',
                     lambda m: f'vehsPerHour="{float(m.group(1)) * scale:g}"', txt)
        rou.write_text(txt, encoding="utf-8")
    return str(dst / Path(sumocfg).name)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``evperc-sumo``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sumocfg", default="examples/sumo/demo.sumocfg")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--latency", type=float, default=0.5)
    ap.add_argument("--miss", type=float, default=0.08)
    ap.add_argument("--false-per-h", type=float, default=6.0)
    ap.add_argument("--max-t", type=float, default=420.0)
    ap.add_argument("--out", default="results/sumo")
    ap.add_argument("--demand-scale", type=float, nargs="+", default=[1.0],
                    help="multipliers of the scenario demand (e.g. 0.5 1 1.5 2); one study each")
    args = ap.parse_args(argv)
    conds = default_conditions(args.latency, args.miss, args.false_per_h)
    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_demand: dict[str, dict] = {}
    for scale in args.demand_scale:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = args.sumocfg if scale == 1.0 else scaled_sumocfg(args.sumocfg, scale, tmp)
            jobs = [(cfg, args.seed0 + s, c, args.max_t)
                    for s in range(args.seeds) for c in conds]
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                rows = list(pool.map(run_one, jobs))
        sub = out if len(args.demand_scale) == 1 else out / f"demand_x{scale:g}"
        sub.mkdir(parents=True, exist_ok=True)
        summary = summarise(rows)
        summary["conditions_spec"] = [asdict(c) for c in conds]
        summary["demand_scale"] = scale
        (sub / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(sub / "runs.csv", index=False)
        by_demand[f"x{scale:g}"] = summary["conditions"]
        print(f"demand x{scale:g}")
        print(json.dumps(summary["conditions"], indent=2))
    if len(args.demand_scale) > 1:
        (out / "by_demand.json").write_text(json.dumps(by_demand, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
