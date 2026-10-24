"""Evaluation: cue ablation, operating points, lead distance, map matching, latency.

``evperc-eval`` generates emulated scenarios (see :mod:`evperc.synth`), fits the fusion models and
the operating threshold on development seeds, and reports on disjoint held-out seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import evperc
from evperc.fusion import FusionModel, features
from evperc.pipeline import EVPipeline
from evperc.synth import SynthConfig, TruthRow, build_grid, generate_scenario

MODELS = ("class_only", "flash_only", "fused")
DEV_SEED0, TEST_SEED0 = 1000, 0


@dataclass
class TrackLog:
    """Per-track series needed for offline operating-point analysis."""

    is_ev: bool
    t: np.ndarray
    class_score: np.ndarray
    flash: list[float | None]
    dist_to_end: np.ndarray
    speed: float


def _mask(x: np.ndarray, model: str) -> np.ndarray:
    x = x.copy()
    if model == "class_only":
        x[1:3] = 0.0
    elif model == "flash_only":
        x[0] = 0.0
    return x


def run_scenarios(n: int, seed0: int, cfg: SynthConfig | None = None):
    """Run the live pipeline on ``n`` scenarios; collect logs, match accuracy and latency."""
    graph = build_grid()
    logs: list[TrackLog] = []
    hmm_ok, nn_ok, n_match, head_err = 0, 0, 0, []
    profiler_frames: list[float] = []
    stages: dict[str, list[float]] = {}
    for k in range(n):
        rng = np.random.default_rng(seed0 + k)
        frames, truth, metas = generate_scenario(graph, rng, cfg)
        tr = {(r.track_id, r.t): r for r in truth}
        pipe = EVPipeline(graph)
        per: dict[int, dict[str, list]] = {m.track_id: {"t": [], "c": [], "f": [], "d": []}
                                          for m in metas}
        for t, obs in frames.items():
            pipe.process(t, obs)
            for o in obs:
                row: TruthRow = tr[(o.track_id, o.t)]
                cs, fs, _ = pipe.last_features[o.track_id]
                d = per[o.track_id]
                d["t"].append(o.t)
                d["c"].append(cs)
                d["f"].append(fs)
                d["d"].append(row.dist_to_end_m)
                m = pipe.last_match.get(o.track_id)
                if row.true_edge is not None and m is not None:
                    n_match += 1
                    hmm_ok += m.edge_id == row.true_edge
                    nn_ok += graph.nearest_edge(pipe._motion[o.track_id].position) == row.true_edge
                h = pipe.last_heading.get(o.track_id)
                if h is not None and row.speed_mps >= 3.0:
                    head_err.append(abs((h - row.heading_rad + math.pi) % (2 * math.pi) - math.pi))
        for meta in metas:
            d = per[meta.track_id]
            if d["t"]:
                logs.append(TrackLog(meta.is_ev, np.array(d["t"]), np.array(d["c"]), d["f"],
                                     np.array(d["d"]), 0.0))
        profiler_frames += pipe.profiler.samples["frame_total"]
        for st in ("motion", "flash", "fusion", "map_match"):
            stages.setdefault(st, []).extend(pipe.profiler.samples[st])
    return logs, {"hmm_acc": hmm_ok / max(n_match, 1), "nn_acc": nn_ok / max(n_match, 1),
                  "n_match": n_match, "heading_err_deg_median": float(
                      np.degrees(np.median(head_err))) if head_err else float("nan")}, (
        profiler_frames, stages)


def fit_models(logs: list[TrackLog]) -> dict[str, FusionModel]:
    """Fit one logistic model per cue configuration on frame-level labelled features."""
    x = np.array([features(c, f) for lg in logs for c, f in zip(lg.class_score, lg.flash,
                                                                 strict=True)])
    y = np.array([float(lg.is_ev) for lg in logs for _ in lg.t])
    return {m: FusionModel().fit(np.array([_mask(r, m) for r in x]), y) for m in MODELS}


def posterior_series(lg: TrackLog, model: FusionModel, name: str, ema: float = 0.4) -> np.ndarray:
    """EMA-smoothed posterior over a track (same rule as the online confirmer)."""
    out, prev = [], None
    for c, f in zip(lg.class_score, lg.flash, strict=True):
        p = model.predict(_mask(features(c, f), name))
        prev = p if prev is None else ema * p + (1 - ema) * prev
        out.append(prev)
    return np.array(out)


def first_alarm(series: np.ndarray, thr: float, n_confirm: int) -> int | None:
    """Index of the frame where ``n_confirm`` consecutive values reach ``thr``."""
    run = 0
    for i, v in enumerate(series):
        run = run + 1 if v >= thr else 0
        if run >= n_confirm:
            return i
    return None


def evaluate_models(logs, models, thresholds, n_confirm: int = 5) -> dict:
    """Operating curve per model: detection rate, false-alarm rate, lead distance."""
    out: dict[str, list[dict]] = {}
    for name in MODELS:
        series = [posterior_series(lg, models[name], name) for lg in logs]
        rows = []
        for thr in thresholds:
            det, fa, lead = 0, 0, []
            n_ev = sum(lg.is_ev for lg in logs)
            n_non = len(logs) - n_ev
            for lg, s in zip(logs, series, strict=True):
                i = first_alarm(s, thr, n_confirm)
                if i is None:
                    continue
                if lg.is_ev:
                    det += 1
                    if np.isfinite(lg.dist_to_end[i]):
                        lead.append(float(lg.dist_to_end[i]))
                else:
                    fa += 1
            rows.append({"thr": float(thr), "detection_rate": det / max(n_ev, 1),
                         "false_alarm_rate": fa / max(n_non, 1),
                         "median_lead_m": float(np.median(lead)) if lead else float("nan")})
        out[name] = rows
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pick_threshold(rows: list[dict], max_far: float) -> float:
    """Smallest threshold whose development false-alarm rate does not exceed ``max_far``."""
    ok = [r["thr"] for r in rows if r["false_alarm_rate"] <= max_far]
    return min(ok) if ok else max(r["thr"] for r in rows)


def auc(logs, models, name: str) -> float:
    """Track-level AUC using the maximum smoothed posterior of each track."""
    sc = np.array([posterior_series(lg, models[name], name).max() for lg in logs])
    y = np.array([lg.is_ev for lg in logs])
    pos, neg = sc[y], sc[~y]
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``evperc-eval``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", type=int, default=60)
    ap.add_argument("--test", type=int, default=120)
    ap.add_argument("--max-far", type=float, default=0.01)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev_logs, _, _ = run_scenarios(args.dev, DEV_SEED0)
    models = fit_models(dev_logs)
    grid = np.concatenate([np.linspace(0.3, 0.95, 14), [0.97, 0.98, 0.99, 0.995]])
    dev_curves = evaluate_models(dev_logs, models, grid)
    chosen = {m: pick_threshold(dev_curves[m], args.max_far) for m in MODELS}
    test_logs, match, (frame_t, stages) = run_scenarios(args.test, TEST_SEED0)
    curves = evaluate_models(test_logs, models, grid)
    table = {}
    for m in MODELS:
        r = min(curves[m], key=lambda x: abs(x["thr"] - chosen[m]))
        n_ev = sum(lg.is_ev for lg in test_logs)
        n_non = len(test_logs) - n_ev
        table[m] = {**r, "auc": auc(test_logs, models, m),
                    "detection_ci95": wilson_ci(round(r["detection_rate"] * n_ev), n_ev),
                    "false_alarm_ci95": wilson_ci(round(r["false_alarm_rate"] * n_non), n_non)}
    lat = {"frame_total_ms": {p: float(np.percentile(np.array(frame_t) * 1e3, q))
                              for p, q in (("p50", 50), ("p95", 95), ("p99", 99))},
           "stages_p95_ms": {k: float(np.percentile(np.array(v) * 1e3, 95))
                             for k, v in stages.items()}}
    result = {"operating_point": {"max_dev_false_alarm_rate": args.max_far, "thresholds": chosen},
              "test": table, "map_matching": match, "latency_cpu": lat,
              "n_dev_tracks": len(dev_logs), "n_test_tracks": len(test_logs),
              "n_test_ev_tracks": int(sum(lg.is_ev for lg in test_logs)),
              "fusion_weights": {m: models[m].weights.tolist() for m in MODELS},
              "versions": {"evperc": evperc.__version__, "python": platform.python_version(),
                           "numpy": np.__version__}, "curves": curves,
              "synth_config": SynthConfig().__dict__}
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _figure(curves, table, out / "operating_curves.png")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("operating_point", "test", "map_matching", "latency_cpu",
                               "n_test_tracks", "n_test_ev_tracks")}, indent=2))
    return 0


def _figure(curves: dict, table: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in MODELS:
        far = [r["false_alarm_rate"] for r in curves[m]]
        dr = [r["detection_rate"] for r in curves[m]]
        ax[0].plot(far, dr, marker="o", ms=3, label=f"{m} (AUC {table[m]['auc']:.3f})")
        ax[1].plot(far, [r["median_lead_m"] for r in curves[m]], marker="o", ms=3, label=m)
    ax[0].set_xlabel("false-alarm rate (per non-EV track)")
    ax[0].set_ylabel("detection rate (EV tracks)")
    ax[0].set_title("Operating curves (held-out)")
    ax[1].set_xlabel("false-alarm rate (per non-EV track)")
    ax[1].set_ylabel("median distance to edge end at alarm [m]")
    ax[1].set_title("Lead distance")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
