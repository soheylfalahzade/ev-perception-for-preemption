"""Data-driven animation of a scenario: tracks, cue posterior and the emitted event."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from evperc.fusion import ConfirmConfig, FusionModel
from evperc.pipeline import EVPipeline
from evperc.synth import build_grid, generate_scenario


def render_scenario_gif(out: str | Path, seed: int = 11, results_json: str | Path | None = None,
                        threshold: float = 0.85, step_s: float = 0.3, fps: int = 10,
                        max_frames: int = 80) -> dict:
    """Run the pipeline on one emulated scenario and write an animated GIF.

    Left: road grid with observed track positions coloured by the smoothed emergency posterior
    (ring = ground-truth emergency vehicle, used for display only). Right: posterior per track
    and the confirmation threshold; a star marks the emitted event.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    weights = None
    if results_json is not None:
        weights = np.array(json.loads(Path(results_json).read_text())["fusion_weights"]["fused"])
    graph = build_grid()
    frames, truth, metas = generate_scenario(graph, np.random.default_rng(seed))
    is_ev = {m.track_id: m.is_ev for m in metas}
    pipe = EVPipeline(graph, FusionModel(weights) if weights is not None else FusionModel(),
                      ConfirmConfig(threshold=threshold, confirm_frames=5))
    log: list[tuple[float, list, dict, list]] = []
    for t, obs in frames.items():
        ev = pipe.process(t, obs)
        post = {o.track_id: pipe.last_features[o.track_id][2] for o in obs}
        log.append((t, obs, post, ev))
    ev_times = [t for t, obs, _, _ in log if any(is_ev[o.track_id] for o in obs)]
    t0, t1 = min(ev_times), max(ev_times)
    grid_t = np.arange(t0, t1, step_s)
    idx = [int(np.searchsorted([r[0] for r in log], g)) for g in grid_t][:max_frames]
    series: dict[int, tuple[list[float], list[float]]] = {m.track_id: ([], []) for m in metas}
    for t, _obs, post, _ in log:
        for tid, p in post.items():
            series[tid][0].append(t)
            series[tid][1].append(p)
    alarms = [(r[0], e) for r in log for e in r[3]]
    images = []
    cmap = plt.get_cmap("viridis")
    for i in idx:
        t, obs, _, _ = log[min(i, len(log) - 1)]
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=72,
                                   gridspec_kw={"width_ratios": [1, 1.15]})
        for pl in graph.polylines.values():
            a.plot(pl[:, 0], pl[:, 1], color="#c8c8c8", lw=1.0, zorder=1)
        hit = [e for (ta, e) in alarms if ta <= t]
        for e in hit:
            pl = graph.polylines[e.edge_id]
            a.plot(pl[:, 0], pl[:, 1], color="#d62728", lw=3.0, zorder=2)
        for o in obs:
            p = dict(zip([x.track_id for x in obs], [log[min(i, len(log) - 1)][2][x.track_id]
                                                      for x in obs], strict=True))[o.track_id]
            a.scatter(*o.xy_m, s=46, color=cmap(p), zorder=3, edgecolor="none")
            if is_ev[o.track_id]:
                a.scatter(*o.xy_m, s=150, facecolor="none", edgecolor="#d62728", lw=1.6, zorder=4)
        a.set_xlim(-30, 390)
        a.set_ylim(-30, 390)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"t = {t - t0:4.1f} s   (ring: ground-truth EV)", fontsize=9)
        for tid, (ts, ps) in series.items():
            m = np.array(ts) <= t
            if m.any():
                ev_track = is_ev[tid]
                b.plot(np.array(ts)[m] - t0, np.array(ps)[m], lw=2.4 if ev_track else 1.0,
                       color="#d62728" if ev_track else "#9a9a9a",
                       alpha=1.0 if ev_track else 0.7)
        b.axhline(threshold, color="k", ls="--", lw=0.8)
        for ta, e in alarms:
            if ta <= t:
                b.plot(ta - t0, e.confidence, marker="*", ms=14, color="gold", mec="k", zorder=5)
        b.set_xlim(0, t1 - t0)
        b.set_ylim(0, 1.02)
        b.set_xlabel("time [s]", fontsize=8)
        b.set_ylabel("smoothed EV posterior", fontsize=8)
        b.tick_params(labelsize=7)
        b.set_title("cue fusion and confirmation", fontsize=9)
        fig.tight_layout()
        canvas = cast(FigureCanvasAgg, fig.canvas)
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
                      .quantize(colors=96))
        plt.close(fig)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return {"frames": len(images), "events": len(alarms)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``evperc-gif``)."""
    import argparse

    ap = argparse.ArgumentParser(description=render_scenario_gif.__doc__)
    ap.add_argument("--out", default="docs/assets/scenario.gif")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--results", default="docs/results/results.json")
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(render_scenario_gif(args.out, args.seed, args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
