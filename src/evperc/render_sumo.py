"""Side-by-side animation of the closed-loop study: no preemption versus preemption at a range.

``evperc-sumo-gif`` replays one traffic seed twice in SUMO (identical until the preemption acts),
draws both runs on the approach lane and writes an animated GIF. Requires SUMO and ``matplotlib``.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from evperc.sumo_preemption import Condition, scaled_sumocfg, simulate

Trace = list


def record(sumocfg: str, seed: int, cond: Condition, max_t: float = 600.0) -> tuple[dict, Trace]:
    """Run one condition and return its summary and per-step trace."""
    trace: Trace = []
    res = simulate((sumocfg, seed, cond, max_t), trace)
    return res, trace


def render_gif(runs: list[tuple[str, dict, Trace]], out: str | Path, fps: int = 8,
               pad_s: float = 12.0, x_range: tuple[float, float] = (180.0, 505.0)) -> None:
    """Draw the runs one above the other and save an animated GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    t0 = min(next(s[0] for s in tr if s[1] is not None) for _, _, tr in runs) - pad_s
    t1 = max(r["ev_time_s"] + next(s[0] for s in tr if s[1] is not None)
             for _, r, tr in runs) + pad_s / 2
    times = np.arange(max(t0, 0.0), t1, 1.0)
    light = {"G": "#3fb27f", "g": "#3fb27f", "y": "#e3b341", "r": "#e5534b"}
    frames = []
    by_t = [{round(s[0]): s for s in tr} for _, _, tr in runs]
    for t in times:
        fig, axes = plt.subplots(len(runs), 1, figsize=(9.6, 1.9 * len(runs) + 0.5), dpi=90)
        fig.patch.set_facecolor("#0d1117")
        for ax, (title, res, _), snap in zip(np.atleast_1d(axes), runs, by_t, strict=True):
            ax.set_facecolor("#161b22")
            ax.set_xlim(*x_range)
            ax.set_ylim(-3.6, 9.6)
            ax.axhspan(0.0, 6.4, color="#21262d", zorder=0)
            ax.axhline(3.2, color="#30363d", lw=0.8, ls=(0, (6, 6)), zorder=1)
            ax.set_yticks([])
            ax.tick_params(colors="#8b949e", labelsize=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
            s = snap.get(int(round(t))) or snap[max(k for k in snap if k <= round(t))] \
                if round(t) >= min(snap) else None
            if s is not None:
                _, ev, cars, state, pre = s
                for x, y, v in cars:
                    c = matplotlib.colormaps["RdYlGn"](min(v / 13.9, 1.0))
                    ax.add_patch(FancyBboxPatch((x - 4.5, y - 0.9), 4.5, 1.8,
                                                boxstyle="round,pad=0.05", fc=c, ec="none",
                                                zorder=3))
                if ev is not None:
                    ax.add_patch(FancyBboxPatch((ev[0] - 5.5, ev[1] - 1.1), 5.5, 2.2,
                                                boxstyle="round,pad=0.05", fc="#ffffff",
                                                ec="#ff3b30", lw=2.0, zorder=4))
                    if int(t) % 2 == 0:
                        ax.plot([ev[0] - 2.7], [ev[1] + 1.6], "o", color="#ff3b30", ms=5,
                                zorder=5)
                    else:
                        ax.plot([ev[0] - 2.7], [ev[1] + 1.6], "o", color="#0a84ff", ms=5,
                                zorder=5)
                ax.plot([300], [-1.9], "s", color=light.get(state[0], "#888"), ms=11, zorder=6)
                ax.text(303, -1.9, "forced green" if pre else "signal", color="#8b949e",
                        fontsize=8, va="center")
            ax.text(x_range[1] - 3, 8.4, f"{title}   EV delay {res['ev_time_s']:.1f} s",
                    color="#e6edf3", fontsize=10, ha="right", va="center")
        fig.suptitle(f"t = {t:.0f} s", color="#8b949e", fontsize=9, x=0.02, ha="left", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        frames.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)
    adaptive = Image.Palette.ADAPTIVE
    pal = [f.convert("P", palette=adaptive, colors=64) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0,
                optimize=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``evperc-sumo-gif``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sumocfg", default="examples/sumo/demo.sumocfg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range-m", type=float, default=90.0)
    ap.add_argument("--demand-scale", type=float, default=1.5)
    ap.add_argument("--out", default="docs/assets/sumo_preemption.gif")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = scaled_sumocfg(args.sumocfg, args.demand_scale, tmp)
        runs = []
        for title, cond in (("no preemption", Condition("none", None)),
                            (f"preemption from {args.range_m:g} m",
                             Condition("pre", args.range_m, 0.5))):
            res, tr = record(cfg, args.seed, cond)
            runs.append((title, res, tr))
    render_gif(runs, args.out)
    print(f"wrote {args.out}: " + ", ".join(f"{t} {r['ev_time_s']:.1f} s" for t, r, _ in runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
