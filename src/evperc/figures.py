"""Figures for the demand sweep (``evperc-figures``)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RANGES = (30, 60, 90, 120, 150)


def plot_demand(by_demand: dict, out: str | Path) -> None:
    """Benefit of preemption against detection range, one line per demand level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = list(by_demand)
    cols = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [3, 2]})
    for lvl, col in zip(levels, cols, strict=False):
        c = by_demand[lvl]
        y = [-c[f"range_{r}m"]["delta_vs_none_s"] for r in RANGES]
        lo = [-c[f"range_{r}m"]["delta_ci95"][1] for r in RANGES]
        hi = [-c[f"range_{r}m"]["delta_ci95"][0] for r in RANGES]
        a.plot(RANGES, y, "-o", color=col, label=f"demand {lvl}", lw=2)
        a.fill_between(RANGES, lo, hi, color=col, alpha=0.15, lw=0)
    a.set_xlabel("detection range (m)")
    a.set_ylabel("delay saved for the emergency vehicle (s)")
    a.set_title("Benefit of earlier detection depends on demand", fontsize=11)
    a.set_xticks(RANGES)
    a.legend(frameon=False)
    a.grid(alpha=0.25)
    base = [by_demand[lvl]["none"]["ev_time_mean_s"] for lvl in levels]
    b.bar(levels, base, color=cols[:len(levels)])
    for i, v in enumerate(base):
        b.text(i, v + 1.5, f"{v:.0f} s", ha="center", fontsize=9)
    b.set_ylabel("delay without preemption (s)")
    b.set_title("Baseline delay", fontsize=11)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-demand", default="docs/results/sumo_demand.json")
    ap.add_argument("--out", default="docs/assets/demand_sweep.png")
    args = ap.parse_args(argv)
    plot_demand(json.loads(Path(args.by_demand).read_text(encoding="utf-8")), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
