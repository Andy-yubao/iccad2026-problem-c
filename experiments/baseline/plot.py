"""Render saved layouts/metrics only; does not load datasets or run a solver."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    result = json.loads(args.results.read_text(encoding="utf-8"))
    if result["status"] != "completed":
        raise ValueError("Plot only a completed run with no execution errors")
    cases, strategies = result["cases"], result["strategies"]
    out = args.results.resolve().parent
    colors = {"row": "#aeb7c2", "pack": "#e7a543", "search": "#168779", "no_soft": "#ae6592"}
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    width = .8 / len(strategies)
    for k, strategy in enumerate(strategies):
        values = [c["variants"][strategy]["metrics"]["cost"] for c in cases]
        x = np.arange(len(cases)) - .4 + width * (k + .5)
        ax.bar(x, values, width=width, label=strategy, color=colors.get(strategy))
        for xx, value in zip(x, values):
            ax.text(xx, value + .07, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(cases)), [f"dev {i+1}\nn={c['input_summary']['block_count']}" for i, c in enumerate(cases)])
    ax.set_ylim(0, 11.2)
    ax.set_ylabel("Neutral-runtime quality cost (lower is better)")
    passed = sum(c["variants"][s]["metrics"]["is_feasible"] for c in cases for s in strategies)
    ax.set_title(f"Frozen {result['panel']}5 | independently feasible {passed}/{len(cases)*len(strategies)} outputs\nLocal development evidence; not a contest score")
    ax.axhline(1, color="#777777", linewidth=.8, linestyle="--")
    ax.legend(ncols=len(strategies), loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "quality_comparison.png", dpi=160)
    plt.close(fig)
    selected = max(cases, key=lambda c: c["input_summary"]["block_count"])
    shown = [s for s in ("pack", "search", "no_soft") if s in strategies]
    fig, axes = plt.subplots(1, len(shown), figsize=(5 * len(shown), 5.5), layout="constrained", squeeze=False)
    constraints = np.asarray(selected["block_constraints"])
    palette = plt.get_cmap("tab20")
    for ax, strategy in zip(axes[0], shown):
        record = selected["variants"][strategy]
        metric = record["metrics"]
        pos = np.asarray(record["positions"])
        for i, (x, y, w, h) in enumerate(pos):
            group = int(constraints[i, 3])
            face = palette((group - 1) % 20) if group else "#d4dae2"
            fixed, preplaced = bool(constraints[i, 0]), bool(constraints[i, 1])
            ax.add_patch(Rectangle((x, y), w, h, facecolor=face, alpha=.8,
                                   edgecolor="#171e27" if fixed or preplaced else "white",
                                   linewidth=1.1 if fixed or preplaced else .5,
                                   hatch="///" if preplaced else None))
        high = (pos[:, :2] + pos[:, 2:]).max(axis=0)
        ax.set_xlim(0, high[0] * 1.02)
        ax.set_ylim(0, high[1] * 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{strategy} | cost {metric['cost']:.3f}\nHPWL gap {metric['hpwl_gap']:+.1%}, area gap {metric['area_gap']:+.1%}\nBoundary / grouping / MIB: {metric['boundary_violations']} / {metric['grouping_violations']} / {metric['mib_violations']}", fontsize=10)
    fig.suptitle(f"Largest dev panel case: n={selected['input_summary']['block_count']} | colors identify grouping sets\nHatched: preplaced; dark outline: immutable shape; panels use their own coordinate ranges", fontsize=12)
    fig.savefig(out / "layout_comparison.png", dpi=160)
    plt.close(fig)
    print(out / "quality_comparison.png")
    print(out / "layout_comparison.png")


if __name__ == "__main__":
    main()
