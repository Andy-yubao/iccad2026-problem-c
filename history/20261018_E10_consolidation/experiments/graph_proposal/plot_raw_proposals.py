"""Plot stage-A saved raw proposals, without datasets, checkpoints or inference.

Usage: python -B -m experiments.graph_proposal.plot_raw_proposals --run-dir RUN
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SERIES = (("train", "deterministic", "Train / model", "#2563eb"),
          ("dev", "deterministic", "Dev / model", "#ea580c"),
          ("train", "heuristic", "Train / heuristic", "#0891b2"),
          ("dev", "heuristic", "Dev / heuristic", "#a855f7"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_cases(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid complete JSON record at {path}:{index}") from error
    if not rows:
        raise ValueError(f"No cases in {path}")
    return rows


def values(cases, panel, variant, metric):
    result = []
    for row in cases:
        if row.get("panel") == panel:
            value = row.get("variants", {}).get(variant, {}).get("diagnostics", {}).get(metric)
            if finite(value):
                result.append(value)
    return sorted(result)


def ecdf(ax, numbers, *, color, label=None, linestyle="-", scale=1):
    if numbers:
        ax.step([numbers[0] * scale] + [v * scale for v in numbers],
                [0] + [(i + 1) / len(numbers) for i in range(len(numbers))],
                where="post", color=color, linewidth=1.6, linestyle=linestyle,
                label=label)


def plot_diagnostics(cases, result, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(left=.065, right=.98, bottom=.15, top=.82, hspace=.4, wspace=.28)
    specs = [
        ("pair_relative_local_mean", "Pair-relative center error", "Mean error / local module size", 1),
        ("center_error_local_mean", "Absolute center error", "Mean error / local module size", 1),
        ("pair_overlap_area_fraction", "Pairwise overlap area", "Sum of pair intersections / total block area (%)", 100),
        ("translation_radius_lb", "Necessary translation-radius lower bound", "Maximum per-block motion / local size (%)", 100),
        ("direction_any_separation_retained_fraction", "Reference direction retention", "Retained reference pair relations (%)", 100),
        ("shape_log_ratio_rmse_free", "Free-block aspect-ratio error", "RMSE of log(width / height)", 1),
    ]
    counts = {}
    for ax, (metric, title, xlabel, scale) in zip(axes.flat, specs):
        for panel, variant, label, color in SERIES:
            numbers = values(cases, panel, variant, metric)
            counts.setdefault(label, {})[metric] = len(numbers)
            ecdf(ax, numbers, color=color, scale=scale)
            if metric == "direction_any_separation_retained_fraction":
                ecdf(ax, values(cases, panel, variant,
                                "direction_any_center_order_retained_fraction"),
                     color=color, linestyle="--", scale=scale)
        ax.set(title=title, xlabel=xlabel, ylabel="Fraction of layouts", ylim=(0, 1.02))
        ax.grid(alpha=.2)
        if metric in {"pair_relative_local_mean", "center_error_local_mean", "pair_overlap_area_fraction"}:
            ax.set_xscale("symlog", linthresh=.01 if scale == 1 else .1)
            ax.set_xlim(left=0)
        if metric == "translation_radius_lb":
            ax.axvline(5, color="#475569", linestyle=":", linewidth=1)
            ax.axvline(10, color="#475569", linestyle="--", linewidth=1)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(left=0)
            ax.text(.03, .97, "5% dotted; 10% dashed\nFixed predicted shapes only;\nnecessary, not sufficient", transform=ax.transAxes,
                    va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        if metric == "direction_any_separation_retained_fraction":
            ax.set_xlim(-2, 102)
            ax.text(.03, .97, "Solid: any reference separation axis\nDashed: any reference center-order axis",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    handles = [Line2D([0], [0], color=color, lw=2, label=label)
               for _, _, label, color in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .89), ncol=4, frameon=False)
    ntrain = sum(row.get("panel") == "train" for row in cases)
    ndev = sum(row.get("panel") == "dev" for row in cases)
    fig.suptitle(f"Raw decoded proposals, no postprocessing | train n={ntrain}, dev n={ndev}\n"
                 f"Run status: {result.get('status', 'unknown')}", fontsize=15, y=.98)
    fig.text(.065, .055,
             "Each layout has equal weight. Undefined / absent metrics are omitted; sample counts are saved in figure_selection.json.\n"
             "Local-size units follow the saved diagnostics. Pairwise overlap sums can exceed 100% (not union area).\n"
             "These are proposal diagnostics: low HPWL or small bounding boxes in an illegal layout are not feasible solution quality.",
             fontsize=9, color="#475569")
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)
    return counts


def worker_id(row):
    value = row.get("input_summary", {}).get("worker")
    if isinstance(value, int):
        return value
    match = re.search(r"worker[_ /\\-]?(\d+)", json.dumps([row.get("source"), row.get("case_id")]))
    return int(match.group(1)) if match else None


def positions(row, variant):
    result = row.get("variants", {}).get(variant, {}).get("positions", [])
    if not result or any(len(rect) != 4 or not all(finite(v) for v in rect) for rect in result):
        raise ValueError(f"Invalid {variant} XYWH coordinates in {row.get('case_id')}")
    return result


def overlap_regions(rectangles, tolerance):
    regions, blocks = [], set()
    for i, (x, y, w, h) in enumerate(rectangles):
        for j in range(i + 1, len(rectangles)):
            a, b, c, d = rectangles[j]
            left, bottom = max(x, a), max(y, b)
            width, height = min(x + w, a + c) - left, min(y + h, b + d) - bottom
            if width > tolerance and height > tolerance:
                blocks.update((i, j))
                regions.append((left, bottom, width, height))
    return regions, blocks


def plot_overlays(selected, path, heading):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D

    if not selected:
        return
    single = len(selected) == 1
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 6 if single else 4.6 * len(selected)), squeeze=False)
    fig.subplots_adjust(top=.76 if single else .93, bottom=.20 if single else .075,
                        left=.055, right=.985, hspace=.37, wspace=.15)
    palette = plt.get_cmap("tab20")
    for rowindex, row in enumerate(selected):
        reference, proposal = positions(row, "reference"), positions(row, "deterministic")
        if len(reference) != len(proposal):
            raise ValueError(f"Block count differs in {row.get('case_id')}")
        all_rects = reference + proposal
        left = min(rect[0] for rect in all_rects)
        bottom = min(rect[1] for rect in all_rects)
        right = max(rect[0] + rect[2] for rect in all_rects)
        top = max(rect[1] + rect[3] for rect in all_rects)
        span = max(right - left, top - bottom, 1e-12)
        overlaps, overlap_blocks = overlap_regions(proposal, span * 1e-9)
        constraints = row.get("constraints", [])
        diagnostic = row["variants"]["deterministic"].get("diagnostics", {})
        bound = diagnostic.get("translation_radius_lb")
        bound_text = f"{bound * 100:.1f}%" if finite(bound) else "undefined"
        description = f"worker {worker_id(row)} | n={len(reference)} | translation LB={bound_text}"
        for column, ax in enumerate(axes[rowindex]):
            if column == 2:
                for x, y, width, height in reference:
                    ax.add_patch(Rectangle((x, y), width, height, fill=False,
                                           edgecolor="#64748b", linestyle="--", linewidth=.8, alpha=.7))
            rectangles = reference if column == 0 else proposal
            for block, (x, y, width, height) in enumerate(rectangles):
                flags = constraints[block] if block < len(constraints) else []
                preplaced = len(flags) > 1 and bool(flags[1])
                fixed = bool(flags[0]) if flags else False
                color = palette(block % 20)
                edge = "#dc2626" if column > 0 and block in overlap_blocks else "#334155"
                ax.add_patch(Rectangle((x, y), width, height, facecolor=color,
                                       alpha=.24 if column == 2 else .55,
                                       edgecolor=edge, linewidth=1.15 if edge == "#dc2626" else .55,
                                       hatch="//" if fixed else None))
                ax.text(x + width / 2, y + height / 2, str(block), fontsize=4.6,
                        ha="center", va="center", clip_on=True, color="#0f172a")
                if preplaced:
                    ax.plot(x + width / 2, y + height / 2, "ks", ms=2)
            if column > 0:
                for x, y, width, height in overlaps:
                    ax.add_patch(Rectangle((x, y), width, height,
                                           facecolor="#dc2626", alpha=.22, edgecolor="none"))
            if column == 2:
                for ref, pred in zip(reference, proposal):
                    start = (ref[0] + ref[2] / 2, ref[1] + ref[3] / 2)
                    end = (pred[0] + pred[2] / 2, pred[1] + pred[3] / 2)
                    ax.annotate("", xy=end, xytext=start,
                                arrowprops={"arrowstyle": "->", "lw": .65, "color": "#0f172a", "alpha": .6})
            ax.set_xlim(left - .04 * span, right + .04 * span)
            ax.set_ylim(bottom - .04 * span, top + .04 * span)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7)
            title = ("Reference", "Frozen model (raw decoded)", "Reference + model; displacement arrows")[column]
            ax.set_title(f"{title}\n{description}", fontsize=9)
            ax.set_xlabel("x", fontsize=8)
            if column == 0:
                ax.set_ylabel("y", fontsize=8)
    fig.suptitle(f"{heading}\nRaw decoded layouts; no postprocessing; same coordinates and scale within each row", fontsize=14, y=.985)
    handles = [Patch(facecolor="white", edgecolor="#334155", hatch="//", label="Fixed shape"),
               Line2D([], [], color="black", marker="s", linestyle="none", ms=3, label="Preplaced"),
               Patch(facecolor="#dc2626", alpha=.3, label="Positive-area model intersections"),
               Line2D([], [], color="#64748b", linestyle="--", label="Reference in overlay")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .06 if single else .03), ncol=4, fontsize=9, frameon=False)
    fig.text(.055, .012, "Block IDs are identical across panels. Red intersection plotting threshold: 1e-9 of row span per axis. Low illegal-layout HPWL is not feasible performance.", fontsize=8, color="#475569")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def case_identity(row):
    return {"panel": row.get("panel"), "case_id": row.get("case_id"),
            "source": row.get("source"), "worker": worker_id(row),
            "translation_radius_lb": row.get("variants", {}).get("deterministic", {}).get("diagnostics", {}).get("translation_radius_lb")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cases_path = run_dir / "cases.jsonl"
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cases = load_cases(cases_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    counts = plot_diagnostics(cases, result, run_dir / "proposal_diagnostics.png")
    dev = sorted((row for row in cases if row.get("panel") == "dev"), key=lambda row: str(row.get("case_id")))
    fixed_workers = (0, 24, 49, 74, 99)
    fixed = [next((row for row in dev if worker_id(row) == worker), None) for worker in fixed_workers]
    fixed = [row for row in fixed if row is not None]
    plot_overlays(fixed, run_dir / "proposal_overlays.png", "Prespecified dev workers: 0, 24, 49, 74, 99")
    ranked = [row for row in dev if finite(case_identity(row)["translation_radius_lb"])]
    ranked.sort(key=lambda row: (case_identity(row)["translation_radius_lb"], str(row.get("case_id"))))
    quantiles = [ranked[index] for index in (0, (len(ranked) - 1) // 2, len(ranked) - 1)] if ranked else []
    plot_overlays(quantiles, run_dir / "proposal_overlays_by_translation_lb.png",
                  "Dev translation lower-bound ranks: smallest / lower median / largest")
    selection = {
        "source": "Saved cases.jsonl only; no dataset or checkpoint reads; no inference.",
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "run_status": result.get("status"),
        "fixed_rule": "Dev workers 0,24,49,74,99; one case per worker. Lexical case_id tie-break if duplicates.",
        "fixed": [case_identity(row) for row in fixed],
        "missing_fixed_workers": [worker for worker in fixed_workers if not any(worker_id(row) == worker for row in fixed)],
        "rank_rule": "Dev only, finite deterministic translation_radius_lb ascending, case_id tie-break; ranks 0, floor((n-1)/2), n-1. These are diagnostic extrema, not representative estimates.",
        "ranked_dev_count": len(ranked), "translation_lb_rank_examples": [case_identity(row) for row in quantiles],
        "ecdf_available_layout_counts": counts,
        "plot_overlap_tolerance": "Both intersection dimensions > 1e-9 * combined reference/model span; visualization only, not hard feasibility.",
        "caveats": ["Raw decoded proposals, no postprocessing.", "Translation lower bound is necessary only for fixed predicted shapes; it is not a repair solver.", "Undefined metrics omitted from their own ECDF; see per-metric counts.", "Illegal-layout HPWL/bbox cannot be interpreted as feasible solution performance."],
    }
    (run_dir / "figure_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "cases": len(cases), "fixed_examples": len(fixed), "rank_examples": len(quantiles)}))


if __name__ == "__main__":
    main()
