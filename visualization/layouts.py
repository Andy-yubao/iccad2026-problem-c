"""Saved-layout renderer extracted from E6 plot_gallery.py (2026-10-17).
No dataset access, inference, repair or official scoring. Original visual style retained.
"""
from __future__ import annotations
import math
STAGES = (("reference", "Reference layout"), ("raw", "Attention + geometry proposal"),
          ("final", "Frozen postprocessing"))
HATCHES = ("///", "\\\\\\", "xxx", "...", "+++", "ooo", "***", "|||", "---", "OO")
SIDES = ((1, "L"), (2, "R"), (4, "T"), (8, "B"))


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def display(value, precision=3):
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if numeric(value):
        return f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return "n/a" if value is None else str(value)


def group_ids(constraints, column):
    return sorted({int(row[column]) for row in constraints if int(row[column]) > 0})


def overlap_blocks(rects, tolerance):
    blocks = set()
    for i, (x, y, w, h) in enumerate(rects):
        for j in range(i + 1, len(rects)):
            a, b, c, d = rects[j]
            if min(x + w, a + c) - max(x, a) > tolerance and min(y + h, b + d) - max(y, b) > tolerance:
                blocks.update((i, j))
    return blocks


def boundary_status(variant):
    details = variant.get("soft_details", {})
    return {int(row["block"]): {int(bit) for bit in row.get("failed_bits", [])}
            for row in details.get("boundary", [])}


def validate_case(row, stages):
    constraints = row["constraints"]
    if any(len(flags) != 5 for flags in constraints):
        raise ValueError(f"Invalid constraint shape for {row['case_id']}")
    for stage, _ in stages:
        rects = row["variants"][stage]["positions"]
        if rects is None:
            continue
        if len(rects) != len(constraints) or any(len(rect) != 4 or not all(numeric(v) for v in rect)
                                                or rect[2] <= 0 or rect[3] <= 0 for rect in rects):
            raise ValueError(f"Invalid {stage} XYWH in {row['case_id']}")


def status_text(variant):
    metrics = variant.get("metrics", {})
    return (f"Hard legal: {display(metrics.get('is_feasible'))}    Q: {display(metrics.get('cost'))}\n"
            f"Violations  B / G / M: {display(metrics.get('boundary_violations'))} / "
            f"{display(metrics.get('grouping_violations'))} / {display(metrics.get('mib_violations'))}")


def detail_text(variant):
    details = variant.get("soft_details", {})
    lines = []
    for key, prefix, suffix in (("group_components", "G", "contact components"),
                                ("mib_distinct_shapes", "M", "distinct shapes")):
        items = details.get(key, {})
        values = sorted(items.items(), key=lambda item: int(item[0]))
        chunks = [values[i:i + 5] for i in range(0, len(values), 5)]
        lines.append(suffix + ":")
        lines.extend("  " + ",  ".join(f"{prefix}{group}={value}" for group, value in chunk) for chunk in chunks)
        if not chunks:
            lines.append("  none / unavailable")
    return "\n".join(lines)


def draw_boundary_badges(ax, rect, code, failures):
    x, y, w, h = rect
    locations = {1: ((x, y + h * .5), (4, 0), "left", "center"),
                 2: ((x + w, y + h * .5), (-4, 0), "right", "center"),
                 4: ((x + w * .5, y + h), (0, -4), "center", "top"),
                 8: ((x + w * .5, y), (0, 4), "center", "bottom")}
    for bit, letter in SIDES:
        if not code & bit:
            continue
        color = "#a16207" if failures is None else ("#b91c1c" if bit in failures else "#15803d")
        location, offset, horizontal, vertical = locations[bit]
        ax.annotate(letter, xy=location, xytext=offset, textcoords="offset points",
                    ha=horizontal, va=vertical, fontsize=6.0, weight="bold", color="white",
                    bbox={"boxstyle": "round,pad=.12", "facecolor": color, "edgecolor": "white", "linewidth": .35},
                    zorder=8, clip_on=True)


def plot_case(row, destination, *, stages=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D

    stages = stages or STAGES
    columns = len(stages)
    if not 1 <= columns <= 3:
        raise ValueError("Layout comparison supports one to three saved stages")
    validate_case(row, stages)
    constraints = row["constraints"]
    groups, mibs = group_ids(constraints, 3), group_ids(constraints, 2)
    cmap = plt.get_cmap("Set3")
    colors = {group: cmap(i % 12) for i, group in enumerate(groups)}
    patterns = {group: HATCHES[i % len(HATCHES)] for i, group in enumerate(mibs)}
    rectangles = [rect for stage, _ in stages for rect in (row["variants"][stage]["positions"] or [])]
    xmin = min(rect[0] for rect in rectangles)
    ymin = min(rect[1] for rect in rectangles)
    xmax = max(rect[0] + rect[2] for rect in rectangles)
    ymax = max(rect[1] + rect[3] for rect in rectangles)
    span = max(xmax - xmin, ymax - ymin, 1e-12)
    margin = span * .055
    limits = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)
    # Reserve separate bands for stage statistics, group tables and shared legend.
    max_groups = max(math.ceil(len(groups) / 5), 1) + max(math.ceil(len(mibs) / 5), 1)
    table_height = max(.072, .018 * (max_groups + 2))
    legend_columns = 7 if columns == 3 else (5 if columns == 2 else 3)
    legend_rows = math.ceil((len(groups) + len(mibs) + 7) / legend_columns)
    legend_height = max(.065, legend_rows * .026 + .025)
    table_bottom = .065 + legend_height + .025
    table_top = table_bottom + table_height
    axes_bottom = table_top + .135
    plt.rcParams.update({"font.size": 10, "hatch.linewidth": .45, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(19.5 if columns == 3 else (13.0 if columns == 2 else 8.8), 11.7))
    lefts = [.035 + index * .323 for index in range(3)] if columns == 3 else [.05 + index * .92 / columns for index in range(columns)]
    width = .30 if columns == 3 else .92 / columns - .025
    axes = [fig.add_axes((left, axes_bottom, width, .88 - axes_bottom)) for left in lefts]
    for index, ((stage, title), ax) in enumerate(zip(stages, axes)):
        variant = row["variants"][stage]
        rects = variant["positions"]
        if rects is None:
            ax.text(.5, .5, variant.get("unavailable", "No returned layout"), transform=ax.transAxes,
                    ha="center", va="center", wrap=True)
            ax.set(xlim=limits[:2], ylim=limits[2:], title=title)
            ax.set_aspect("equal", adjustable="box")
            continue
        overlap = overlap_blocks(rects, span * 1e-9)
        failed = boundary_status(variant)
        ownxmin = min(rect[0] for rect in rects)
        ownymin = min(rect[1] for rect in rects)
        ownxmax = max(rect[0] + rect[2] for rect in rects)
        ownymax = max(rect[1] + rect[3] for rect in rects)
        ax.add_patch(Rectangle((ownxmin, ownymin), ownxmax - ownxmin, ownymax - ownymin,
                               fill=False, edgecolor="#64748b", linewidth=.8, linestyle="--", zorder=1))
        for block, (rect, flags) in enumerate(zip(rects, constraints)):
            x, y, width, height = rect
            fixed, preplaced, mib, group, boundary = map(int, flags)
            ax.add_patch(Rectangle((x, y), width, height,
                                   facecolor=colors.get(group, "#e5e7eb"), edgecolor="#475569",
                                   linewidth=.55, hatch=patterns.get(mib), alpha=.78, zorder=2))
            if block in overlap:
                ax.add_patch(Rectangle((x, y), width, height, fill=False,
                                       edgecolor="#dc2626", linewidth=1.5, zorder=4))
            labels = [str(block)]
            if group > 0 or mib > 0:
                labels.append(" ".join(([f"G{group}"] if group > 0 else []) + ([f"M{mib}"] if mib > 0 else [])))
            if fixed or preplaced:
                labels.append(" ".join((["F"] if fixed else []) + (["P"] if preplaced else [])))
            ax.text(x + width / 2, y + height / 2, "\n".join(labels), ha="center", va="center",
                    fontsize=5.5, color="#111827", linespacing=1.0, zorder=6, clip_on=True,
                    bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none", "pad": .35})
            draw_boundary_badges(ax, rect, boundary, failed.get(block))
        ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="x", ylabel="y")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=13, pad=12)
        left = lefts[index]
        fig.text(left, axes_bottom - .04, status_text(variant), fontsize=10, linespacing=1.7, va="top")
        fig.text(left, table_top, detail_text(variant), fontsize=8.5,
                 family="monospace", linespacing=1.5, va="top", color="#334155")
    fig.suptitle(f"{row['panel']} | {row['case_id']} | {len(constraints)} blocks", fontsize=15 if columns == 3 else 11, y=.975)
    caption = "Same module IDs and coordinate scale across columns; each column shows only its own layout." if columns > 1 else "Saved E14 output; module IDs and constraints follow the source task."
    fig.text(.5, .935, caption, ha="center", fontsize=10 if columns > 1 else 9, color="#475569")
    legend = [Patch(facecolor=colors[group], edgecolor="#475569", label=f"Grouping G{group}") for group in groups]
    legend += [Patch(facecolor="white", edgecolor="#475569", hatch=patterns[group], label=f"MIB M{group}") for group in mibs]
    legend += [Patch(facecolor="#e5e7eb", edgecolor="#475569", label="No grouping"),
               Patch(facecolor="#15803d", label="L/R/T/B satisfied"),
               Patch(facecolor="#b91c1c", label="L/R/T/B violated"),
               Patch(facecolor="#a16207", label="Boundary status missing"),
               Patch(facecolor="none", edgecolor="#dc2626", linewidth=1.5, label="Overlapping block"),
               Line2D([], [], color="#64748b", ls="--", label="Own layout bbox"),
               Patch(facecolor="white", label="F: fixed shape; P: preplaced")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .065), ncol=legend_columns,
               fontsize=8, frameon=True, edgecolor="#e2e8f0", title="Constraint identities are unchanged across all three stages" if columns == 3 else "Constraint key for the saved layout", title_fontsize=9)
    fig.text(.035, .024,
             ("B/G/M = official boundary / grouping / MIB violation counts. Q uses neutral runtime; illegal proposals are not feasible-quality results.\n"
              "Boundary badges show status against each layout's own bbox. Group components=1 and MIB distinct shapes=1 indicate consistency. "
              "Overlap outlines are visual only (1e-9 span tolerance).") if columns == 3 else
             "B/G/M = official boundary / grouping / MIB violations; Q uses neutral runtime.\n"
             "Badges refer to this layout's bbox. Red overlap outlines use a visual-only tolerance.",
             fontsize=8, color="#475569")
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    return {"axis_limits": list(limits), "bounding_policy": "union of all three stages plus 5.5% maximum-span margin; no blocks cropped",
            "axes_bounds": [list(ax.get_position().bounds) for ax in axes],
            "grouping_colors": {str(group): list(colors[group]) for group in groups},
            "mib_hatches": {str(group): patterns[group] for group in mibs},
            "plot_overlap_axis_tolerance": span * 1e-9}



def geometry_details(positions,constraints):
    """Visual geometry diagnostics copied from E6; no official score computation."""
    import numpy as np
    from shapely.geometry import box
    from shapely.ops import unary_union
    p=np.asarray(positions,dtype=float);c=np.asarray(constraints,dtype=int)
    lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0)
    dist=np.column_stack((p[:,0]-lo[0],hi[0]-p[:,0]-p[:,2],hi[1]-p[:,1]-p[:,3],p[:,1]-lo[1]))
    boundary=[]
    for i in np.flatnonzero(c[:,4]):
        bits=[bit for bit in (1,2,4,8) if c[i,4]&bit]
        failed=[bit for bit in bits if abs(dist[i,{1:0,2:1,4:2,8:3}[bit]])>=1e-6]
        boundary.append({'block':int(i),'code':int(c[i,4]),'failed_bits':failed})
    groups={};mibs={}
    for g in np.unique(c[:,3]):
        if g<=0:continue
        union=unary_union([box(x,y,x+w,y+h) for x,y,w,h in p[c[:,3]==g]])
        groups[str(g)]=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
    for g in np.unique(c[:,2]):
        if g<=0:continue
        mibs[str(g)]=len({(round(float(w),4),round(float(h),4)) for w,h in p[c[:,2]==g,2:]})
    return {'boundary':boundary,'group_components':groups,'mib_distinct_shapes':mibs}


def plot_evaluation_case(row, name, destination):
    """Adapt the saved E7 evaluation schema, without computing official scores."""
    value = row["variants"][name]
    final = value.get("final", {})
    variants = {
        "reference": {"positions": row["reference_positions"]},
        "raw": {"positions": value.get("raw_positions")},
        "final": {"positions": final.get("positions"), "metrics": final.get("metrics", {})},
    }
    for stage, variant in variants.items():
        if variant["positions"] is None:
            variant["unavailable"] = "No returned layout\n" + value.get("error", {}).get("message", "")
        else:
            variant["soft_details"] = geometry_details(variant["positions"], row["constraints"])
        # No fabricated official metrics when the saved row does not contain them.
    adapted = {"case_id": row["case_id"], "panel": row.get("panel", row["case_id"].split(":")[0]),
               "constraints": row["constraints"], "variants": variants}
    seconds = value.get("postprocess_seconds")
    title = "Current postprocessing" + (f" ({seconds:.3f}s)" if seconds is not None else "")
    return plot_case(adapted, destination, stages=(("reference", "Reference layout"),
        ("raw", name + " proposal"), ("final", title)))
