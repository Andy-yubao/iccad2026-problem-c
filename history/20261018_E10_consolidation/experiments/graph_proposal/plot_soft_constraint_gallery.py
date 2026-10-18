"""Render saved reference/raw/final layouts with their soft-constraint status.

Reads config.json and cases.jsonl only. No tensors, checkpoints or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path


STAGES = (("reference", "Reference layout"), ("raw", "Raw model proposal"),
          ("final", "Current postprocessing"))
PANELS = (("train", "训练集"), ("dev", "验证集"), ("local_test", "已访问测试集"))
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


def read_rows(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed cases.jsonl at line {index}") from error
    return rows


def resolve_positions(cases):
    """Resolve saved experiment geometry in memory, retaining canonical JSON files."""
    root = Path(__file__).resolve().parents[2]
    experiments_root = (root / "experiments").resolve()
    cache = {}
    for row in cases:
        for stage, _ in STAGES:
            variant = row["variants"][stage]
            if "positions" in variant:
                continue
            reference = variant.get("positions_source")
            if not reference:
                raise ValueError(f"Missing positions or positions_source: {row['case_id']} / {stage}")
            path = (root / reference["file"]).resolve()
            if not path.is_relative_to(experiments_root) or path.suffix.lower() != ".jsonl":
                raise ValueError(f"Geometry references must be saved experiment JSONL: {path}")
            if path not in cache:
                records = read_rows(path)
                cache[path] = {item["case_id"]: item for item in records}
                if len(cache[path]) != len(records):
                    raise ValueError(f"Duplicate case_id in geometry source: {path}")
            source_row = cache[path][reference["case_id"]]
            variant["positions"] = source_row["variants"][reference["variant"]]["positions"]


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


def validate_case(row):
    constraints = row["constraints"]
    if any(len(flags) != 5 for flags in constraints):
        raise ValueError(f"Invalid constraint shape for {row['case_id']}")
    for stage, _ in STAGES:
        rects = row["variants"][stage]["positions"]
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


def plot_case(row, destination):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D

    validate_case(row)
    constraints = row["constraints"]
    groups, mibs = group_ids(constraints, 3), group_ids(constraints, 2)
    cmap = plt.get_cmap("Set3")
    colors = {group: cmap(i % 12) for i, group in enumerate(groups)}
    patterns = {group: HATCHES[i % len(HATCHES)] for i, group in enumerate(mibs)}
    rectangles = [rect for stage, _ in STAGES for rect in row["variants"][stage]["positions"]]
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
    legend_rows = math.ceil((len(groups) + len(mibs) + 7) / 7)
    legend_height = max(.065, legend_rows * .026 + .025)
    table_bottom = .065 + legend_height + .025
    table_top = table_bottom + table_height
    axes_bottom = table_top + .135
    fig = plt.figure(figsize=(19.5, 11.7))
    axes = [fig.add_axes((.035 + index * .323, axes_bottom, .30, .88 - axes_bottom)) for index in range(3)]
    for index, ((stage, title), ax) in enumerate(zip(STAGES, axes)):
        variant = row["variants"][stage]
        rects = variant["positions"]
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
        left = .035 + index * .323
        fig.text(left, axes_bottom - .04, status_text(variant), fontsize=10, linespacing=1.7, va="top")
        fig.text(left, table_top, detail_text(variant), fontsize=8.5,
                 family="monospace", linespacing=1.5, va="top", color="#334155")
    fig.suptitle(f"{row['panel']} | {row['case_id']} | {len(constraints)} blocks", fontsize=15, y=.975)
    fig.text(.5, .935, "Same module IDs and coordinate scale across columns; each column shows only its own layout.",
             ha="center", fontsize=10, color="#475569")
    legend = [Patch(facecolor=colors[group], edgecolor="#475569", label=f"Grouping G{group}") for group in groups]
    legend += [Patch(facecolor="white", edgecolor="#475569", hatch=patterns[group], label=f"MIB M{group}") for group in mibs]
    legend += [Patch(facecolor="#e5e7eb", edgecolor="#475569", label="No grouping"),
               Patch(facecolor="#15803d", label="L/R/T/B satisfied"),
               Patch(facecolor="#b91c1c", label="L/R/T/B violated"),
               Patch(facecolor="#a16207", label="Boundary status missing"),
               Patch(facecolor="none", edgecolor="#dc2626", linewidth=1.5, label="Overlapping block"),
               Line2D([], [], color="#64748b", ls="--", label="Own layout bbox"),
               Patch(facecolor="white", label="F: fixed shape; P: preplaced")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .065), ncol=7,
               fontsize=8, frameon=True, edgecolor="#e2e8f0", title="Constraint identities are unchanged across all three stages", title_fontsize=9)
    fig.text(.035, .024,
             "B/G/M = official boundary / grouping / MIB violation counts. Q uses neutral runtime; illegal proposals are not feasible-quality results.\n"
             "Boundary badges use saved status against each layout's own bbox. Group components=1 and MIB distinct shapes=1 indicate consistency. "
             "Overlap outlines are visual only (1e-9 span tolerance).", fontsize=8, color="#475569")
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    return {"axis_limits": list(limits), "bounding_policy": "union of all three stages plus 5.5% maximum-span margin; no blocks cropped",
            "grouping_colors": {str(group): list(colors[group]) for group in groups},
            "mib_hatches": {str(group): patterns[group] for group in mibs},
            "plot_overlap_axis_tolerance": span * 1e-9}


def write_index(run_dir, records):
    sections = []
    for panel, label in PANELS:
        cards = []
        for row in records:
            if row["panel"] != panel:
                continue
            image_path = html.escape(row["image"], quote=True)
            caption = html.escape(f"{row['sequence']:02d} · {row['case_id']}")
            cards.append(f'<article><h3>{caption}</h3><a href="{image_path}" target="_blank"><img src="{image_path}" alt="{caption}" loading="lazy"></a>'
                         '<p>点击图片查看大图。三栏依次为参考、原始模型、当前后处理。</p></article>')
        sections.append(f'<section id="{panel}"><h2>{label}（{len(cards)}例）</h2>{"".join(cards)}</section>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>布局软约束图集</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f1f5f9;color:#172033}header,main{max-width:1500px;margin:auto;padding:24px}header{position:sticky;top:0;background:#ffffffed;border-bottom:1px solid #ddd;z-index:2}h1{font-size:24px;margin:0 0 12px}nav a{display:inline-block;margin-right:22px;color:#1d4ed8}article{background:white;padding:18px;border:1px solid #dde3ec;border-radius:10px;margin:20px 0}article img{width:100%;height:auto}h3{font-size:17px}p{line-height:1.6;font-size:14px}section{scroll-margin-top:230px}small{color:#475569}</style>
<header><h1>参考布局、原始提议与当前后处理</h1><nav><a href="#train">训练集</a><a href="#dev">验证集</a><a href="#local_test">已访问测试集</a></nav>
<p>同一例三栏采用相同坐标尺度和模块编号，每栏只展示自身布局。颜色表示 grouping，纹理表示 MIB；L/R/T/B 徽记绿为满足、红为违反。F/P 表示固定形状/预放置，红轮廓表示重叠。</p>
<small>测试图仅复用已访问样例，用于描述已有结果，不参与模型或后处理调参。非法原始提议的低线长不能视为可行解质量；Q为中性时间口径，非正式比赛成绩。</small></header><main>'''
    document += "".join(sections) + "</main></html>"
    (run_dir / "index.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config_path, cases_path = run_dir / "config.json", run_dir / "cases.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = read_rows(cases_path)
    if not cases:
        raise ValueError("No saved cases to plot")
    unknown = {row.get("panel") for row in cases} - {panel for panel, _ in PANELS}
    if unknown:
        raise ValueError(f"Unknown panels: {unknown}")
    counts = {panel: sum(row["panel"] == panel for row in cases) for panel, _ in PANELS}
    if any(count > 10 for count in counts.values()):
        raise ValueError("Expected at most ten preselected cases per panel; do not select cases by plotting results")
    resolve_positions(cases)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "hatch.linewidth": .45, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    images = run_dir / "images"
    images.mkdir(exist_ok=True)
    counters = {panel: 0 for panel, _ in PANELS}
    records = []
    for row in cases:
        panel = row["panel"]
        counters[panel] += 1
        relative = f"images/{panel}_{counters[panel]:02d}.png"
        style = plot_case(row, run_dir / relative)
        records.append({"panel": panel, "sequence": counters[panel], "case_id": row["case_id"],
                        "source": row.get("source"), "image": relative, "render": style})
        print(json.dumps({"image": relative, "case_id": row["case_id"]}), flush=True)
    write_index(run_dir, records)
    manifest = {"input_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "input_cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                "config_status": config.get("status"), "counts": counters, "figures": records,
                "selection_rule": "All saved cases in JSONL order; no score-based filtering or fresh data access.",
                "styles": {"columns": [stage for stage, _ in STAGES], "grouping": "same G ID, same Set3 fill; no group grey",
                           "mib": "same M ID, same hatch; IDs printed on blocks", "boundary": "L/R/T/B on corresponding block edge; green satisfied, red violated, ochre missing status",
                           "overlap": "red block outlines only; no red intersection fills", "fixed_preplaced": "F and P labels",
                           "references": "No cross-stage arrows, reference overlays, or unrelated net lines",
                           "bounds": "Shared union coordinate bounds per case; own bbox shown dashed; 5.5% margin; no geometric cropping"},
                "test_policy": "Only already-accessed test examples; illustrations do not participate in tuning.",
                "artifacts": {"index": "index.html", "images": "images/", "preview": None}}
    (run_dir / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
