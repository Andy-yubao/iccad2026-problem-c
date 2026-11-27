"""E6 saved-gallery adapter; visual style maintained in visualization.layouts."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path


STAGES = (("reference", "Reference layout"), ("raw", "Attention + geometry proposal"),
          ("final", "Frozen E1 postprocessing"))
PANELS = (("train", "训练集"), ("dev", "验证集"), ("local_test", "已访问测试集"))
from visualization.layouts import plot_case as _plot_case


def plot_case(row, destination):
    return _plot_case(row, destination, stages=STAGES)


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
<title>E6 新架构＋几何损失：15例布局图集</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f1f5f9;color:#172033}header,main{max-width:1500px;margin:auto;padding:24px}header{position:sticky;top:0;background:#ffffffed;border-bottom:1px solid #ddd;z-index:2}h1{font-size:24px;margin:0 0 12px}nav a{display:inline-block;margin-right:22px;color:#1d4ed8}article{background:white;padding:18px;border:1px solid #dde3ec;border-radius:10px;margin:20px 0}article img{width:100%;height:auto}h3{font-size:17px}p{line-height:1.6;font-size:14px}section{scroll-margin-top:230px}small{color:#475569}</style>
<header><h1>新架构＋几何损失：参考、原始预测与冻结后处理</h1><nav><a href="#train">训练集</a><a href="#dev">验证集</a><a href="#local_test">已访问测试集</a></nav>
<p>同一例三栏采用相同坐标尺度和模块编号，每栏只展示自身布局。颜色表示 grouping，纹理表示 MIB；L/R/T/B 徽记绿为满足、红为违反。F/P 表示固定形状/预放置，红轮廓表示重叠。</p>
<small>本图集使用冻结的 attention_geometry best：训练、验证、已访问测试各5例。测试仅复用原图集已访问成员，对新模型重新推理并运行冻结后处理，不参与调参。非法原始提议的低线长不能视为可行解质量；Q为中性时间口径，非正式比赛成绩。</small></header><main>'''
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
    if any(count != 5 for count in counts.values()):
        raise ValueError("Expected exactly five preselected cases per panel; do not select cases by plotting results")
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
                "test_policy": "Five already-accessed members, frozen new-model inference and unchanged postprocessing; no tuning.",
                "artifacts": {"index": "index.html", "images": "images/", "preview": None}}
    (run_dir / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
