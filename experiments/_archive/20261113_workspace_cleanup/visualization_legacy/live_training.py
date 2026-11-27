"""Read-only log companion: live local-file Loss chart plus saved image pairs.

Does not import, signal, pause, or modify the trainer or its frozen sources.
Classic local script loading avoids fetch() restrictions on file:// pages.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time


TERMINAL = {"user_stopped", "time_limit", "early_stopped", "epoch_limit", "failed"}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_lines(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # A concurrent writer may not yet have finished the last line.
    return result


def points(run, stage):
    result = []
    for row in read_lines(run / "progress.jsonl"):
        a, b = row.get("train_loss"), row.get("dev_loss")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)):
            continue
        seen = row.get("seen_count", round(row.get("epoch_fraction", 0) * 16000))
        result.append({"steps": row["steps"], "seen": seen, "stage": stage,
            "event": row.get("event", "monitor"), "monitor": row.get("monitor_id"),
            "total": {"train": a, "dev": b},
            **{key: {"train": row.get("train", {}).get(key), "dev": row.get("dev", {}).get(key)}
               for key in ("all_masked_loss", "random_masked_loss")}})
    return result


def latest_gallery(folder):
    candidates = []
    for manifest in folder.glob("*/manifest.json"):
        data = read_json(manifest)
        if data and data.get("cases") and (manifest.parent / "index.html").exists():
            candidates.append((data.get("steps", -1), manifest.stat().st_mtime_ns, manifest.parent.name, data))
    if not candidates:
        return None
    _, _, name, data = max(candidates, key=lambda item: item[:2])
    return {"name": name, "steps": data.get("steps"), "label": data.get("label", ""),
        "cases": [{"member": c["member"], "reference": name + "/" + c["reference"],
                   "generated": name + "/" + c["generated"]} for c in data["cases"]]}


def payload(run, parent, gallery):
    history = points(parent, "FP32 · batch2") if parent else []
    current = points(run, "BF16 · batch4")
    heartbeat = read_json(run / "heartbeat.json", {})
    state = read_json(run / "state.json", {})
    controller = read_json(run.with_name(run.name + "_launch") / "state.json", {})
    if controller.get("status") == "failed":
        state = {**state, "status": "failed"}
    return {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "history": history, "current": current, "heartbeat": heartbeat,
        "state": {k: v for k, v in state.items() if k != "order"},
        "gallery": latest_gallery(gallery), "refresh_seconds": 10,
        "test_evaluated": False, "data_note": "固定train500 / dev500监测；测试集封存。曲线只在完整监测结果写出后增加点。"}


HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>E102 · Loss与图片对比</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172033;font:15px system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:auto;padding:25px}h1{font-size:26px;margin:0 0 8px}h2{font-size:19px;margin:0}
p{line-height:1.65;margin:6px 0}.muted{color:#64748b;font-size:13px}.panel{background:#fff;border:1px solid #dce3eb;border-radius:14px;padding:22px;margin-top:20px}
.status{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.badge{padding:5px 10px;border-radius:20px;background:#e0f2fe;color:#075985;font-size:13px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}.metric{background:#fff;border:1px solid #dce3eb;border-radius:12px;padding:16px}.metric label{display:block;color:#64748b;font-size:13px}.metric strong{display:block;font-size:24px;margin-top:5px;font-variant-numeric:tabular-nums}
.controls{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0;align-items:center}select,button{font:inherit;background:#fff;border:1px solid #cbd5e1;border-radius:7px;padding:7px 10px;color:#172033}button{cursor:pointer}label{font-size:13px}
#plot{width:100%;height:auto;display:block;overflow:visible}.legend{display:flex;gap:22px;font-size:13px}.swatch{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:7px}.train{background:#2563eb}.dev{background:#ea580c}
#hover{min-height:26px;font-size:13px;color:#334155;margin-top:7px;font-variant-numeric:tabular-nums}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.image-card{background:white;border:1px solid #dce3eb;border-radius:12px;padding:18px;margin-top:18px}.image-card h3{font-size:13px;font-weight:500;margin:0 0 12px}.image-card figure{margin:0}.image-card figcaption{color:#64748b;font-size:13px;margin-bottom:8px}.image-card img{display:block;width:100%;border:1px solid #e2e8f0}.section-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;align-items:center}.error{color:#b91c1c}
@media(max-width:640px){.wrap{padding:12px}.metrics{grid-template-columns:1fr 1fr}.panel{padding:14px}.pair{gap:8px}h1{font-size:22px}.image-card{padding:10px}}
</style></head><body><div class="wrap">
<h1>E102 · 训练Loss与图片对比</h1><div class="status"><span class="badge" id="runStatus">读取训练状态…</span><span class="muted" id="updated"></span></div>
<p class="muted">训练继续运行；此页仅只读日志和已保存图片。每10秒更新，无需重新打开入口。</p>
<div class="metrics"><div class="metric"><label>累计参数更新 / 累计样本次</label><strong id="step">—</strong><span class="muted" id="seen">—</span></div>
<div class="metric"><label>最近固定 train500 总Loss</label><strong id="trainValue">—</strong><span class="muted" id="trainStep">—</span></div>
<div class="metric"><label>最近固定 dev500 总Loss</label><strong id="devValue">—</strong><span class="muted">验证集；不是封存测试集</span></div>
<div class="metric"><label>最近批次训练Loss</label><strong id="batchValue">—</strong><span class="muted">单批波动，不与固定500均值混作同一曲线</span></div></div>
<section class="panel"><h2>固定监测 Loss</h2><div class="controls">
<label>显示 <select id="range"><option value="current">本次BF16续训</option><option value="all">全部历史（分精度显示）</option></select></label>
<label>Loss <select id="lossKind"><option value="total">总Loss（选模使用）</option><option value="all_masked_loss">全空白生成Loss</option><option value="random_masked_loss">随机遮罩去噪Loss</option></select></label>
<label>横轴 <select id="xKind"><option value="steps">累计参数更新</option><option value="seen">累计样本次</option></select></label>
<label><input type="checkbox" id="zero"> 纵轴从0开始</label><button id="refresh">立即刷新</button></div>
<div class="legend"><span><i class="swatch train"></i>训练 train500</span><span><i class="swatch dev"></i>验证 dev500</span></div>
<svg id="plot" viewBox="0 0 1080 350" role="img" aria-label="固定train500和dev500损失折线图"></svg><div id="hover">移动到曲线上查看数值。</div>
<p class="muted" id="chartNote"></p><p class="muted">纵轴默认自动缩放；FP32与BF16分段绘制，不跨精度连线。总Loss＝0.5×全空白生成Loss＋0.5×随机遮罩去噪Loss。</p>
</section><section class="panel"><div class="section-head"><h2>同例 Reference / 生成图片</h2><span class="badge" id="imageStep">等待图集</span></div><p class="muted" id="imageNote"></p></section>
<main id="images"></main><p class="muted" id="sourceNote">测试集尚未评价，本页不展示测试Loss。</p></div>
<script src="__ASSET__"></script></body></html>'''


def replace(source, target):
    deadline = time.monotonic() + 2
    while True:
        try:
            source.replace(target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def write(path, content):
    temporary = path.with_suffix(path.suffix + ".dashboard.tmp")
    temporary.write_text(content, encoding="utf-8")
    replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.gallery.mkdir(parents=True, exist_ok=True)
    asset = Path(__file__).with_name("live_training_dashboard.js").resolve()
    relative = os.path.relpath(asset, args.gallery.resolve()).replace("\\", "/")
    page = HTML.replace("__ASSET__", relative)
    last_hash = None
    while True:
        data = payload(args.run, args.parent, args.gallery)
        content = "window.E102_DASHBOARD=" + json.dumps(data, ensure_ascii=True, separators=(",", ":")) + ';window.dispatchEvent(new Event("e102-data"));\n'
        try:
            write(args.gallery / "dashboard_data.js", content)
            if not (args.gallery / "dashboard.html").exists():
                write(args.gallery / "dashboard.html", page)
            # The unmodified trainer may publish a new redirect on each best.
            # Restore just the mutable entrance; saved monitor pages stay intact.
            current = (args.gallery / "index.html").read_text(encoding="utf-8") if (args.gallery / "index.html").exists() else ""
            if current != page:
                write(args.gallery / "index.html", page)
            signature = hashlib.sha256(json.dumps({"step": data["heartbeat"].get("steps"),
                "points": len(data["current"]), "gallery": data["gallery"]["name"] if data["gallery"] else None}).encode()).hexdigest()
            if signature != last_hash:
                last_hash = signature
                state = {"status": "finished" if data["state"].get("status") in TERMINAL else "watching",
                    "pid": os.getpid(), "updated_utc": data["generated_at"],
                    "training_step": data["heartbeat"].get("steps"), "current_monitor_points": len(data["current"]),
                    "trainer_modified": False, "test_read": False}
                write(args.gallery / "dashboard_status.json", json.dumps(state, indent=2) + "\n")
        except (OSError, ValueError) as error:
            print("Dashboard refresh delayed: " + str(error), flush=True)
        if args.once or data["state"].get("status") in TERMINAL:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
