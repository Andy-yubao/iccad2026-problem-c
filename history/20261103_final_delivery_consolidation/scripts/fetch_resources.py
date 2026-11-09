"""Fetch non-dataset official resources using Python's standard library.

Run from any directory: python scripts/fetch_resources.py
Existing files are verified by content type and included in the manifest.
This script never downloads FloorSet datasets.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE = "https://drive.google.com/file/d/{}/view?usp=sharing"
RESOURCES = [
    ("references/official/C_20260522.pdf", "1gwb6eQo50tS9jITlo5b6Edg1oWYiHV5C", "pdf"),
    ("references/official/C_QA_20260827.pdf", "1aWoA-xS7mGR_ycrD0yXgN3dvPuU7kDRM", "pdf"),
    ("references/official/C_Submission_Guidelines_20260616.pdf", "1OiKhswOKrlLNStzUHt1IDN6Ej2kXXJNW", "pdf"),
    ("third_party/official/op_wrapper.py", "1m9_ZDtZISr78d5phee5QzsCDd_VkQHaH", "python"),
    ("references/official/results/C_Alpha_Top5_20260720.csv", "1hwEo9sdY3OsBE3KDHEzCHmew5wRCXhXJ", "text"),
    ("references/official/results/C_Median_Runtime_Alpha.csv", "1bo0VBOLZ0yrW8OIKv3vT6gaf3bIb_aZD", "text"),
    ("references/official/results/C_beta_evaluation_report_hidden_final.txt", "19BvzEwQ_6yZMEiQ7SaUfJht5MqP16PH6", "text"),
    ("references/official/results/C_beta_leaderboard_20260823.csv", "1Wj7Du2lvpm9_fPNpIQqgXeSfANPjbyMu", "text"),
    ("references/official/results/C_median_runtimes_beta_hidden.csv", "1NqbQLFai8jJ_l17AMDvEUdNmI1NJ7tpQ", "text"),
    ("references/papers/FloorSet_2405.05480.pdf", "https://arxiv.org/pdf/2405.05480", "pdf"),
    ("references/official/Problems.html", "https://www.iccad-contest.org/Problems", "html"),
    ("references/official/FAQ.html", "https://www.iccad-contest.org/FAQ.html", "html"),
    ("references/official/News.html", "https://www.iccad-contest.org/News.html", "html"),
]


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FloorSet-research-archive/1.0"}), timeout=90)


def validate(data, kind):
    if not data:
        raise ValueError("Empty response")
    head = data[:1000].lstrip().lower()
    if kind == "pdf" and not data.startswith(b"%PDF-"):
        raise ValueError("Expected PDF; received another format")
    if kind != "html" and (b"<!doctype html" in head or b"<html" in head):
        raise ValueError("HTML response instead of the requested file (possible access restriction)")
    if kind == "python":
        compile(data.decode("utf-8-sig"), "op_wrapper.py", "exec")


def fetch(source, kind):
    url = source if source.startswith("https://") else "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode({"id": source, "export": "download", "confirm": "t"})
    with request(url) as response:
        data = response.read()
    if not source.startswith("https://") and b"download-form" in data[:200000]:
        page = data.decode("utf-8")
        action = re.search(r'<form[^>]+action="([^"]+)"', page)
        params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
        if action and params:
            url = html.unescape(action.group(1)) + "?" + urllib.parse.urlencode({k: html.unescape(v) for k, v in params.items()})
            with request(url) as response:
                data = response.read()
    validate(data, kind)
    return data


def main():
    records = []
    for rel, source, kind in RESOURCES:
        path = ROOT / rel
        record = {"path": rel, "source_url": source if source.startswith("https://") else DRIVE.format(source)}
        try:
            if path.exists():
                data = path.read_bytes()
                validate(data, kind)
            else:
                for attempt in range(3):
                    try:
                        data = fetch(source, kind)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 * (attempt + 1))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            record.update(status="available", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            print(f"OK {rel} ({len(data)} bytes)", flush=True)
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            print(f"FAILED {rel}: {exc}", flush=True)
        records.append(record)
    manifest = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "datasets_downloaded": False, "resources": records}
    target = ROOT / "references/resources_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(r["status"] == "failed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
