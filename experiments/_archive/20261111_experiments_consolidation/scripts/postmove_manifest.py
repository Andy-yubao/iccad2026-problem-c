#!/usr/bin/env python
"""Post-move manifest for the 2026-11-11 experiments consolidation.

Python is used instead of PowerShell because part of the archived tree
(E101 isolated run roots with copied history caches) exceeds the Win32
MAX_PATH limit at the new, longer location; CPython handles long paths.
Produces the same 4-column TSV as the pre-move manifest and compares both.
"""
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT_UNPREFIXED = r"E:\project\ICCAD\2026\problemC\history\20261111_experiments_consolidation\experiments"
ROOT = "\\\\?\\" + ROOT_UNPREFIXED
OUT = Path(r"E:\project\ICCAD\2026\problemC\tmp\postmove_manifest.csv")
PRE = Path(r"E:\project\ICCAD\2026\problemC\tmp\premove_manifest.csv")
HASH_LIMIT = 8 * 1024 * 1024
PFX = len("\\\\?\\") + len(ROOT_UNPREFIXED) + 1  # prefix + root + separator

def iso_from_ns(ns):
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.isoformat()

def stat_file(full):
    st = os.stat(full)
    sha = ""
    if st.st_size <= HASH_LIMIT:
        h = hashlib.sha256()
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest().upper()
    rel = full[PFX:]
    return rel, st.st_size, iso_from_ns(st.st_mtime_ns), sha

rows = []
dirs_walked = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirs_walked += 1
    for name in filenames:
        rows.append(os.path.join(dirpath, name))

print(f"dirs_walked={dirs_walked} files={len(rows)}", flush=True)

with ThreadPoolExecutor(max_workers=32) as ex:
    results = list(ex.map(stat_file, rows, chunksize=256))

with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write("relpath\tsize\tmtime_utc\tsha256\r\n")
    for rel, size, mtime, sha in sorted(results):
        f.write(f"{rel}\t{size}\t{mtime}\t{sha}\r\n")
print("POST_CSV_WRITTEN", flush=True)

# ---- compare with pre-move manifest ----
def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def load(path):
    data = {}
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().strip("\r\n").split("\t")
        assert header == ["relpath", "size", "mtime_utc", "sha256"], header
        for line in f:
            rel, size, mtime, sha = line.rstrip("\r\n").split("\t")
            data[rel] = (int(size), parse_ts(mtime), sha.upper())
    return data

pre = load(PRE)
post = load(OUT)
problems = []
missing = sorted(set(pre) - set(post))
extra = sorted(set(post) - set(pre))
if missing:
    problems.append(f"MISSING_IN_POST={len(missing)} first={missing[:5]}")
if extra:
    problems.append(f"EXTRA_IN_POST={len(extra)} first={extra[:5]}")
size_diff = [r for r in set(pre) & set(post) if pre[r][0] != post[r][0]]
if size_diff:
    problems.append(f"SIZE_DIFF={len(size_diff)} first={size_diff[:5]}")
sha_diff = [r for r in set(pre) & set(post)
            if pre[r][2] and post[r][2] and pre[r][2] != post[r][2]]
if sha_diff:
    problems.append(f"SHA_DIFF={len(sha_diff)} first={sha_diff[:5]}")
mtime_delta_max = 0.0
mtime_diff = []
for r in set(pre) & set(post):
    delta = abs((pre[r][1] - post[r][1]).total_seconds())
    mtime_delta_max = max(mtime_delta_max, delta)
    if delta > 1.0:
        mtime_diff.append(r)
if mtime_diff:
    problems.append(f"MTIME_DIFF_OVER_1S={len(mtime_diff)} first={mtime_diff[:5]}")

print(f"pre_files={len(pre)} post_files={len(post)}")
print(f"sha_pre_compared={sum(1 for v in pre.values() if v[2])} sha_post_compared={sum(1 for v in post.values() if v[2])}")
print(f"max_mtime_delta_seconds={mtime_delta_max}")
if problems:
    print("VERIFY_FAILED")
    for p in problems:
        print(p)
    sys.exit(1)
print("VERIFY_OK_SETS_SIZES_SHA_MTIME_ALL_MATCH")
