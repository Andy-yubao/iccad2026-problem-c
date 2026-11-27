#!/usr/bin/env python
"""Build manifest.json for the 2026-11-11 experiments consolidation.

Re-checks pre/post identity (parsed mtimes, not raw strings: the pre-move
manifest was produced by PowerShell with 'Z' timestamps, the post-move one
by Python with '+00:00'), emits manifest.json and copies verification
evidence plus the one-off scripts into the archive.
"""
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"E:\project\ICCAD\2026\problemC")
ARCHIVE = ROOT / "history" / "20261111_experiments_consolidation"
TMP = ROOT / "tmp"

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def load(csv_path):
    rows = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        header = f.readline().strip("\r\n").split("\t")
        assert header == ["relpath", "size", "mtime_utc", "sha256"], header
        for line in f:
            rel, size, mtime, sha = line.rstrip("\r\n").split("\t")
            rows[rel] = (int(size), parse_ts(mtime), sha.upper())
    return rows

pre = load(TMP / "premove_manifest.csv")
post = load(TMP / "postmove_manifest.csv")

missing = sorted(set(pre) - set(post))
extra = sorted(set(post) - set(pre))
size_diff = [r for r in set(pre) & set(post) if pre[r][0] != post[r][0]]
sha_diff = [r for r in set(pre) & set(post)
            if pre[r][2] and post[r][2] and pre[r][2] != post[r][2]]
mtime_max = max((abs((pre[r][1] - post[r][1]).total_seconds())
                 for r in set(pre) & set(post)), default=0.0)
ok = not (missing or extra or size_diff or sha_diff) and mtime_max <= 1.0
if not ok:
    raise SystemExit(f"VERIFY_FAILED missing={len(missing)} extra={len(extra)} "
                     f"size={len(size_diff)} sha={len(sha_diff)} mtime_max={mtime_max}")

per_dir = defaultdict(lambda: [0, 0])
for rel, (size, _mtime, _sha) in pre.items():
    top = rel.split("\\", 1)[0]
    per_dir[top][0] += 1
    per_dir[top][1] += size

now = datetime.now(timezone.utc).isoformat()
moves = [{
    "old": f"experiments/{name}",
    "new": f"history/20261111_experiments_consolidation/experiments/{name}",
    "file_count": per_dir[name][0],
    "bytes": per_dir[name][1],
} for name in sorted(per_dir)]

manifest = {
    "status": "verified_complete",
    "created_utc": now,
    "authorization": (
        "User 2026-11-11: 按工作区整理规范和历史归档要求，把 experiments 整理为仅保留 "
        "E119_global_initial_generator 与 E120_node12_collision；其余实验原件同盘归档，不删除唯一证据"
    ),
    "pre_move_checks": {
        "queue_states": "所有已登记实验 queue_state.json status=complete（E119/E120/E121-E130）",
        "live_processes": "记录的队列控制器PID均不在存活进程列表；移动前30分钟 experiments 下零文件写入",
        "reparse_points": "experiments 全树扫描 junction/symlink 数量为0（隔离运行根内为同卷硬链接，rename 不破坏）",
        "runtime_references": (
            "protocol/paths.json 与 datasets/、scoring/ 源码均不引用 experiments 目录；"
            "E119/E120 为自包含本地算法副本，不引用其他实验"
        ),
    },
    "move_method": "逐目录同卷 rename（mv），rename 不改写内容；移动后用独立工具重新枚举并核对",
    "verification": {
        "method": (
            "移动前（PowerShell）与移动后（Python）各生成逐文件清单：相对路径/字节大小/UTC mtime；"
            "≤8MiB 文件两侧各算 SHA256。比较结果：文件集合完全一致，大小全部相等，"
            f"{sum(1 for v in pre.values() if v[2])} 个 SHA256 全部相等，"
            f"mtime 解析后最大差 {mtime_max} 秒（时间戳精度舍入）"
        ),
        "long_path_note": (
            "E101 隔离运行根（runs/20261103_replay 与 runs/20261104_chain64k_seed0 的 s42/s74/sf）"
            "内含整棵 history 缓存结构；归档位置前缀更长后，部分深层路径超过 Win32 MAX_PATH 260，"
            "PowerShell 5.1 无法访问。读取该子树须用支持长路径的工具（Python 3.6+ 或 \\\\?\\ 前缀）。"
            "移动前清单无任何访问错误，说明全部文件在原位置均于 MAX_PATH 内、被完整记录"
        ),
        "pre_manifest": "verification/premove_manifest.csv",
        "post_manifest": "verification/postmove_manifest.csv",
        "result_record": "verification/verify_result.txt",
        "files_compared": len(pre),
        "sha256_compared": sum(1 for v in pre.values() if v[2]),
        "max_mtime_delta_seconds": mtime_max,
        "verified_utc": now,
    },
    "kept_in_experiments": [
        "E119_global_initial_generator",
        "E120_node12_collision",
        "README.md（重写为当前区导航）",
        "__init__.py",
    ],
    "deleted_rebuildable": ["experiments/__pycache__（唯一 __init__.cpython-310.pyc，可重建缓存）"],
    "total_file_count": len(pre),
    "total_bytes": sum(v[0] for v in pre.values()),
    "moves": moves,
    "references_note": (
        "docs/output/visualization 历史条目中的 experiments/<archived> 相对链接指向旧位置，"
        "按本 manifest 解释，不批改冻结证据；.gitignore 中指向旧路径的规则成为 no-op，保留"
    ),
    "e98_note": (
        "E98 曾于 2026-11-04 从 history/20261103_final_delivery_consolidation 迁回 experiments"
        "（该归档 E98_restore_20261104.json），本次按用户指令再次归档；历史记录保持不变"
    ),
}

(ARCHIVE / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
ver = ARCHIVE / "verification"
ver.mkdir(exist_ok=True)
shutil.copy2(TMP / "premove_manifest.csv", ver / "premove_manifest.csv")
shutil.copy2(TMP / "postmove_manifest.csv", ver / "postmove_manifest.csv")
result_text = (
    "postmove_manifest.py verify result (2026-11-11)\n"
    f"pre_files={len(pre)} post_files={len(post)}\n"
    f"sha256_compared={sum(1 for v in pre.values() if v[2])} all_equal=True\n"
    f"missing_in_post={len(missing)} extra_in_post={len(extra)} "
    f"size_diff={len(size_diff)} sha_diff={len(sha_diff)}\n"
    f"max_mtime_delta_seconds={mtime_max}\n"
    "VERIFY_OK_SETS_SIZES_SHA_MTIME_ALL_MATCH\n"
)
(ver / "verify_result.txt").write_text(result_text, encoding="utf-8")
scripts = ARCHIVE / "scripts"
scripts.mkdir(exist_ok=True)
for s in [
    "scan_reparse_points.ps1",
    "recent_writes_check.ps1",
    "make_manifest.ps1",
    "move_to_archive.sh",
    "postmove_manifest.py",
    "build_archive_manifest.py",
]:
    shutil.copy2(TMP / s, scripts / s)

print("MANIFEST_WRITTEN")
print(f"files={len(pre)} dirs={len(moves)} bytes={sum(v[0] for v in pre.values())} sha={sum(1 for v in pre.values() if v[2])}")
