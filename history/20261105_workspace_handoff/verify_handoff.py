"""Verify navigation and this cleanup's preserved evidence; never run experiments."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re


def main():
    archive = Path(__file__).resolve().parent
    root = archive.parent.parent
    current = [
        root / "README.md", root / "AGENTS.md",
        *sorted((root / "docs").glob("*.md")),
        root / "experiments/README.md",
        *sorted((root / "experiments").glob("*/README.md")),
        root / "output/README.md", root / "history/README.md",
        root / "output/model_architecture_proposals_20261105.md",
        archive / "README.md",
    ]
    missing, links = [], []
    for path in current:
        text = path.read_text(encoding="utf-8-sig")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).strip().split(' "', 1)[0].strip("<>")
            if not target or target.startswith(("https:", "http:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            row = {"from": path.relative_to(root).as_posix(), "target": target}
            links.append(row)
            if not resolved.exists():
                missing.append(row)

    failures = []
    snapshots = json.loads((archive / "documents_snapshot.json").read_text(encoding="utf-8"))
    for row in snapshots:
        path = root / row["snapshot"]
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append({"kind": "document_snapshot", "path": row["snapshot"]})
    moves = json.loads((archive / "moves.json").read_text(encoding="utf-8"))["moves"]
    for row in moves:
        path = root / row["to"]
        stat = path.stat()
        valid = (not (root / row["from"]).exists()
                 and stat.st_size == row["bytes"] and stat.st_mtime_ns == row["mtime_ns"]
                 and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"])
        if not valid:
            failures.append({"kind": "move", "path": row["to"]})
    states = json.loads((archive / "run_state_snapshot.json").read_text(encoding="utf-8"))["sources"]
    for row in states:
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            failures.append({"kind": "state_evidence_drift", "path": row["path"]})

    # Evidence index includes local references used by the new synthesis/status,
    # excluding navigation pages, numeric payloads, and model checkpoints.
    evidence = {}
    for name in ["START_HERE.md", "research_history.md", "current_status.md"]:
        source = root / "docs" / name
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0]
            if target.startswith(("http:", "https:")):
                continue
            path = (source.parent / unquote(target)).resolve()
            if not path.is_file() or path.suffix not in {".md", ".json"}:
                continue
            rel = path.relative_to(root).as_posix()
            if rel.startswith("docs/") or path.name == "README.md":
                continue
            data = path.read_bytes()
            evidence[rel] = {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (archive / "evidence_index.json").write_text(
        json.dumps({"scope": "Compact linked evidence only; no dataset/checkpoint payloads",
                    "sources": list(evidence.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "pass" if not missing and not failures else "fail",
        "current_markdown_files": len(current), "local_links_checked": len(links),
        "missing_local_links": missing, "preservation_failures": failures,
        "document_snapshots_verified": len(snapshots), "script_moves_verified": len(moves),
        "run_evidence_hashes_unchanged": len(states), "linked_evidence_sources": len(evidence),
        "tmp_empty": not any((root / "tmp").iterdir()),
        "scope": "Current navigation and this cleanup only; not all historical links or scientific validity",
    }
    (archive / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
