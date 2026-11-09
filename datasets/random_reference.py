"""Independent raw-reference dataset; no legacy split/filter or MIB transform."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import ROOT

REGISTRY = ROOT / "protocol/random_reference40k_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def registry(path=REGISTRY):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = dict(value)
    expected = identity.pop("manifest_sha256")
    if digest(identity) != expected:
        raise ValueError("New dataset manifest checksum mismatch")
    pools = []
    members = []
    for role, count in (("train", 36000), ("dev", 2000), ("test", 2000)):
        records = value["sets"][role]["records"]
        pairs = {tuple(r["member"]) for r in records}
        if len(records) != count or len(pairs) != count:
            raise ValueError("Wrong sample count or duplicate member")
        pool = set(value["shard_partitions"][role])
        if any(s not in pool or not 0 <= row < 112 for s, row in pairs):
            raise ValueError("Member outside new shard partition")
        pools.append(pool)
        members.append(pairs)
    if any(pools[a] & pools[b] or members[a] & members[b]
           for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("New split overlaps")
    return value


class RandomReference:
    """Train/dev only. Free geometry, metrics, and images are supervision only."""
    def __init__(self, role, registry_path=REGISTRY):
        if role not in ("train", "dev", "train_monitor500", "dev_monitor500"):
            raise PermissionError("Test payloads remain sealed after preparation")
        self.reg = registry(registry_path)
        self.role = role.split("_")[0]
        self.records = self.reg["sets"][self.role]["records"]
        self.indices = (self.reg["panels"][role] if "monitor" in role
                        else list(range(len(self.records))))
        config = json.loads((ROOT / "protocol/paths.json").read_text(encoding="utf-8"))
        self.folder = (ROOT / config["random_reference40k_root"]).resolve()
        if not self.folder.is_relative_to(ROOT):
            raise ValueError("Dataset path escapes workspace")
        report = json.loads((self.folder / "report.json").read_text(encoding="utf-8"))
        if report["manifest_sha256"] != self.reg["manifest_sha256"]:
            raise ValueError("Dataset completion identity mismatch")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record = self.records[self.indices[index]]
        with np.load(self.folder / record["payload"], allow_pickle=False) as item:
            inputs = (len(item["areas"]), *[item[key].copy() for key in
                      ("areas", "b2b", "p2b", "pins", "constraints", "specs")])
            # Keep labels separate; never feed the canvas or free geometry to a model.
            supervision = {key: item[key].copy() for key in ("reference_xywh", "stored_metrics")}
        return {"member": record["member"], "solver_inputs": inputs,
                "supervision": supervision, "image_path": self.folder / record["image"]}
