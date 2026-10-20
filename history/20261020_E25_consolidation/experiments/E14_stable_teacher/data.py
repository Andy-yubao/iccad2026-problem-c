"""Versioned, input-only MIB transformation and frozen partition membership."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from datasets.paths import ROOT
from datasets.worksets import FrozenMemberReader
from datasets.cases import _build_case

HERE = Path(__file__).resolve().parent
VERSION = 'E14.mean_area.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Windows readers and indexers can briefly hold a handle without DELETE
    # sharing. Preserve the complete temp file and retry only the atomic move.
    for attempt in range(30):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(.02 * (attempt + 1), .2))


def mean_area_inputs(inputs):
    values = tuple(v.clone() if hasattr(v, 'clone') else v for v in inputs)
    _, areas, _, _, _, constraints, specs = values
    c, s = np.asarray(constraints), np.asarray(specs)
    locked = c[:, :2].any(1)
    changes = []
    for group in sorted(set(c[:, 2]) - {0}):
        ids = np.flatnonzero(c[:, 2] == group)
        old = areas[ids].tolist()
        areas[ids] = float(areas[ids].double().mean())
        common = float(areas[ids[0]])
        rigid = ids[locked[ids]]
        if len(rigid):
            wh = s[rigid[0], 2:]
            if not np.allclose(s[rigid, 2:], wh, rtol=0, atol=1e-10):
                return None, 'locked_dimensions_disagree'
            if abs(float(wh.prod()) / common - 1) >= .009999:
                return None, 'mean_area_incompatible_with_locked_shape'
        else:
            wh = np.full(2, np.sqrt(common))
        changes.append({'group': int(group), 'blocks': ids.tolist(), 'old_areas': old,
                        'new_area': common, 'shape_witness': wh.tolist(),
                        'locked_blocks': rigid.tolist()})
    return values, changes


def candidates(name):
    """Worker-round-robin order, hash-ranked shards and rows, metadata only."""
    split = json.loads((ROOT / 'protocol/split_v1.json').read_text(encoding='utf-8'))
    rank = lambda tag, value: hashlib.sha256(f'{VERSION}:20261019:{tag}:{value}'.encode()).hexdigest()
    workers = []
    for w in range(100):
        shards = sorted((s for s in split['shards'][name] if s.startswith(f'worker_{w}/')),
                        key=lambda s: (rank(name, s), s))
        workers.append([(s, r) for s in shards for r in sorted(range(112), key=lambda r: (rank('row', f'{s}:{r}'), r))])
    return [workers[w][i] for i in range(len(workers[0])) for w in range(100)]


def reader(members, split='dev'):
    digest=hashlib.sha256(json.dumps({'version':VERSION,'members':members},sort_keys=True).encode()).hexdigest()
    return FrozenMemberReader(members, split, identity=digest)


def balanced_panel(members, *, exclude=(), per_worker=5, tag='confirmation'):
    omitted={tuple(p) for p in exclude}
    chosen=[]
    for worker in range(100):
        available=[p for p in members if p[0].startswith(f'worker_{worker}/') and tuple(p) not in omitted]
        available.sort(key=lambda p:hashlib.sha256(f'{VERSION}:{tag}:{p[0]}:{p[1]}'.encode()).hexdigest())
        if len(available)<per_worker:
            raise ValueError(f'Insufficient {tag} members for worker_{worker}')
        chosen.extend(available[:per_worker])
    return chosen


def derived_case(raw):
    inputs, changes = mean_area_inputs(raw.solver_inputs)
    if inputs is None:
        raise ValueError(changes)
    case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6],
                       raw.reference_positions, raw.stored_metrics)
    case.solver_inputs = inputs
    return case, changes
