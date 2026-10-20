"""Frozen v4 solver-generated layouts, public inputs and separate supervision.

No model/loss/search code lives here. Original local_test is never accessed.
The student test roles are denied by default, even though their source is train.
"""
import hashlib
import json
import io
from functools import lru_cache
import numpy as np
from .paths import ROOT
from .worksets import FrozenMemberReader
from .cases import _build_case, ResearchCase

REGISTRY = ROOT / 'protocol/distillation_workset_v4.json'
REGISTRY_SHA256 = '4a30f802aa541b3da17d31bfeeead54f4425beb86c69231cfdd367ecc35c4860'
TASK = 'E14.mean_area.v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_area_inputs(inputs):
    """E14.mean_area.v1 public-only data transformation, copied without changes."""
    values = tuple(v.clone() if hasattr(v, 'clone') else v for v in inputs)
    _, areas, _, _, _, constraints, specs = values
    c, s = np.asarray(constraints), np.asarray(specs)
    locked = c[:, :2].any(1)
    for group in sorted(set(c[:, 2]) - {0}):
        ids = np.flatnonzero(c[:, 2] == group)
        areas[ids] = float(areas[ids].double().mean())
        rigid = ids[locked[ids]]
        if len(rigid):
            wh = s[rigid[0], 2:]
            if not np.allclose(s[rigid, 2:], wh, rtol=0, atol=1e-10):
                raise ValueError('locked_dimensions_disagree')
            if abs(float(wh.prod()) / float(areas[ids[0]]) - 1) >= .009999:
                raise ValueError('mean_area_incompatible_with_locked_shape')
    return values


@lru_cache(maxsize=2)
def _reader(shard):
    return FrozenMemberReader([(shard, i) for i in range(112)], 'train',
                              identity=hashlib.sha256(shard.encode()).hexdigest())


class QualityLayouts:
    def __init__(self, role, *, evaluation_authorized=False):
        if sha(REGISTRY) != REGISTRY_SHA256:
            raise ValueError('Frozen v4 registry changed')
        registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
        if role in ('test', 'tail_test') and not evaluation_authorized:
            raise PermissionError('Student test/tail_test is sealed')
        if role not in registry['sets']:
            raise ValueError(role)
        if registry['dataset_version'] != 'E14.distillation_top20000.v1' or registry['task_version'] != TASK:
            raise ValueError('Unsupported data version')
        for name, expected in registry['frozen_original_hashes'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen original changed: ' + name)
        path = ROOT / registry['sets'][role]['path']
        if sha(path) != registry['sets'][role]['sha256']:
            raise ValueError('Student manifest changed')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['role'] != role or manifest['version'] != registry['dataset_version']:
            raise ValueError('Manifest role/version mismatch')
        self.records = manifest['records']
        self.role = role
        if len(self.records) != registry['sets'][role]['count']:
            raise ValueError('Wrong manifest size')
        members = [tuple(r['member']) for r in self.records]
        if len(set(members)) != len(members):
            raise ValueError('Duplicate members')
        self.registry_sha256 = sha(REGISTRY)

    def __len__(self):
        return len(self.records)

    def case(self, index):
        shard, row = self.records[index]['member']
        raw = _reader(shard).case(row)
        inputs = mean_area_inputs(raw.solver_inputs)
        case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6],
                           raw.reference_positions, raw.stored_metrics)
        case.solver_inputs = inputs
        return case

    def __getitem__(self, index):
        record = self.records[index]
        path = (ROOT / record['archive_file']).resolve()
        archive = (ROOT / 'experiments/E14_stable_teacher/runs/teacher_archive25000_v1/layouts').resolve()
        if not path.is_relative_to(archive) or sha(path) != record['archive_sha256']:
            raise ValueError('Invalid layout path/checksum')
        case = self.case(index)
        with np.load(path, allow_pickle=False) as z:
            if str(z['schema']) != 'E14.teacher.mean_area.v1' or str(z['task_version']) != TASK:
                raise ValueError('Layout schema/task version mismatch')
            if not np.array_equal(z['input_areas'], np.asarray(case.solver_inputs[1])):
                raise ValueError('Derived areas differ from stored task')
            positions = z['positions'].copy()
        if positions.shape != (case.block_count, 4) or not np.isfinite(positions).all() or (positions[:, 2:] <= 0).any():
            raise ValueError('Invalid target layout')
        return {'member': tuple(record['member']), 'case': case, 'positions': positions}

    def archive_item(self, index):
        """Read full archives directly, or resolve imported label-only archives.

        Archive hashes were fixed with v4. Stored original scoring anchors stay
        on the supervision side. Hard checks need only public locked specs;
        ordinary reference coordinates are neither reconstructed nor exposed.
        """
        import torch
        record = self.records[index]
        path = ROOT / record['archive_file']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['archive_sha256']:
            raise ValueError('Frozen archive checksum mismatch')
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            if str(z['schema']) != 'E14.teacher.mean_area.v1' or str(z['task_version']) != TASK:
                raise ValueError('Archive schema/task mismatch')
            # Imported accepted labels intentionally predate the self-contained
            # archive format: they have no record_json or public graph arrays.
            # Use the frozen member reader for those cases, including the same
            # original scoring anchors and derived-area consistency checks.
            required = {'record_json', 'input_areas', 'b2b', 'p2b', 'pins', 'constraints', 'specs'}
            if not required.issubset(z.files):
                return self[index]
            info = json.loads(str(z['record_json']))
            if tuple(info['member']) != tuple(record['member']):
                raise ValueError('Archive member mismatch')
            values = [torch.from_numpy(z[k].copy()) for k in
                      ('input_areas', 'b2b', 'p2b', 'pins', 'constraints', 'specs')]
            positions = z['positions'].copy()
        n = len(values[0])
        if n != record['n'] or positions.shape != (n, 4) or not np.isfinite(positions).all() or (positions[:,2:]<=0).any():
            raise ValueError('Archive geometry/count mismatch')
        metrics = info['metrics']
        anchors = {'hpwl_baseline': float(metrics['hpwl_baseline']),
                   'area_baseline': float(metrics['bbox_area_baseline'])}
        if not all(np.isfinite(v) and v >= 0 for v in anchors.values()) or anchors['area_baseline'] <= 0:
            raise ValueError('Invalid archive scoring anchors')
        reference = public_hard_reference(values[0], values[4], values[5])
        case = ResearchCase(info['case_id'], {'member': record['member'], 'archive': record['archive_file'],
            'reference_scope': 'public hard specs only; original denominators from frozen archive'},
            (n, *values), reference, anchors)
        return {'member': tuple(record['member']), 'case': case, 'positions': positions}


def public_hard_reference(areas, constraints, specs):
    """Only fixed WH/preplaced XYWH are used by the unchanged hard checkers."""
    areas, c, s = np.asarray(areas), np.asarray(constraints), np.asarray(specs)
    reference = np.column_stack((np.zeros((len(areas), 2)), np.sqrt(areas), np.sqrt(areas)))
    locked, pre = c[:, :2].any(1), c[:, 1] != 0
    reference[locked, 2:] = s[locked, 2:]
    reference[pre] = s[pre]
    return reference.tolist()
