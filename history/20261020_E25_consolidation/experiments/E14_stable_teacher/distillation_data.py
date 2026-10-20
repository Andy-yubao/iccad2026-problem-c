"""Access v4 student data; primary/tail tests denied until explicit evaluation use."""
import json
from functools import lru_cache
import numpy as np
from .data import ROOT,VERSION,reader,derived_case,sha
from .targets import SCHEMA,validate
from .frozen.packing import prepare_problem

REGISTRY=ROOT/'protocol/distillation_workset_v4.json'
FIELDS=('positions','geometry','separation','contact_forest','group_contact_positive','geometry_learned_mask',
        'boundary_satisfied','mib_same','mib_labels','group_same_component','group_components','teacher_area_ratio')


@lru_cache(maxsize=4)
def _get_reader(shard):
    # Process-local cache; the dataset itself stays pickleable for Windows spawn.
    return reader([(shard,i) for i in range(112)],'train')


class DistillationData:
    def __init__(self,role='train',*,evaluation_authorized=False):
        registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
        if role not in registry['sets']:raise ValueError(f'Unknown split {role}')
        if role in ('test','tail_test') and not evaluation_authorized:
            raise PermissionError('Student test/tail_test is sealed; explicit post-training frozen evaluation authorization required')
        for p,expected in registry['frozen_original_hashes'].items():
            if sha(ROOT/p)!=expected:raise ValueError('Original frozen split changed')
        path=ROOT/registry['sets'][role]['path']
        if sha(path)!=registry['sets'][role]['sha256']:raise ValueError('Frozen student manifest changed')
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['role']!=role or manifest['version']!=registry['dataset_version']:raise ValueError('Student dataset version/role mismatch')
        self.role=role;self.records=manifest['records'];self.version=registry['dataset_version']
        if len({tuple(r['member']) for r in self.records})!=len(self.records):raise ValueError('Duplicate student member')

    def __len__(self):return len(self.records)

    def case(self,index):
        shard,row=self.records[index]['member']
        return derived_case(_get_reader(shard).case(row))[0]

    def __getitem__(self,index):
        r=self.records[index];path=ROOT/r['archive_file']
        if sha(path)!=r['archive_sha256']:raise ValueError('Teacher layout checksum changed')
        case=self.case(index)
        with np.load(path) as z:
            if str(z['schema'])!=SCHEMA or str(z['task_version'])!=VERSION:raise ValueError('Teacher schema/task version mismatch')
            if not np.array_equal(z['input_areas'],np.asarray(case.solver_inputs[1])):raise ValueError('Derived input areas differ from teacher task')
            target={k:z[k].copy() for k in FIELDS}
        validate(target,prepare_problem(*case.solver_inputs))
        # Public inputs and supervision remain separate. Q and original free layouts
        # are absent from this returned network-input contract.
        return {'member':r['member'],'solver_inputs':case.solver_inputs,'teacher':target}
