"""Frozen train16k view of random40k: index preparation and train/dev reading only."""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .paths import ROOT
from .random_reference import REGISTRY as PARENT_REGISTRY
from .random_reference import RandomReference, digest, registry as parent_registry

REGISTRY = ROOT/'protocol/random_reference16k_v1.json'
POLICY = ROOT/'protocol/research_training16k_v1.json'
REPORT = ROOT/'protocol/random_reference16k_preparation.json'
ALLOWED_ROLES = ('train', 'train_monitor500', 'dev', 'dev_monitor500')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy():
    value = json.loads(POLICY.read_text(encoding='utf-8'))
    if (value['parent_training_count'], value['default_training_count'],
            value['training_monitor_count']) != (36000,16000,500):
        raise ValueError('Unexpected registered subset sizes')
    return value


def selected_indices(spec):
    # Sorting affects enumeration only, not the uniform sample membership.
    return sorted(random.Random(spec['selection_seed']).sample(range(36000),16000))


def subset_registry(path=REGISTRY):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    signed = dict(value)
    expected = signed.pop('manifest_sha256')
    if digest(signed) != expected:
        raise ValueError('Subset manifest checksum mismatch')
    spec = policy()
    parent = parent_registry()
    if value['policy_sha256'] != file_sha(POLICY):
        raise ValueError('Subset policy changed; migration/new identity required')
    if (value['parent']['file_sha256'] != file_sha(PARENT_REGISTRY) or
            value['parent']['manifest_sha256'] != parent['manifest_sha256']):
        raise ValueError('Frozen parent changed')
    indices = value['train_parent_indices']
    if (len(indices) != 16000 or any(type(i) is not int for i in indices) or
            indices != selected_indices(spec)):
        raise ValueError('Subset membership differs from registered sampling')
    members = [parent['sets']['train']['records'][i]['member'] for i in indices]
    if len({tuple(m) for m in members}) != 16000 or digest(members) != value['train_members_sha256']:
        raise ValueError('Subset member mapping mismatch')
    monitor = value['panels']['train_monitor500']
    expected_monitor = random.Random(spec['training_monitor_seed']).sample(range(16000),500)
    if monitor != expected_monitor:
        raise ValueError('Subset train monitor changed')
    if value['panels']['dev_monitor500'] != parent['panels']['dev_monitor500']:
        raise ValueError('Original dev monitor changed')
    for role in ('dev','test'):
        if digest(parent['sets'][role]['records']) != value['unchanged_roles'][role]['records_sha256']:
            raise ValueError('Parent development/test membership changed')
    return value, parent


class RandomReference16k:
    """Same sample fields as RandomReference; test roles are rejected before I/O."""
    def __init__(self, role, registry_path=REGISTRY):
        if role not in ALLOWED_ROLES:
            raise PermissionError('Only train/dev views are allowed; test payloads remain sealed')
        self.reg, parent = subset_registry(registry_path)
        self.role = role.split('_')[0]
        self._parent = RandomReference(self.role)
        self.folder = self._parent.folder
        if file_sha(self.folder/'report.json') != self.reg['parent']['completion_report_sha256']:
            raise ValueError('Parent completion evidence changed')
        self.parent_indices = (self.reg['train_parent_indices'] if self.role=='train'
                               else list(range(2000)))
        self.records = [parent['sets'][self.role]['records'][i] for i in self.parent_indices]
        self.indices = (self.reg['panels'][role] if 'monitor' in role
                        else list(range(len(self.records))))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        local = self.indices[index]
        item = self._parent[self.parent_indices[local]]
        if tuple(item['member']) != tuple(self.records[local]['member']):
            raise ValueError('Parent payload/member mapping changed')
        # Supervision and public inputs retain the parent's explicit separation.
        return item


def prepare():
    if REGISTRY.exists():
        raise FileExistsError('Frozen subset already exists; use check, never silently resample')
    spec = policy()
    parent = parent_registry()
    # Check parent completion metadata only. No training/dev/test payload is opened.
    data = RandomReference('train')
    completed = data.folder/'report.json'
    report = json.loads(completed.read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['counts'] != dict(train=36000,dev=2000,test=2000):
        raise ValueError('Parent dataset is not complete')
    indices = selected_indices(spec)
    value = dict(dataset_version='random_reference16k.v1', policy_version=spec['policy_version'],
        created_at=datetime.now(timezone.utc).isoformat(), policy_sha256=file_sha(POLICY),
        parent=dict(registry=PARENT_REGISTRY.relative_to(ROOT).as_posix(),
            file_sha256=file_sha(PARENT_REGISTRY), manifest_sha256=parent['manifest_sha256'],
            completion_report_sha256=file_sha(completed),
            payload_checksums_sha256=report['checksums_sha256']),
        counts=dict(train=16000,dev=2000,test=2000),
        sampling=dict(seed=spec['selection_seed'],without_replacement=True,
            population='parent train36000 only',enumeration='ascending parent index',
            filtering='none; no N balancing, labels, scores or predictions'),
        train_parent_indices=indices,
        train_members_sha256=digest([parent['sets']['train']['records'][i]['member'] for i in indices]),
        panels=dict(train_monitor500=random.Random(spec['training_monitor_seed']).sample(range(16000),500),
                    dev_monitor500=parent['panels']['dev_monitor500']),
        unchanged_roles={role:dict(count=2000,records_sha256=digest(parent['sets'][role]['records']))
                         for role in ('dev','test')},
        payload_storage='Reuse existing parent NPZ/PNG, original areas/specifications; no duplication',
        applies_after='E114 staged layer4/12 supervision; all current E114 arms keep train36000',
        test_payload_reads=0,training_started=False)
    value['manifest_sha256'] = digest(value)
    # Exclusive creation protects the registered subset from accidental replacement.
    with REGISTRY.open('x',encoding='utf-8') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    subset_registry()
    print(json.dumps(dict(status='subset_frozen',counts=value['counts'],
                          manifest=REGISTRY.relative_to(ROOT).as_posix(),payloads_read=0)))


def check(payloads=0):
    if payloads not in range(6):
        raise ValueError('Implementation inspection is limited to at most five train members')
    value, parent = subset_registry()
    train = RandomReference16k('train')
    monitor = RandomReference16k('train_monitor500')
    dev = RandomReference16k('dev')
    dev_monitor = RandomReference16k('dev_monitor500')
    assert [len(v) for v in (train,monitor,dev,dev_monitor)] == [16000,500,2000,500]
    assert {tuple(train.records[i]['member']) for i in monitor.indices} <= {tuple(r['member']) for r in train.records}
    assert [dev_monitor.records[i]['member'] for i in dev_monitor.indices] == [
        parent['sets']['dev']['records'][i]['member'] for i in parent['panels']['dev_monitor500']]
    blocked = False
    try:
        RandomReference16k('test')
    except PermissionError:
        blocked = True
    assert blocked
    checked = []
    if payloads:
        ledger = train.folder/'checksums.jsonl'
        if file_sha(ledger) != value['parent']['payload_checksums_sha256']:
            raise ValueError('Parent payload checksum ledger changed')
        chosen = monitor.indices[:payloads]
        targets = {train.records[i]['id']:i for i in chosen}
        hashes = {}
        for line in ledger.read_text(encoding='utf-8').splitlines():
            for record in json.loads(line)['records']:
                if record['id'] in targets:
                    hashes[record['id']] = record['payload_sha256']
        for index in chosen:
            record = train.records[index]
            if file_sha(train.folder/record['payload']) != hashes[record['id']]:
                raise ValueError('Parent payload checksum mismatch')
            item = train[index]
            assert tuple(item['member']) == tuple(record['member'])
            n, areas, _, _, _, _, specs = item['solver_inputs']
            assert len(areas)==n and specs.shape==(n,4)
            assert item['supervision']['reference_xywh'].shape==(n,4)
            checked.append(dict(subset_index=index,parent_index=train.parent_indices[index],member=record['member']))
    result = dict(status='passed',manifest_sha256=value['manifest_sha256'],
        counts=value['counts'],train_monitor_count=500,dev_monitor_count=500,
        uniform_sampling_reproduced=True,train_members_unique_and_within_parent=True,
        parent_dev_and_test_metadata_unchanged=True,dev_monitor_unchanged=True,
        train_payloads_checked=len(checked),checked=checked,dev_payloads_read=0,test_payloads_read=0,
        test_reader_rejected=True,training_started=False,parent_files_written=0,
        source_sha256=file_sha(Path(__file__)))
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checked',)},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check'))
    parser.add_argument('--payloads',type=int,choices=range(6),default=0)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.payloads:
            parser.error('prepare creates indices only; use check --payloads N separately')
        prepare()
    else:
        check(args.payloads)


if __name__=='__main__':
    main()
