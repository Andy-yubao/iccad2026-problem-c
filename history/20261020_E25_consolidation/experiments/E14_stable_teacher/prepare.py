"""Materialize compatible dev/train members; never read sealed local_test."""
import argparse
from collections import Counter
import json
import time
from .data import HERE, ROOT, VERSION, candidates, reader, mean_area_inputs, save, sha, balanced_panel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=['dev', 'train'], required=True)
    ap.add_argument('--count', type=int)
    args = ap.parse_args()
    count = args.count or (2000 if args.split == 'dev' else 16000)
    out = HERE / 'runs/data_v1'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.split}.json'
    if path.exists():
        obj = json.loads(path.read_text(encoding='utf-8'))
        assert obj['count'] == count and obj['version'] == VERSION
        print('EXISTS', path, flush=True)
        return
    begin = time.perf_counter()
    preferred = []
    if args.split == 'dev':
        legacy = ROOT / 'experiments/E10_integrated_baseline/runs/20261018_mean_area_mib100_seed0/panel.json'
        preferred = [(e['source']['shard'], e['source']['sample_index']) for e in json.loads(legacy.read_text(encoding='utf-8'))['members']]
    preferred_set = set(preferred)
    pairs = preferred + [p for p in candidates(args.split) if p not in preferred_set]
    source = reader(pairs, args.split)
    quota = count // 100
    selected, reserve, rejected = [], [], Counter()
    coverage = Counter()
    examined = 0
    # Full shard round-robin can thrash the raw loader. Cache selected shards
    # in reader's ordinary cache; each row remains a predetermined candidate.
    for i, pair in enumerate(pairs):
        w = pair[0].split('/')[0]
        if coverage[w] >= quota and i >= len(preferred):
            continue
        raw = source.case(i)
        inputs, audit = mean_area_inputs(raw.solver_inputs)
        examined += 1
        if inputs is None:
            rejected[audit] += 1
            continue
        selected.append({'member': list(pair), 'changes': audit})
        coverage[w] += 1
        if len(selected) % 100 == 0:
            print('ACCEPT', len(selected), 'examined', examined, 'seconds', round(time.perf_counter()-begin, 1), flush=True)
        if len(selected) == count:
            break
    # Some workers have no compatible task; fill only in fixed order, not by Q.
    if len(selected) < count:
        used = {tuple(e['member']) for e in selected}
        for i, pair in enumerate(pairs):
            if pair in used:
                continue
            raw = source.case(i)
            inputs, audit = mean_area_inputs(raw.solver_inputs)
            examined += 1
            if inputs is None:
                rejected[audit] += 1
                continue
            selected.append({'member': list(pair), 'changes': audit})
            coverage[pair[0].split('/')[0]] += 1
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f'Insufficient compatible members: {len(selected)}/{count}')
    obj = {'version': VERSION, 'split': args.split, 'count': count,
           'split_sha256': sha(ROOT / 'protocol/split_v1.json'),
           'workset_sha256': sha(ROOT / 'protocol/workset_v2.json'),
           'selection': 'legacy dev100 first; worker quotas in versioned hash order; fill deficits in the same order; public compatibility only',
           'examined': examined, 'rejections': dict(rejected), 'worker_counts': dict(coverage),
           'seconds': time.perf_counter()-begin, 'members': selected,
           'local_test_read': False}
    save(path, obj)
    if args.split == 'dev':
        primary = selected[:100]
        # Preferred 100 are always accepted and retained in their original order.
        assert [e['member'] for e in primary] == [list(p) for p in preferred]
        save(out / 'panels.json', {'version': VERSION, 'development100': [e['member'] for e in primary],
                                  'confirmation500': balanced_panel([e['member'] for e in selected],exclude=[e['member'] for e in primary]),
                                  'confirmation_rule': 'five hash-ranked compatible dev members per worker, excluding all development100; frozen before any confirmation solve',
                                  'selection_uses_Q': False, 'local_test_read': False})
        save(out / 'partition.json', {'version': VERSION, 'targets': {'train':16000,'dev':2000,'local_test':2000},
            'original_split_unchanged': True, 'local_test_read': False,
            'test_status': 'sealed; deterministic candidate order and public compatibility rule registered, membership not materialized',
            'test_candidate_order_sha256': __import__('hashlib').sha256(json.dumps(candidates('local_test')).encode()).hexdigest(),
            'test_selection': '20 compatible members per worker in versioned hash order, deficits filled in same order; requires separate sealed-read authorization',
            'train_generation': 'new corrected tasks only; never use the legacy teacher bank',
            'files': {'train':'train.json','dev':'dev.json','panels':'panels.json'}})
    print('SAVED', path, count, flush=True)


if __name__ == '__main__':
    main()
