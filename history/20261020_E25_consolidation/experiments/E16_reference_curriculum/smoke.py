"""At most five real train/dev cases: new data/cache/controller behavior."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(); args.output = args.output.resolve()
    # Set the pilot identity before importing any module that binds REGISTRY.
    here = Path(__file__).resolve().parent
    if not args.output.is_relative_to(here/'runs'):
        raise ValueError('Smoke output must stay in E16/runs')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['E16_REGISTRY'] = str(args.output/'pilot_registry.json')
    import numpy as np
    import torch
    from datasets.paths import ROOT
    from datasets.reference_layouts import (VERSION, TASK, digest, sha, inspect_shard,
                                            ReferenceLayouts, registry)
    from scoring.evaluation import evaluate_case
    from .io import save, sources
    from .balancing import common_quotas, select_records, rank
    from .cache import build_role, PackedRows
    from .data import prepare, collate
    from .loss import losses, weights
    from .model import GeometryModel, decode
    from .train import ARMS, run, seed_all, gradient_norm, PlateauController, selection
    from .prepare import create_registry

    started = time.perf_counter(); seed_all(0)
    v3 = json.loads((ROOT/'protocol/derived_workset_v3.json').read_text(encoding='utf-8'))
    records, scans = {}, []
    for parent, count in (('train', 3), ('dev', 2)):
        selected = v3['sets'][parent]['members'][:count]
        records[parent] = []
        by_shard = {}
        for shard, row in selected:
            by_shard.setdefault(shard, []).append(row)
        for shard, rows in by_shard.items():
            scan = inspect_shard(shard, parent, sample_rows=rows); scans.append(scan)
            found = dict(scan['accepted'])
            assert set(found) == set(rows)
            for row in rows:
                records[parent].append({'member': [shard, row], 'n': found[row],
                                        'case_id': f'{parent}:{shard}:{row}'})
    records['local_test'] = []
    records['train_monitor500'] = list(records['train'])
    records['dev_monitor500'] = list(records['dev'])
    pilot = {'dataset_version': VERSION, 'task_version': TASK, 'pilot': True,
             'frozen_original_hashes': {p: sha(ROOT/p) for p in
                ('protocol/split_v1.json', 'protocol/workset_v2.json', 'protocol/derived_workset_v3.json')},
             'sets': {r: {'count': len(rows), 'records': rows} for r, rows in records.items()}}
    pilot['manifest_sha256'] = digest(pilot); save(args.output/'pilot_registry.json', pilot)
    assert registry()['pilot']
    # Equal, supply-limited and impossible strata are deterministic and explicit.
    supply = {r: {20: 8000 if r == 'train' else 1000,
                  40: 8000 if r == 'train' else 1000,
                  80: 8000 if r == 'train' else 1000} for r in ('train', 'dev', 'local_test')}
    q, cap = common_quotas(supply)
    assert sum(q.values()) == 2000 and max(q.values())-min(q.values()) <= 1
    assert common_quotas(supply)[0] == q
    supply['local_test'][80] = 20
    limited, _ = common_quotas(supply)
    assert limited[80] == 20 and sum(limited.values()) == 2000
    rejected = False
    try:
        common_quotas({r: {20: 1} for r in supply})
    except ValueError:
        rejected = True
    assert rejected
    metadata_rows = [{'n': 20, 'member': [f'worker_0/layouts_{i}.th', 0]} for i in range(12)]
    a = select_records(metadata_rows, {20: 5}, 'train', seed=1)
    b = select_records(metadata_rows, {20: 5}, 'train', seed=2)
    assert a != b and a == select_records(list(reversed(metadata_rows)), {20: 5}, 'train', seed=1)
    # Full-size registry assembly uses synthetic metadata only, no additional
    # real layouts. The old v2 test panel is excluded before balancing.
    split = json.loads((ROOT/'protocol/split_v1.json').read_text(encoding='utf-8'))
    fake_scans = []
    for role, shard_count in (('train', 150), ('dev', 25), ('local_test', 30)):
        for shard in split['shards'][role][:shard_count]:
            fake_scans.append({'parent': role, 'shard': shard, 'size': 0, 'mtime_ns': 0,
                'accepted': [[r, 20] for r in range(112)], 'raw_n_counts': {20: 112}, 'rejections': {}})
    synthetic = create_registry(fake_scans, identity={'frozen_original_hashes': pilot['frozen_original_hashes']}, seed=20261019)
    assert [synthetic['sets'][r]['count'] for r in ('train', 'dev', 'local_test')] == [16000, 2000, 2000]
    assert synthetic['historical_test_exclusion']['candidate_count'] == 2500
    v2 = json.loads((ROOT/'protocol/workset_v2.json').read_text(encoding='utf-8'))
    old_test = {(e['shard'], r) for e in v2['members']['local_test'] for r in e['sample_indices']}
    assert not old_test & {tuple(r['member']) for r in synthetic['sets']['local_test']['records']}
    seal_ok = False
    try:
        ReferenceLayouts('local_test')
    except PermissionError:
        seal_ok = True
    assert seal_ok

    direct = [[prepare(ReferenceLayouts(role)[i]) for i in range(count)]
              for role, count in (('train', 3), ('dev', 2))]
    cache_root = args.output/'cache'
    for role, count in (('train', 3), ('dev', 2)):
        # Pause before the first chunk, then resume with the same identity.
        assert build_role(role, cache_root/role, indices=range(count), workers=2,
                          chunk_size=2, should_stop=lambda: True) is False
        assert build_role(role, cache_root/role, indices=range(count), workers=2, chunk_size=2)
    packed = [PackedRows(role, cache_root=cache_root, allow_pilot=True)
              for role in ('train', 'dev_monitor500')]
    for rows, cache in zip(direct, packed):
        for i, original in enumerate(rows):
            cached = cache[i]
            assert original['member'] == cached['member']
            for key in original['graph']:
                assert torch.equal(original['graph'][key], cached['graph'][key]), key
            for key in ('geometry', 'b2b', 'p2b', 'pins'):
                assert torch.equal(original['target'][key], cached['target'][key]), key
            for key in ('wire_anchor', 'area_anchor'):
                assert original['target'][key] == cached['target'][key]
            graph, targets = collate([original], args.device)
            candidate = (decode(targets[0]['geometry'][None], graph)[0]*graph['scale'][0]).cpu().double().numpy()
            c, s = np.asarray(original['case'].solver_inputs[5]), np.asarray(original['case'].solver_inputs[6])
            candidate[c[:, :2].any(1), 2:] = s[c[:, :2].any(1), 2:]
            candidate[c[:, 1] != 0] = s[c[:, 1] != 0]
            assert evaluate_case(original['case'], candidate.tolist(), 0)['metrics'] == evaluate_case(cached['case'], candidate.tolist(), 0)['metrics']
    graph, targets = collate(direct[0]+direct[1], args.device)
    arm_results = {}
    for arm in ARMS:
        seed_all(0); model = GeometryModel(192 if arm == 'curriculum_large' else 96).to(args.device)
        measured = {}
        for epoch in ((1, 20) if arm != 'supervised_small' else (1,)):
            values = losses(model(graph), graph, targets, arm, epoch,
                            include_quality=arm != 'supervised_small')
            params = list(model.parameters()); w = weights(arm, epoch)
            norms = {k: gradient_norm(values[k], params) for k in ('supervision', 'quality_proxy', 'overlap') if k in values}
            values['total'].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            measured[str(epoch)] = {'weights': w, 'losses': {k: float(v.detach()) for k, v in values.items()},
                                    'gradient_norms': norms,
                                    'weighted_gradient_norms': {k: v*w[{'quality_proxy':'quality'}.get(k, k)] for k, v in norms.items()}}
            model.zero_grad(set_to_none=True)
        arm_results[arm] = {'parameters': sum(p.numel() for p in model.parameters()), 'finite_backward': True, 'diagnostics': measured}
    assert selection({'quality_proxy': 2., 'overlap': .3, 'supervision': 999.}) == 5.
    # LR transitions complete before early stop; tiny cumulative improvements
    # are judged against the last meaningful baseline, not the last epoch.
    p = torch.nn.Parameter(torch.tensor(1.)); opt = torch.optim.AdamW([p], lr=1e-4)
    control = PlateauController(opt)
    for epoch in range(1, 23):
        control.step(1., epoch)
        assert not control.should_stop()
    control.step(1., 23)
    assert control.should_stop() and control.drops == 2 and opt.param_groups[0]['lr'] == 2.5e-5

    config = argparse.Namespace(arm='curriculum_small', output=args.output/'controller', device=args.device,
        seed=0, max_hours=.1, max_epochs=1, batch_size=2, resume=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Postprocessing is forbidden during training and validation')
    with patch('experiments.E16_reference_curriculum.evaluation.quality_monitor', forbidden), \
         patch('experiments.E16_reference_curriculum.evaluation.optimize', forbidden), \
         patch('experiments.E16_reference_curriculum.evaluation.polish', forbidden):
        run(config, implementation_check=True, stop_after_steps=1, check_rows=packed)
        first = torch.load(config.output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        assert first['status'] == 'user_stop' and first['progress']['cursor'] == 2
        config.resume = True
        run(config, implementation_check=True, check_rows=packed)
    final = torch.load(config.output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert final['finished'] and final['progress']['epoch'] == 1 and final['progress']['step'] == 2
    assert final['elapsed_seconds'] >= first['elapsed_seconds']
    assert final['identity']['sources'] == sources()
    save(args.output/'results.json', {'passed': True, 'unique_members': [r['member'] for r in records['train']+records['dev']],
        'count': 5, 'module_counts': [r['n'] for r in records['train']+records['dev']],
        'stratification_checks': {'equal': q, 'limited': limited, 'insufficient_rejected': rejected,
                                  'same_seed_reproducible': True, 'different_seed_changes_selection': True},
        'candidate_scan_public_only': True, 'sealed_test_denied': seal_ok,
        'synthetic_full_registry_counts_and_historical_test_exclusion': True,
        'cache_values_and_scoring_identical': True, 'cache_pause_resume': True,
        'arms': arm_results, 'controller_stop_resume': True, 'lr_three_tiers_and_early_stop': True,
        'postprocessing_calls': 0, 'test_read': False, 'formal_training_started': False,
        'elapsed_seconds': time.perf_counter()-started})
    print('PASS: five unique train/dev cases; dataset, cache, three arms and stop/resume; no test or postprocessing', flush=True)


if __name__ == '__main__':
    main()
