"""Read saved results only; no new model inference or postprocessing."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.E90_auxiliary_weight_placer.io import HERE, save, sha
from experiments.E90_auxiliary_weight_placer.metrics import module_overlap, union_area

RUNS = dict(E90='E90_auxiliary_weight_placer/runs/20261031_aux3_seed0',
            E87='E87_simple_supervised_placer/runs/20261030_simple_seed0',
            E74='E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0')


def main():
    geometry = {}
    monitors = {}
    member_list = None
    for name, run in RUNS.items():
        folder = ROOT/'experiments'/run
        terminal = folder/'terminal/results.json'
        data = json.loads(terminal.read_text(encoding='utf-8'))
        records = sorted(data['records'], key=lambda r: r['index'])
        members = [r['member'] for r in records]
        if member_list is None:
            member_list = members
        assert len(records) == 500 and members == member_list
        derived = []
        modules = 0
        passed = 0
        for r in records:
            p = np.asarray(r['draft_positions'], dtype=float)
            assert len(p) == r['n'] and np.all(p[:, 2:] > 0)
            v = module_overlap(p)
            area = p[:, 2:].prod(1).sum()
            bbox = np.prod((p[:, :2]+p[:, 2:]).max(0)-p[:, :2].min(0))
            void = bbox-union_area(p)
            modules += len(p)
            passed += len(p)-v['N10']
            derived.append(dict(index=r['index'], member=r['member'], n=r['n'],
                module_pass10_fraction=v['module_pass10_fraction'], N10=v['N10'],
                all_modules_pass10=v['all_modules_pass10'], max_covered_fraction=v['max_covered_fraction'],
                pair_intersection_fraction=r['draft_diagnostics']['overlap_area_fraction'],
                bbox_over_A=float(bbox/area), empty_fraction_of_bbox=float(void/bbox)))
        geometry[name] = dict(source=terminal.relative_to(ROOT).as_posix(), source_sha256=sha(terminal),
            count=500, module_count=modules, module_pass10_pooled=passed/modules,
            all_modules_pass10_count=sum(r['all_modules_pass10'] for r in derived),
            worst_module_coverage_mean=float(np.mean([r['max_covered_fraction'] for r in derived])),
            worst_module_coverage_p95=float(np.percentile([r['max_covered_fraction'] for r in derived], 95)),
            pair_intersection_fraction_mean=float(np.mean([r['pair_intersection_fraction'] for r in derived])),
            bbox_over_A_mean=float(np.mean([r['bbox_over_A'] for r in derived])),
            empty_fraction_of_bbox_mean=float(np.mean([r['empty_fraction_of_bbox'] for r in derived])),
            records=derived)
        if name in ('E90', 'E87'):
            progress = json.loads((folder/'progress.json').read_text(encoding='utf-8'))
            selected = []
            for h in progress['history']:
                m = h['dev_monitor']
                selected.append(dict(epoch=h['epoch'], step=h['step'], is_best=h['epoch'] == progress['best_epoch'],
                    **{k: m[k] for k in ('reference', 'position', 'shape', 'overlap_total', 'penetration',
                         'module_load_mean', 'module_max_excess10', 'task', 'wirelength', 'compactness',
                         'soft', 'pair_intersection_fraction')}))
            monitors[name] = dict(source=(folder/'progress.json').relative_to(ROOT).as_posix(),
                                  best_epoch=progress['best_epoch'], records=selected)
    result = dict(geometry500=geometry, fixed_dev500_monitors=monitors,
                  new_inference_calls=0, new_postprocessing_calls=0, test_read=False,
                  note='Exact module union coverage from saved real-module rectangles; pooled module denominator explicit. '
                       'Monitoring and geometry500 are different saved execution paths with small floating point differences.')
    save(HERE/'runs/20261031_aux3_seed0/inspection500.json', result)
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'records'} for k, v in geometry.items()}, indent=2))


if __name__ == '__main__':
    main()
