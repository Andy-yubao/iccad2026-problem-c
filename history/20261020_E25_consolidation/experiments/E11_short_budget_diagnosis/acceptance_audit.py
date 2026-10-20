"""Five actual decoder-oracle failures: inspect before/after canonicalization."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

from .run import (ROOT, HERE, TEACHERS, sha, save, np, torch, LayoutReader,
                  workset_manifest, _build_case, prepare_problem, topology, contact_tree, score)
from .representation import decoder_oracle
from .backend.conic_geometry import step
from .backend.reduction import reduce_edges
from .backend.frozen.packing import _dimensions
from .backend.frozen.canonical_geometry import canonicalize
from .backend.frozen.solver import hard_valid
from .backend.observe import invalid_reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); begin = time.perf_counter(); torch.set_num_threads(2)
    result = json.loads((args.representation_run/'results.json').read_text(encoding='utf-8'))
    assert result['status'] == 'completed' and result['trial_count'] == 200
    previous = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.representation_run/'cases').glob('*.json'))]
    selected = [r['panel_index'] for r in previous if next(t for t in r['trials'] if t['arm']=='decoder_oracle_relations')['metrics']['cost'] >= 5]
    assert len(selected) == 5
    args.output.mkdir(parents=True, exist_ok=False)
    config = {'command': [sys.executable, *sys.argv], 'selected_workers': selected,
              'selection': 'All five decoder_oracle_relations cases with saved Q>=5, no additional members.',
              'hypothesis': 'Canonicalization may invalidate an otherwise hard-legal native candidate before acceptance.',
              'representation_results_sha256': sha(args.representation_run/'results.json'),
              'parent_elapsed_seconds': result['shared_elapsed_seconds'], 'shared_cap_seconds': 900,
              'first_step_seconds': 1., 'step_elastic': 100000., 'threads': 2,
              'local_test_read': False, 'training': False,
              'scope': 'Targeted numerical counterexample audit; not a performance benchmark or extra generalization panel.',
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in HERE.rglob('*.py')}}
    save(args.output/'config.json', config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob('*.py')): archive.write(path, path.relative_to(ROOT).as_posix())
    teachers = {r['case_id']: r for r in map(json.loads, TEACHERS.read_text(encoding='utf-8').splitlines())}
    reader = LayoutReader(workset_manifest(), 'dev_quality'); rows = []
    for index in selected:
        if result['shared_elapsed_seconds']+time.perf_counter()-begin >= 900:
            raise RuntimeError('Shared cumulative budget exhausted')
        raw = reader.case(index)
        case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6], raw.reference_positions, raw.stored_metrics)
        teacher = np.asarray(teachers[raw.case_id]['variants']['conic_robust']['final']['positions'])
        proposal = decoder_oracle(teacher, case.solver_inputs); p = prepare_problem(*case.solver_inputs)
        dims = _dimensions(p, proposal[:, 2]/proposal[:, 3], True)
        state = np.column_stack((proposal[:, :2], *dims))
        full = topology(teacher, p.constraints)
        guide = {'edges': full, 'contacts': contact_tree(teacher, p, full)}
        candidate, stats = step(p, state, reduce_edges(full, p.n), seconds=1., conservative=True,
                                elastic=100000., quadratic_wire=True, diagonal_wire=True, guidance=guide)
        row = {'case_id': raw.case_id, 'panel_index': index, 'step_stats': stats, 'variants': {}}
        if candidate is not None:
            original = candidate.copy(); canonical = canonicalize(candidate, p)
            for name, value in [('before_canonical', original), ('after_canonical', canonical)]:
                cleaned = value.copy(); movable = ~p.preplaced
                xy = cleaned[:, :2]; xy[(xy < 0)&(xy >= -1e-9)&movable[:, None]] = 0.
                info = {'raw_hard_valid': hard_valid(value, p), 'raw_reasons': invalid_reasons(value, p),
                        'after_same_tiny_negative_clip_hard_valid': hard_valid(cleaned, p),
                        'after_clip_reasons': invalid_reasons(cleaned, p), 'positions': cleaned.tolist()}
                if hard_valid(cleaned, p):
                    info['scoring'] = score(case, cleaned, stats['seconds'])
                row['variants'][name] = info
            row['canonical_max_abs_change'] = float(np.max(np.abs(canonical-original)))
        rows.append(row)
        print(index, {k: v['after_same_tiny_negative_clip_hard_valid'] for k, v in row['variants'].items()}, flush=True)
        save(args.output/'results.json', {'status': 'completed' if len(rows)==5 else 'running',
             'elapsed_seconds': time.perf_counter()-begin,
             'shared_elapsed_seconds': result['shared_elapsed_seconds']+time.perf_counter()-begin,
             'cases': rows})


if __name__ == '__main__': main()
