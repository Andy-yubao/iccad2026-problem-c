"""Freeze the selected L0 and compare it to the recorded E149 best52."""
import json
from pathlib import Path

import numpy as np

from .io import HERE, ROOT, save, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finalize(output):
    run = HERE / 'runs/e164_l0_seed0'
    result = read(run / 'results.json')
    if result['error'] or result['status'] not in ('early_stop', 'budget', 'max_epochs'):
        raise RuntimeError('Training did not finish normally')
    if result['steps'] <= 0 or result['best_epoch'] is None:
        raise RuntimeError('Training produced no usable checkpoint')
    progress = read(run / 'progress.json')
    best = next(row for row in progress['history'] if row['epoch'] == result['best_epoch'])
    checkpoint = run / 'checkpoints/best.pt'
    baseline = read(HERE / 'baseline_e149.json')
    panels = read(run / 'panels.json')
    if panels['dev_monitor_members'] != baseline['dev_monitor_members']:
        raise ValueError('E149 and E164 dev500 member identities differ')
    differences = (
        np.asarray(best['dev_monitor']['per_case_original_no_overlap'])
        - np.asarray(baseline['per_case_original_no_overlap'])
    )
    if differences.shape != (500,):
        raise ValueError('Expected exactly 500 paired development results')
    e149 = baseline['original_no_overlap']
    new = best['dev_monitor']['initial_no_overlap_total']
    freeze = dict(
        arm='e164_l0', best_epoch=result['best_epoch'],
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(checkpoint),
        selection='fixed dev500 E164 quarter-overlap objective',
        own_selection_loss=best['dev_monitor']['total'], optimizer_steps=result['steps'],
    )
    report = dict(
        frozen=freeze, training=result, train_components=best['train_monitor'],
        dev_components=best['dev_monitor'], baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
        e149_dev_original_no_overlap=e149, e164_dev_original_no_overlap=new,
        e149_dev_quarter_overlap=baseline['quarter_overlap'],
        e164_dev_quarter_overlap=best['dev_monitor']['total'],
        e149_dev_raw_overlap=baseline['raw_overlap'],
        e164_dev_raw_overlap=best['dev_monitor']['overlap_total'],
        no_overlap_improvement_fraction=(e149-new)/e149,
        better=int((differences < -1e-9).sum()),
        equal=int((np.abs(differences) <= 1e-9).sum()),
        worse=int((differences > 1e-9).sum()), test_read=False,
        note='Same original dev500 and metrics; Both used the same original train36k membership.',
    )
    save(output / 'frozen_best.json', freeze)
    save(output / 'loss_results.json', report)
    line = (f"[LOSS] E164 best_epoch={result['best_epoch']} steps={result['steps']} "
            f"dev_quarter_overlap={report['e164_dev_quarter_overlap']:.9f} "
            f"dev_E149_no_overlap={new:.9f} historical_E149={e149:.9f}")
    (output / 'loss.log').write_text(line + '\n', encoding='utf-8')
    print(line, flush=True)
    return report
