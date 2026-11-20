"""Prepare one independent experiment; never launch formal training."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'experiments/E134_supervision_interval'
OLD = 'E134.supervision_interval.v1.seed0'
NEW = 'E139.delayed_overlap.v1.seed0'
OLD_MODULE = 'experiments.E134_supervision_interval'
NEW_MODULE = 'experiments.E139_delayed_overlap'
ARM = 'e119_delayed_overlap'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    if (HERE/'provenance.json').exists():
        raise FileExistsError('Already prepared; never overwrite a registered experiment')
    names = ('__init__.py', 'augmentation.py', 'batches.py', 'captured_projection.py',
             'cuda_step.py', 'data.py', 'features.py', 'group_tree.py', 'ports.py',
             'resident.py', 'schedule.py', 'simple_loss.py', 'step_control.py',
             'original_model.py', 'e119_model.py', 'io.py', 'train.py', 'run_queue.py')
    copied = {}
    mapping = {OLD_MODULE: NEW_MODULE, OLD: NEW}
    evaluation = list((PARENT/'evaluation').rglob('*.py'))+[PARENT/'evaluation/budget.json']
    for source in [PARENT/name for name in names]+evaluation:
        name = source.relative_to(PARENT).as_posix()
        destination = HERE/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_text(encoding='utf-8')
        for old, new in mapping.items():
            content = content.replace(old, new)
        if name == 'evaluation/run.py':
            content = content.replace("'# E134 '", "'# E139 '")
        if name == 'run_queue.py':
            content = content.replace('Foreground four-cell queue.', 'Foreground single-arm queue.')
            content = content.replace('All four cells and original dev500 postprocessing completed.',
                                      'E139 training and original dev500 postprocessing completed.')
            content = content.replace('# E134 四组结果', '# E139 延后重叠监督结果')
            content = content.replace('只比较本轮四组，不以历史E119 Q≈1.02作为可靠标杆。',
                                      '复用E134对照另行比较；不以历史E119 Q≈1.02作为可靠标杆。')
        destination.write_text(content, encoding='utf-8')
        # Cache producers and all unchanged primitives retain exact parent bytes.
        if content == source.read_text(encoding='utf-8'):
            shutil.copyfile(source, destination)
        copied[name] = dict(parent=source.relative_to(ROOT).as_posix(), sha256=sha(source))

    loss = (PARENT/'loss.py').read_text(encoding='utf-8')
    start = loss.index('def per_case(')
    end = loss.index('\ndef configuration(', start)
    loss = loss[:start]+'''TOTAL_COMPONENTS = ('reference_weighted', 'overlap_weighted', 'wire_compact_weighted',
                    'grouping_weighted', 'boundary_weighted', 'public_shape_weighted',
                    'public_preplaced_position_weighted')


def supervised_per_case(state, g, layer, objective=DEFAULT):
    from .loss_config import SUPERVISED_LAYERS, NO_OVERLAP_LAYERS
    if layer not in SUPERVISED_LAYERS:
        raise ValueError(layer)
    out = base_per_case(state, g, objective)
    out['full_total'] = out['total']
    out['full_overlap_weighted'] = out['overlap_weighted']
    if layer in NO_OVERLAP_LAYERS:
        # Omit the entire collision group: penetration, mean load and max excess.
        # Rebuild from retained terms instead of subtracting large float totals.
        out['overlap_weighted'] = torch.zeros_like(out['overlap_weighted'])
        out['total'] = sum(out[name] for name in TOTAL_COMPONENTS if name != 'overlap_weighted')
    return out


def per_case(states, g, phase, objective=DEFAULT):
    from .loss_config import layer_weights
    if states.ndim != 4 or states.shape[0] != 12:
        raise ValueError('Expected twelve original readouts S1..S12')
    weights = layer_weights(phase)
    values = {k: supervised_per_case(states[k-1], g, k, objective) for k in weights}
    out = {name: sum(weights[k]*values[k][name] for k in weights) for name in values[12]}
    out.update(final_total=values[12]['total'], original_final_total=values[12]['original_total'])
    for k, value in values.items():
        out[f'layer{k:02d}_total'] = value['total']
        out[f'layer{k:02d}_full_total'] = value['full_total']
        out[f'layer{k:02d}_overlap_raw'] = value['overlap_total']
        out[f'layer{k:02d}_overlap_weighted'] = value['overlap_weighted']
    return out

'''+loss[end:]
    (HERE/'loss.py').write_text(loss, encoding='utf-8')
    (HERE/'model.py').write_text('''"""Original E119, unchanged; no latent replanning or corruption branch."""
from .e119_model import GeometryModel as E119


def GeometryModel(architecture='e119', **configuration):
    if architecture != 'e119':
        raise ValueError(architecture)
    return E119(**configuration)
''', encoding='utf-8')

    plan = json.loads((PARENT/'plan.json').read_text(encoding='utf-8'))
    weights = {'2': .05, '6': .05, '8': .05, '10': .05, '12': .8}
    controls = []
    for arm in ('e119_stride3', 'e119_stride2'):
        path = PARENT/'runs'/f'{arm}_seed0/evaluation_no_mib_dev500/results.json'
        controls.append(dict(arm=arm, result=path.relative_to(ROOT).as_posix(), results_sha256=sha(path)))
    plan.update(experiment='E139 delayed overlap supervision', revision=NEW,
        status='preparation_only', authorization='2026-11-15 user: prepare experiment and training entry only; do not start training',
        architecture={'e119': 'unchanged E119 two-global + pair-bias 12-layer model'},
        parameters={'e119': 712275}, trainable_parameters={'e119': 712275},
        arms={ARM: dict(layer_weights=weights, auxiliary_layers=[2, 6, 8, 10],
            no_overlap_layers=[2, 6], full_loss_layers=[8, 10, 12], auxiliary_total=.2,
            terminal_layer=12, terminal_weight=.8)},
        initialization='scratch seed0; exact E134 E119 initialization; no historical checkpoint',
        loss='0.05*(L2_without_overlap + L6_without_overlap + L8_full + L10_full) + 0.8*L12_full',
        overlap_omission='Only direct overlap group removed at S2/S6; raw collision diagnostics retained. No geometry bypass, detach, channel masking or draft overwrite.',
        data_migration='Explicit continuation of current 36k supervision study, reusing E134 members/cache and monitoring panels; no new sampling',
        comparison='Reuse frozen E134 stride3 and stride2 results; changed layer schedule and collision placement are joint changes, no causal isolation or multi-seed reliability claim',
        comparison_controls=controls, new_candidates=1,
        preflight='Original five cached training cases only; objective and gradient checks plus disposable guarded update; inherited continuation engine proof. No formal training/dev/test/postprocessing.',
        queue='Prepared single arm: explicit --start -> training -> sole own-Loss-best freeze -> original dev500 evaluation; never auto-launch during preparation')
    save(HERE/'plan.json', plan)
    evalplan = json.loads((PARENT/'evaluation/plan.json').read_text(encoding='utf-8'))
    evalplan.update(revision=NEW+'.dev500', models={},
        authorization='Prepared only; evaluation follows future explicit run_queue --start, never during preparation')
    save(HERE/'evaluation/plan.json', evalplan)
    save(HERE/'provenance.json', dict(parent=PARENT.relative_to(ROOT).as_posix(),
        parent_preflight_sha256=sha(PARENT/'preflight.json'), identity_mapping=mapping,
        copied_originals=copied,
        changes=['single unchanged E119 arm', 'S2/S6 direct collision group omitted only',
                 'S2/S6/S8/S10 weights .05 each; S12 .8', 'queue descriptions and final report text only']))
    save(HERE/'authorization.json', dict(scope=NEW, training_enabled=False, allowed_arms=[],
        prepared_arms=[ARM], automatic_evaluation=True, test_read=False, automatic_retry=False,
        note='Preparation only. Future explicit user invocation of run_queue --start activates this queue; assistant must not start now.'))
    save(HERE/'queue_state.json', dict(status='prepared', completed=[], active=None, formal_training_started=False))


if __name__ == '__main__':
    main()
