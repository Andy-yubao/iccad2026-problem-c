"""Prepare the independent 2x2 experiment. Never starts formal training."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'experiments/E119_global_initial_generator/variants/public_priority_v1'
NODE=ROOT/'experiments/E120_node12_collision'
ARMS=('e119_stride3','e119_stride2','e120_stride3','e120_stride2')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Sources already prepared')
    provenance={}
    names=('__init__.py','augmentation.py','batches.py','captured_projection.py','cuda_step.py','data.py',
           'features.py','group_tree.py','ports.py','resident.py','schedule.py','simple_loss.py','step_control.py','original_model.py')
    for name in names:
        shutil.copy2(BASE/name,HERE/name)
        provenance[name]=dict(source=(BASE/name).relative_to(ROOT).as_posix(),sha256=sha(BASE/name),modified=False)
    assert sha(BASE/'original_model.py')==sha(NODE/'original_model.py')
    for src,name in ((BASE/'initial_model.py','e119_model.py'),(NODE/'model.py','node12_model.py')):
        shutil.copy2(src,HERE/name)
        provenance[name]=dict(source=src.relative_to(ROOT).as_posix(),sha256=sha(src),modified=False)
    (HERE/'evaluation/repair').mkdir(parents=True)
    for src in (BASE/'evaluation/repair').rglob('*.py'):
        dst=HERE/'evaluation/repair'/src.relative_to(BASE/'evaluation/repair')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        provenance[dst.relative_to(HERE).as_posix()]=dict(source=src.relative_to(ROOT).as_posix(),sha256=sha(src),modified=False)
    for name in ('__init__.py','solver.py','budget.json'):
        shutil.copy2(BASE/'evaluation'/name,HERE/'evaluation'/name)
        provenance['evaluation/'+name]=dict(source=(BASE/'evaluation'/name).relative_to(ROOT).as_posix(),sha256=sha(BASE/'evaluation'/name),modified=False)
    loss=(BASE/'loss.py').read_text(encoding='utf-8')
    start=loss.index('def per_case(');end=loss.index('\ndef configuration(',start)
    loss=loss[:start]+'''def per_case(states,g,phase,objective=DEFAULT):
    from .loss_config import layer_weights
    if states.ndim!=4 or states.shape[0]!=12:raise ValueError('Expected twelve original readouts')
    weights=layer_weights(phase)
    values={k:base_per_case(states[k-1],g,objective) for k in weights}
    out={name:sum(weights[k]*values[k][name] for k in weights) for name in values[12]}
    out.update(final_total=values[12]['total'],original_final_total=values[12]['original_total'])
    for k in values:out[f'layer{k:02d}_total']=values[k]['total']
    return out

'''+loss[end:]
    (HERE/'loss.py').write_text(loss,encoding='utf-8')
    io=(BASE/'io.py').read_text(encoding='utf-8')
    io=io.replace("files = list(HERE.rglob('*.py'))", "files = [p for p in HERE.rglob('*.py') if 'preparation_diagnostics' not in p.parts and 'runs' not in p.parts]")
    io=io.replace("('plan.json', 'provenance.json', 'README.md')", "('plan.json', 'provenance.json', 'README.md')")
    (HERE/'io.py').write_text(io,encoding='utf-8')
    train=(BASE/'train.py').read_text(encoding='utf-8')
    train=train.replace('E119.public_priority.v1.seed0','E134.supervision_interval.v1.seed0')
    train=train.replace("global RUN, ARM", "global RUN, ARM, MODEL")
    train=train.replace("ARM = args.arm", "ARM = args.arm\n    MODEL = dict(MODEL, architecture=ARM.split('_')[0])")
    train=train.replace("    plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))\n    seed_all", "    plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))\n    diagnostic = getattr(args, 'diagnostic', None)\n    if diagnostic:\n        plan = dict(plan, train_count=160, maximum_seconds=600, maximum_epochs=10, diagnostic_only=True)\n    seed_all")
    train=train.replace("plan['parameters'] or", "plan['parameters'][ARM.split('_')[0]] or")
    train=train.replace("return requested[0] or (OWNER/'STOP').exists() or (RUN/'STOP').exists()", "return requested[0] or (OWNER/'STOP').exists() or (RUN/'STOP').exists() or (OWNER/'PAUSE_REQUEST').exists()")
    train=train.replace("old_signal = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))", "old_signal = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))\n    old_break = signal.signal(signal.SIGBREAK, lambda *_: requested.__setitem__(0, True)) if hasattr(signal, 'SIGBREAK') else None")
    train=train.replace("        train = load_rows('train', halted=halted, beat=beat)", "        train = load_rows('train', plan['preflight_train_indices'], halted=halted, beat=beat) if diagnostic else load_rows('train', halted=halted, beat=beat)\n        if diagnostic and train is not None:train=(train*32)[:160]")
    train=train.replace("        dev = load_rows('dev_monitor500', halted=halted, beat=beat) if train is not None else None", "        dev = train[:5] if diagnostic and train is not None else load_rows('dev_monitor500', halted=halted, beat=beat) if train is not None else None")
    train=train.replace("train_records = records('train')", "train_records = [dict(member=list(r['member'])) for r in train] if diagnostic else records('train')")
    train=train.replace("dev_records = records('dev_monitor500')", "dev_records = [dict(member=list(r['member'])) for r in dev] if diagnostic else records('dev_monitor500')")
    train=train.replace("or len(dev) != 500", "or len(dev) != (5 if diagnostic else 500)")
    train=train.replace("if {r['member'][0] for r in train} & {r['member'][0] for r in dev}:", "if not diagnostic and {r['member'][0] for r in train} & {r['member'][0] for r in dev}:")
    train=train.replace("monitor_ids = [lookup[tuple(r['member'])] for r in records('train_monitor500')]", "monitor_ids = list(range(5)) if diagnostic else [lookup[tuple(r['member'])] for r in records('train_monitor500')]")
    train=train.replace("        cap = build_capture(model, optimizer, pack, previous['projection'] if previous else None)", "        capture_rng = rng_state()\n        cap = build_capture(model, optimizer, pack, previous['projection'] if previous else None)\n        restore_rng(capture_rng)")
    train=train.replace("                progress['step'] += 1", "                progress['step'] += 1\n                if diagnostic and progress['step'] == (2 if diagnostic == 'pause' else 3):\n                    signal.raise_signal(signal.SIGINT)")
    train=train.replace("signal.signal(signal.SIGINT, old_signal)", "signal.signal(signal.SIGINT, old_signal)\n        if old_break is not None:signal.signal(signal.SIGBREAK, old_break)")
    # Diagnostic runs never overwrite the experiment's formal per-arm status.
    train=train.replace("save(OWNER/f'state_{ARM}.json'", "save((RUN.parent if diagnostic else OWNER)/f'state_{ARM}.json'")
    (HERE/'train.py').write_text(train,encoding='utf-8')
    plan=json.loads((BASE/'plan.json').read_text(encoding='utf-8'))
    plan.update(experiment='E134 E119/E120 x supervision interval2/3',revision='E134.supervision_interval.v1.seed0',
        status='preparation_only',authorization='2026-11-13 user confirmed four cells; prepare only and provide launch command; automatic original dev500 postprocessing; Ctrl+C checkpoint/resume required',
        architecture={'e119':'original two-global + pair-bias 12-layer model','e120':'original two-global + Node12 Q/K 12-layer model'},
        parameters={'e119':712275,'e120':725619},trainable_parameters={'e119':712275,'e120':725619},
        arms={a:dict(auxiliary_layers=[3,6,9] if a.endswith('3') else [2,4,6,8,10],auxiliary_total=.2,terminal_layer=12,terminal_weight=.8) for a in ARMS},
        initialization='scratch seed0; same architecture initialization exact across supervision intervals; no historical checkpoint warm start',
        loss='fixed public-priority v1: original terms + public F/P dimensions weight1 and P center weight1; interval-specific layers, auxiliary total.2 and terminal.8',
        comparison='Four cells compared within this run. Historical E119 Q~1.02 is not a reliable reference or required target.',
        resume='Ctrl+C pauses at accepted update/case boundary. Restore model/Adam/RNG/permutation/cursor/seen/LR/earlystop/projection; cumulative budget never reset. Evaluation resumes committed case prefix.',
        preflight='Only original five cached training members; disposable per-cell guarded update, Ctrl+C checkpoint/resume equivalence tests; no formal run/dev/test payloads or postprocessing',
        queue='sequential four cells: train -> frozen own-Loss best -> original dev500 postprocess for each; explicit --start/--resume only',
        new_candidates=4,automatic_retry=False)
    save(HERE/'plan.json',plan)
    evalplan=json.loads((BASE/'evaluation/plan.json').read_text(encoding='utf-8'))
    evalplan.update(revision='E134.supervision_interval.v1.seed0.dev500',authorization='2026-11-13 user: automatic postprocessing after four experiments',models={})
    save(HERE/'evaluation/plan.json',evalplan)
    save(HERE/'provenance.json',dict(copied=provenance,engine_parent=str(BASE.relative_to(ROOT)),
        changes=['four local model/loss cells','pause-aware serial controller','complete checkpoint and evaluation continuation','protected diagnostic-only resume test']))
    save(HERE/'authorization.json',dict(scope='E134.supervision_interval.v1.seed0',training_enabled=False,allowed_arms=[],
        prepared_arms=list(ARMS),automatic_evaluation=True,test_read=False,automatic_retry=False,
        note='Prepared only. Explicit user launch command --start creates the activation receipt; no automatic background launch.'))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None,formal_training_started=False))
    print('Independent sources and frozen four-cell plan prepared; training disabled')

if __name__=='__main__':main()
