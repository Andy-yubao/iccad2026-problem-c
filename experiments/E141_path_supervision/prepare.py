"""Create independent three-arm sources; never launch formal workers."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'experiments/E140_frontend_gb4'
MODULE='experiments.E141_path_supervision'
SCOPE='E141.path_supervision.v1.seed0'
ARMS=('e119_blockage','e119_layerwise','e119_combined')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def replace_function(source,name,replacement):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Already prepared')
    names=('__init__.py','augmentation.py','batches.py','captured_projection.py','cuda_step.py','data.py',
        'features.py','group_tree.py','ports.py','resident.py','schedule.py','simple_loss.py','step_control.py',
        'original_model.py','e119_model.py','model.py','io.py','train.py','run_queue.py','loss.py')
    records={}
    for src in [PARENT/n for n in names]+list((PARENT/'evaluation').rglob('*.py'))+[PARENT/'evaluation/budget.json']:
        rel=src.relative_to(PARENT);dst=HERE/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        source=src.read_text(encoding='utf-8')
        for a,b in [('experiments.E140_frontend_gb4',MODULE),('E140.frontend_gb4.v1.seed0',SCOPE),('E140','E141')]:source=source.replace(a,b)
        dst.write_text(source,encoding='utf-8')
        records[str(rel)]=dict(parent=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    plan=json.loads((PARENT/'plan.json').read_text())
    for key in ('early_gb_multiplier','enhanced_layers','unchanged_full_layers','overlap_omission','comparison_controls'):
        plan.pop(key,None)
    arms={}
    for arm in ARMS:
        dense=arm!='e119_blockage';layers=list(range(1,13)) if dense else [3,6,9,12]
        arms[arm]=dict(layer_weights={str(k):(.8 if k==12 else .2/(len(layers)-1)) for k in layers},
            full_loss_layers=[3,6,9,12],no_overlap_layers=[k for k in layers if k not in (3,6,9,12)],
            blockage=arm!='e119_layerwise',blockage_multiplier=1.,grouping_multiplier=1.,boundary_multiplier=1.)
    plan.update(experiment='E141 three experiments: obstruction / layerwise / combined',revision=SCOPE,
        authorization='2026-11-16 explicit user: prepare and background start three batch64 experiments; automatically evaluate fixed dev500 after each. Confirmed E134 stride3 base; S1..S12 dense with .2/11 auxiliaries and .8 final; no G/Bx4.',
        status='authorized_preparation',arms=arms,batch_size=64,monitor_batch_size=64,
        initialization='scratch seed0; unchanged E119 initialization, no warm start',new_candidates=3,
        loss='Original per-layer terms plus public swept-strip obstruction in arms1/3; no fourth powers. See DESIGN.md.',
        comparison='Three authorized arms; historical E134 stride3 batch32 only contextual, not matched batch64 control.',
        queue='Sequential train -> unique own-Loss-best freeze -> original fixed dev500 evaluation, for each of three arms; stop/error blocks successors.',
        terminal_evaluation='automatic sole own-Loss best, same fixed dev500 and E98.depth_budget.v1; no test',
        preflight='Synthetic obstruction geometry/gradient/invariance checks; original five cached train cases, all-arm objective checks and batch64 guarded disposable updates; checkpoint/resume equivalence.',
        batch_tail='36000 mod64=32: exact eager FP32 tail sharing Adam/projection; no dropped/repeated members',
        engine='FP32 deterministic CUDA graph64 and eager tail32; original AdamW/projection/guards; unchanged LR',
        budget='Each arm4h/200epoch caps; single seed0, no extra candidates or automatic retry')
    save(HERE/'plan.json',plan)
    ev=json.loads((PARENT/'evaluation/plan.json').read_text())
    ev.update(revision=SCOPE+'.dev500',models={},authorization='Explicit user authorized automatic fixed dev500 after each successful training arm.')
    save(HERE/'evaluation/plan.json',ev)
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=list(ARMS),
        automatic_evaluation=True,automatic_retry=False,test_read=False,authorization=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None,formal_training_started=False))
    save(HERE/'provenance.json',dict(parent=str(PARENT.relative_to(ROOT)),copied_originals=records,
        changes=['Independent source copy','batch64 plus separate tail32 graph','registered obstruction and dense supervision','three-arm comparison']))
    path=HERE/'train.py';s=path.read_text(encoding='utf-8')
    s=s.replace('RUN = None','BATCH_SIZE = 64\nRUN = None')
    s=s.replace('def build_capture(model, optimizer, pack, projection=None):','def build_capture(model, optimizer, pack, projection=None, batch_size=BATCH_SIZE):')
    s=s.replace('for i in range(32)','for i in range(batch_size)')
    s=s.replace('len(rows), 32','len(rows), BATCH_SIZE').replace('start+32','start+BATCH_SIZE')
    s=s.replace("    current_ids = None","    current_ids = None\n    captures = {}")
    needle="        restore_rng(capture_rng)\n        if cap.projection.steps"
    s=s.replace(needle,"        captures[BATCH_SIZE] = cap\n        tail = len(train) % BATCH_SIZE\n        if tail:\n            captures[tail] = build_capture(model, optimizer, pack, batch_size=tail)\n            captures[tail].projection = cap.projection\n        restore_rng(capture_rng)\n        if cap.projection.steps")
    s=s.replace("progress['cursor']+32","progress['cursor']+BATCH_SIZE")
    s=s.replace('                pack.batch(current_ids, out=cap.inputs)','                cap = captures[len(current_ids)]\n                pack.batch(current_ids, out=cap.inputs)')
    s=s.replace('samples_per_second=32*','samples_per_second=BATCH_SIZE*')
    path.write_text(s,encoding='utf-8')
    path=HERE/'run_queue.py';s=path.read_text(encoding='utf-8')
    s=replace_function(s,'write_comparison', '''def write_comparison():
    rows=[];panel=None
    for arm in ARMS:
        path=OWNER/'runs'/f'{arm}_seed0/evaluation_no_mib_dev500'
        result=read(path/'results.json')
        if result['status']!='complete' or result['count']!=500:raise ValueError('Incomplete dev500')
        records=[read(path/f'cases/{i:04d}.json') for i in range(500)]
        members=[r['member'] for r in records]
        if panel is None:panel=members
        elif panel!=members:raise ValueError('Panel differs')
        rows.append(dict(result,grouping_violations=sum(r['metrics']['grouping_violations'] for r in records),
            boundary_violations=sum(r['metrics']['boundary_violations'] for r in records),
            all_soft_satisfied=sum(r['metrics']['total_soft_violations']==0 for r in records)))
    save(OWNER/'comparison.json',dict(cells=rows,batch=64,test_read=False,
        note='Single seed, adaptive stopping; own Loss not ranked across objectives; historical batch32 only context.'))
    lines=['# E141 three-arm comparison','',
        '|Arm|Mixed Q|Hard legal|Grouping|Boundary|All soft satisfied|Mean/P95 ms|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['arm']}|{r['mixed_Q']:.6f}|{r['hard_legal']}/500|{r['grouping_violations']}|{r['boundary_violations']}|{r['all_soft_satisfied']}/500|{r['mean_complete_seconds']*1000:.1f}/{r['p95_complete_seconds']*1000:.1f}|")
    (OWNER/'comparison.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
''')
    s=s.replace("common_final_dev_loss=entry['dev_monitor']['original_final_total']","common_final_dev_loss=entry['dev_monitor']['original_final_total']")
    s=s.replace("own_new_final_dev_loss=entry['dev_monitor']['final_total']","own_new_final_dev_loss=entry['dev_monitor']['common_final_total']")
    path.write_text(s,encoding='utf-8')
    print('Independent sources prepared; formal training remains disabled.')

if __name__=='__main__':main()
