"""Create a single independent scratch experiment, preserving historical runs."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENGINE=ROOT/'experiments/E141_path_supervision'
LOSS_PARENT=ROOT/'experiments/E139_delayed_overlap'
ARM='e142_dual_geometry'
SCOPE='E142.dual_geometry.v1.seed0'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def replace_function(s,name,replacement):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=s.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Prepared sources already exist')
    names=('__init__.py','augmentation.py','batches.py','captured_projection.py','cuda_step.py','data.py',
        'features.py','group_tree.py','ports.py','resident.py','schedule.py','step_control.py',
        'original_model.py','e119_model.py','io.py','train.py','run_queue.py')
    copied={}
    sources=[ENGINE/n for n in names]+list((ENGINE/'evaluation').rglob('*.py'))+[ENGINE/'evaluation/budget.json']
    sources += [LOSS_PARENT/n for n in ('loss.py','loss_config.py','simple_loss.py')]
    for src in sources:
        parent=LOSS_PARENT if src.parent==LOSS_PARENT else ENGINE
        rel=src.relative_to(parent);dst=HERE/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        s=src.read_text(encoding='utf-8')
        s=s.replace('experiments.E141_path_supervision','experiments.E142_dual_geometry').replace('E141.path_supervision.v1.seed0',SCOPE).replace('E141','E142')
        if rel.name=='loss_config.py':s=s.replace("'e119_delayed_overlap'",repr(ARM))
        dst.write_text(s,encoding='utf-8');copied[rel.as_posix()]=dict(parent=src.relative_to(ROOT).as_posix(),sha256=sha(src))
    p=HERE/'io.py';s=p.read_text().replace("'plan.json', 'provenance.json', 'README.md'","'plan.json', 'provenance.json', 'README.md', 'DESIGN.md'");p.write_text(s)
    p=HERE/'run_queue.py';s=p.read_text().replace("entry['dev_monitor']['common_final_total']","entry['dev_monitor']['final_total']")
    s=replace_function(s,'write_comparison', '''def write_comparison():
    plan=read(OWNER/'plan.json');control=plan['comparison_control']
    if sha(ROOT/control['result'])!=control['result_sha256'] or sha(ROOT/control['checkpoint'])!=control['checkpoint_sha256']:
        raise ValueError('Historical control changed')
    paths=[(ROOT/control['result']).parent,OWNER/'runs'/f'{ARMS[0]}_seed0/evaluation_no_mib_dev500']
    rows=[];members=None
    for path,label in zip(paths,('E139_existing_batch32',ARMS[0])):
        result=read(path/'results.json');assert result['status']=='complete' and result['count']==500
        cases=[read(path/f'cases/{i:04d}.json') for i in range(500)]
        panel=[r['member'] for r in cases]
        if members is None:members=panel
        elif panel!=members:raise ValueError('Different dev panel')
        rows.append(dict(result,arm=label,grouping_violations=sum(r['metrics']['grouping_violations'] for r in cases),
            boundary_violations=sum(r['metrics']['boundary_violations'] for r in cases),
            all_soft_satisfied=sum(r['metrics']['total_soft_violations']==0 for r in cases)))
    save(OWNER/'comparison.json',dict(cells=rows,note='Same supervision/Loss/panel; changed architecture and batch32->64, historical control, single seed; not architecture-only causal comparison.'))
    lines=['# E142 dual geometry versus historical E139','',
        '|Model|Mixed Q|Hard legal|Grouping|Boundary|All soft satisfied|Mean/P95 ms|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['arm']}|{r['mixed_Q']:.6f}|{r['hard_legal']}/500|{r['grouping_violations']}|{r['boundary_violations']}|{r['all_soft_satisfied']}/500|{r['mean_complete_seconds']*1000:.1f}/{r['p95_complete_seconds']*1000:.1f}|")
    (OWNER/'comparison.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
''');p.write_text(s)
    plan=json.loads((LOSS_PARENT/'plan.json').read_text())
    control_result=LOSS_PARENT/'runs/e119_delayed_overlap_seed0/evaluation_no_mib_dev500/results.json'
    control_weight=LOSS_PARENT/'runs/e119_delayed_overlap_seed0/checkpoints/best.pt'
    plan.update(experiment='E142 dynamic module demand + relation-aware values',revision=SCOPE,status='authorized_preparation',
        authorization='2026-11-16 user approved new architecture, requested E139 supervision, prepare then background scratch training and automatic fixed dev500 evaluation. Previous correction: no continuation training.',
        arms={ARM:dict(layer_weights={'2':.05,'6':.05,'8':.05,'10':.05,'12':.8},no_overlap_layers=[2,6],full_loss_layers=[8,10,12])},
        architecture={'e142':'Original public initial two blocks and 12-layer width64 trunk; module-state encoder before QKV plus pair bias and receiver-aware pair value'},
        parameters={'e142':None},trainable_parameters={'e142':None},batch_size=64,monitor_batch_size=64,
        initialization='scratch seed0, all parameters trained; no checkpoint loading or old Adam migration',
        engine='E141 deterministic FP32 CUDA graph64 plus eager final32; original AdamW/projection/guards',
        batch_tail='562 batches64 plus one32 per full epoch; no dropped or repeated rows',
        loss='Exact E139 objective; no obstruction Loss, no G/B x4, no new proposal loss',
        dynamic_module_features=49,module_encoder_input=65,local_demand_width=64,relation_width=8,
        node_feedback='Current decoded layout/public specs only; detached geometric summaries, shared demand encoder, per-layer fusion; see DESIGN.md',
        pair_feedback='Original18 pair bias preserved; receiver-aware low-dimensional relation message added to each attention head value',
        preflight='Synthetic feature/geometry/invariance/input-boundary checks; exact E139 Loss values/gradients; original5 train cases; batch64 and tail32 guarded updates and exact tail checkpoint replay',
        comparison_control=dict(result=control_result.relative_to(ROOT).as_posix(),result_sha256=sha(control_result),
            checkpoint=control_weight.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(control_weight)),
        comparison='Historical E139 same supervision, different model and batch; not a sole-factor causal comparison',
        queue='Single train -> freeze unique own-Loss-best -> original fixed dev500 + hard checks -> Q log and historical E139 comparison',
        automatic_retry=False,new_candidates=1,automatic_queue=True,maximum_seconds=14400,maximum_epochs=200,
        terminal_evaluation='Automatic own-Loss-best on original fixed dev500, E98.depth_budget.v1, no test; [Q] summary in evaluation and queue logs')
    plan.pop('comparison_controls',None)
    save(HERE/'plan.json',plan)
    ev=json.loads((ENGINE/'evaluation/plan.json').read_text());ev.update(revision=SCOPE+'.dev500',models={},authorization=plan['authorization']);save(HERE/'evaluation/plan.json',ev)
    save(HERE/'provenance.json',dict(engine_parent=ENGINE.relative_to(ROOT).as_posix(),loss_parent=LOSS_PARENT.relative_to(ROOT).as_posix(),copied_originals=copied,
        changes=['new module summaries/demand encoding and relation-aware value network','E139 exact supervision','batch64 engine inherited E141','Q summary logging']))
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=[ARM],automatic_evaluation=True,
        automatic_retry=False,test_read=False,note=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None))
    print('Independent E142 sources prepared; formal start gated on preflight')

if __name__=='__main__':main()
