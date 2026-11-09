"""Create independent E77 sources; never modify completed experiment algorithms."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'experiments/E76_bounded_refinement'
new=ROOT/'experiments/E77_compact_refinement'
assert not new.exists()
new.mkdir()
for p in old.rglob('*'):
    rel=p.relative_to(old)
    if 'runs' in rel.parts or '__pycache__' in rel.parts or not p.is_file():continue
    if p.suffix not in ('.py','.txt'):continue
    if p.name in ('diagnose_failure.py','widen.py','preflight.py','initialization.py','model.py'):continue
    out=new/rel;out.parent.mkdir(parents=True,exist_ok=True)
    text=p.read_text(encoding='utf-8')
    # These three files are exact source-model semantics, authenticated later.
    if p.name not in ('base_model.py','corrector.py','features.py','batches.py'):
        text=text.replace('E76_bounded_refinement','E77_compact_refinement').replace('E76','E77')
        text=text.replace('20261029_frozen_e74_corrector_seed0','20261029_e76_compact_corrector_seed0')
    out.write_text(text,encoding='utf-8')
(new/'source_model.py').write_bytes((old/'model.py').read_bytes())
p=json.loads((old/'provenance.json').read_text(encoding='utf-8'))
source='experiments/E76_bounded_refinement/runs/20261029_frozen_e74_corrector_seed0/checkpoints/best.pt'
sha='1a5651a1e2bf8e9c5f556b4e836548731bd7fbe147a22bc00377c83f61528ba9'
import hashlib,torch
for name in ('best','latest'):
    path=old/f'runs/20261029_frozen_e74_corrector_seed0/checkpoints/{name}.pt'
    p['protected_checkpoints_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
state=torch.load(ROOT/source,map_location='cpu',weights_only=False)
p.update(experiment=new.name,source_experiment='experiments/E76_bounded_refinement',
    initialization_checkpoint=source,initialization_sha256=sha,source_best_epoch=29,source_best_step=29000,
    source_completed_epochs=30,source_completed_elapsed_seconds=1569.3399876999902,
    pretraining_elapsed_seconds=p['pretraining_elapsed_seconds']+state['elapsed_seconds'],
    copy_note='Independent copy. Freeze complete E76, project into frozen E74 public-predicted frame, add zero-init local corrector and project again. Same fixed physical Loss. No cross-experiment private imports.')
p['pretraining_history'].append({'experiment':'E76','selected_checkpoint_seconds':state['elapsed_seconds'],'epoch':29,'step':29000})
(new/'provenance.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan={'experiment':new.name,'seed':0,'train':32000,'dev_monitor':500,'batch_size':32,'max_hours':2,'max_epochs':200,
    'fixed_loss':'Unchanged E76 G+2I+5E, same backward/monitors/full-dev/selection',
    'source':'E76 own-Loss best29, source and its E74 backbone permanently frozen/eval',
    'architecture':'Project E76 into E74 predicted frame; new width96/message32 zero-init local corrector then ordinary differentiable projection into same frame',
    'frame':'Frozen E74 predictions from public inputs only, NOT reference coordinates or labels. Applies to raw draft only; flash unchanged.',
    'fresh_residual_bound':'.15*sqrt(public area) per center axis and .15 log-aspect BEFORE final shape/center projection',
    'initial_function':'Independent projection of decoded E76 into E74 frame; raw canonical state need not equal E76. No claim of exact unprojected warm start.',
    'hypothesis':'E76 overlap gain accompanies expansion in499/500. Geometric frame prevents this shortcut; added local stage tries internal rearrangement.',
    'limits':'Frame can be infeasible for legal packing; clipped directions have ordinary zero gradients; no hard-legal guarantee. Same architecture/data source exposure as E76; combined change, not isolated causal attribution.',
    'optimizer':'Fresh AdamW only new corrector, clip1, decay.0003, original four LR tiers/early-stop; inherited training recorded separately',
    'selection':'own fixed dev500 Loss only; epoch0 eligible; terminal Q once, no checkpoint reselection',
    'budget_role':'Sixth and final formal candidate under initial6-candidate cap; total training remains below18h',
    'test_read':False}
(new/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Correct inherited metadata which did not describe E76 physical objective accurately.
path=new/'train.py';text=path.read_text(encoding='utf-8')
text=text.replace('E77.frozen_e74_bounded_corrector.v1','E77.frozen_e76_frame_corrector.v1')
text=text.replace('E74 encoder frozen and eval; corrector-only dropout .05 and AdamW weight decay .0003','Entire E76 frozen/eval; new corrector dropout .05/decay .0003; hard public-predicted E74 frame')
text=text.replace('continuous collision-gated direction weight2 plus positive box expansion weight.5','G+2 true intersection/public total area+5 positive log-box expansion squared')
text=text.replace('corrector-only Adam; whole encoder frozen; output correction bounded.15','new-corrector Adam; E76 frozen; preprojection residual bounded.15 and final E74 frame')
path.write_text(text,encoding='utf-8')
path=new/'render_drafts.py';text=path.read_text(encoding='utf-8').replace('frozen E74 bounded correction','frozen E76 internal correction in public-predicted E74 frame');path.write_text(text,encoding='utf-8')
# E76 final evidence ledger: immutable raw/Q outputs are read only.
study_path=ROOT/'experiments/E72_topology_posttrain/study.json'
study=json.loads(study_path.read_text(encoding='utf-8'))
entry=study['run_ledger'][-1]
entry.update(status='early_stop_complete_audited',weighted_Q=1.017693062733588,baseline_promoted=False,
    sources_unchanged=True,box_expands_vs_E74=499,box_comparison_count=500)
study['E76_decision']={'weighted_Q':1.017693062733588,'weighted_Q_delta_vs_E42':-.008166571281647307,
    'worker_resampling95':[-.01202106094521793,-.004259477123389307],
    'intersection_dev2000':.027333868680964348,'intersection_improved_vs_E42':1999,
    'bbox500_E76_E74':[1.0504034293500337,1.0367825401622788],
    'box_expands_vs_E74':499,'decision':'Retain but do not promote: human compactness objection. E77 frozen source + E74 public-predicted frame + learned internal correction.',
    'audit':'experiments/E76_bounded_refinement/runs/completion/inspection.json'}
study.update(current_experiment=new.name,current_stage='implementation_preflight',controller_exec_session=None,
    next_decision='Validate projected source, hard frame, frozen source and exact resume on same five train cases; only then start final sixth candidate')
study['last_check']={'utc':datetime.now(timezone.utc).isoformat(),'state':'E76_audited_E77_preparing','experiment':new.name}
study_path.write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Created',new)
