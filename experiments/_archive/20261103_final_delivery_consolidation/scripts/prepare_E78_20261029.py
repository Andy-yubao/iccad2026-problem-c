"""Independent second-phase experiment, authorized by user's continuation."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'experiments/E77_compact_refinement';new=ROOT/'experiments/E78_connectivity_posttrain'
assert not new.exists();new.mkdir()
for p in old.rglob('*'):
    rel=p.relative_to(old)
    if not p.is_file() or 'runs' in rel.parts or '__pycache__' in rel.parts:continue
    if p.suffix not in ('.py','.txt'):continue
    if p.name in ('initialization.py','preflight.py'):continue
    out=new/rel;out.parent.mkdir(parents=True,exist_ok=True)
    text=p.read_text(encoding='utf-8').replace('E77_compact_refinement','E78_connectivity_posttrain').replace('E77','E78')
    text=text.replace('20261029_e76_compact_corrector_seed0','20261029_e77_connectivity_seed0')
    if p.name in ('features.py','batches.py','model.py','base_model.py','source_model.py','corrector.py','frame.py'):
        out.write_bytes(p.read_bytes())
    else:out.write_text(text,encoding='utf-8')
p=json.loads((old/'provenance.json').read_text(encoding='utf-8'))
source='experiments/E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0/checkpoints/best.pt'
for name in ('best','latest'):
    f=old/f'runs/20261029_e76_compact_corrector_seed0/checkpoints/{name}.pt'
    p['protected_checkpoints_sha256'][f.relative_to(ROOT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
p.update(experiment=new.name,source_experiment='experiments/E77_compact_refinement',
    initialization_checkpoint=source,initialization_sha256='69cfe24b7d57c38847c6bed64a91e79df533acb6481be48640fbd5f01dd06d83',
    source_best_epoch=38,source_best_step=38000,source_completed_epochs=40,
    source_completed_elapsed_seconds=6786.0241621999885,
    copy_note='Independent source copy. Exact full E77 best38 weights; freeze its E76 source and fine-tune existing21860-parameter corrector. Original frame remains. New fixed public group-connectivity and weighted b2b-wire objective; no new input labels.')
import torch
ck=torch.load(ROOT/source,map_location='cpu',weights_only=False)
p['pretraining_elapsed_seconds']+=ck['elapsed_seconds']
p['pretraining_history'].append({'experiment':'E77','selected_checkpoint_seconds':ck['elapsed_seconds'],'epoch':38,'step':38000,'note':'Includes conservative startup interruption charge, not all active computation'})
(new/'provenance.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan={'experiment':new.name,'phase':2,'phase_candidate':1,'seed':0,'train':32000,'dev_monitor':500,
    'batch_size':32,'max_hours':2,'max_epochs':200,'test_read':False,
    'authorization':'User: 长任务继续。你可以自行设计和运行实验',
    'source':'Exact E77 own-Loss best38, weights-only new fixed-Loss stage',
    'architecture':'Unchanged E77; full E76 source frozen/eval and public-predicted E74 frame. Fine-tune already-trained last corrector only, no new stage/head.',
    'fixed_loss':'Proposed G+2I+5E + .1*public_group_matrix_tree_contact + .1*positive_log_weighted_b2b_wire_ratio_squared; fixed only after saved-case diagnosis and five train-case preflight',
    'group_goal':'Connected contacts with positive tangential overlap, not fake connectivity from deep overlaps. Public groups, no reference contact topology.',
    'wire_goal':'Exact weighted b2b center Manhattan length reconstructed from public normalized adjacency and degree; reference b2b only normalizes supervised excess. Not full p2b/HPWL claim.',
    'selection':'Unique best by same fixed dev500 Loss; training/monitor/full-dev identical; Q auxiliary terminal once',
    'optimizer':'New AdamW/four original LR levels/clip1/decay.0003; old optimizer and cumulative budget preserved separately',
    'data':'Same frozen nested32k; unchanged train/dev500 and full dev2000; train500 old16k only',
    'limits':'Matrix-tree contact is a proxy, not hard grouping guarantee. Frame may be infeasible; discrete legalizer may alter topology. Combined loss change, not isolated attribution.'}
(new/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
phase={'status':'diagnosing','authorization_utc':datetime.now(timezone.utc).isoformat(),
    'authorization':'User: 长任务继续。你可以自行设计和运行实验','phase':2,
    'previous_phase':'experiments/E72_topology_posttrain/study.json','previous_phase_closed':True,
    'previous_charged_seconds':19426.817579200084,'max_new_formal_candidates':3,'max_phase_training_seconds':21600,
    'max_seconds_per_candidate':7200,'global_training_cap_seconds':64800,
    'current_experiment':new.name,'current_stage':'saved_failure_diagnosis','run_ledger':[],
    'test_read':False,'baseline_promoted':False,'wake_minutes':30}
(new/'study.json').write_text(json.dumps(phase,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
master=ROOT/'experiments/E72_topology_posttrain/study.json';m=json.loads(master.read_text(encoding='utf-8'))
m['authorized_followup_phase2']={'authorization':phase['authorization'],'utc':phase['authorization_utc'],
    'ledger':'experiments/E78_connectivity_posttrain/study.json','phase1_records_frozen':True,
    'new_self_imposed_budget':'At most3 new candidates/6h phase; each<=2h; preserve all previous charged time under18h global cap'}
master.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared',new)
