"""Read only original/new saved dev500; no model, solver or test reads."""
from pathlib import Path
import ast,csv,hashlib,json
import numpy as np
ROOT=Path.cwd();RUN=ROOT/'experiments/E101_final_c5_solver/runs/20261103_final64k_seed0';OUT=Path(__file__).parent
ARCHIVE=ROOT/'history/20261103_final_delivery_consolidation/experiments'
OLD=ARCHIVE/'E97_gpu_hybrid_solver/runs/20261102_round2_candidate5_countweighted_seed0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
old=[read(OLD/'evaluation/terminal'/f'{i:04d}.json') for i in range(500)]
new=[read(RUN/'evaluation/dev_monitor500/cases'/f'{i:04d}.json') for i in range(500)]
assert all(x['member']==y['member'] and x['n']==y['n'] for x,y in zip(old,new))
n=np.array([x['n'] for x in old]);w=np.exp((n-120)/12);w/=w.sum()
def components(m):
    wire=.5*max(0,m['hpwl_gap']);area=.5*max(0,m['area_gap']);soft=m['cost']-1-wire-area
    assert abs(m['cost']-(1+wire+area)*np.exp(2*m['violations_relative']))<1e-8
    total=m['total_soft_violations'];soft=max(0.,soft)
    return dict(wire=wire,area=area,soft=soft,Q=m['cost'],boundary_proportional=soft*m['boundary_violations']/max(1,total),grouping_proportional=soft*m['grouping_violations']/max(1,total),mib_proportional=soft*m['mib_violations']/max(1,total))
def aggregate(rows):
    terms=[components(r['metrics']) for r in rows]
    return {k:float(w@np.array([r[k] for r in terms])) for k in terms[0]}
delta=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(old,new)]);order=np.argsort(-(w*delta));paired=[]
for i,(x,y) in enumerate(zip(old,new)):
    a,b=x['metrics'],y['metrics'];paired.append(dict(index=i,n=int(n[i]),old_Q=a['cost'],new_Q=b['cost'],weighted_delta_Q=float(w[i]*delta[i]),old_wire=a['hpwl_gap'],new_wire=b['hpwl_gap'],old_area=a['area_gap'],new_area=b['area_gap'],old_boundary=a['boundary_violations'],new_boundary=b['boundary_violations'],old_grouping=a['grouping_violations'],new_grouping=b['grouping_violations'],old_mib=a['mib_violations'],new_mib=b['mib_violations']))
result=dict(count=500,paired_members_identical=True,denominators_identical=all(all(x['metrics'][k]==y['metrics'][k] for k in ('hpwl_baseline','bbox_area_baseline','max_possible_violations')) for x,y in zip(old,new)),old=aggregate(old),new=aggregate(new),paired=dict(improved=int((delta<-1e-8).sum()),same=int((abs(delta)<=1e-8).sum()),worse=int((delta>1e-8).sum())),weighted_Q_delta=float(w@delta),top15_weighted_regressions=[paired[i] for i in order[:15]],top5_delta_share=float((w*delta)[order[:5]].sum()/(w@delta)),soft_allocation_note='Exact total soft excess includes multiplicative geometric interaction. Boundary/group/MIB split is proportional accounting, not a causal marginal effect.',no_new_scoring_or_solving=True,test_used=False)
save(OUT/'saved_quality_comparison.json',result)
with (OUT/'paired_final500.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
history={}
for label,path in [('old_L',ARCHIVE/'E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/progress.json'),('old_F',OLD/'progress.json'),('new_L',RUN/'L/progress.json'),('new_F',RUN/'F/progress.json')]:
    p=read(path);best=p['best_epoch'];records=[]
    for r in p['history']:
        row=dict(epoch=r['epoch'])
        if 'train_loss' in r:row.update(train=r['train_loss'],dev=r['dev_loss'])
        else:
            for role in ('train_monitor','dev_monitor'):row[role]={k:v for k,v in r[role].items() if isinstance(v,(int,float)) and k!='count'}
        records.append(row)
    history[label]=dict(best_epoch=best,initial=records[0],best=next(r for r in records if r['epoch']==best),history=records)
save(OUT/'histories.json',history)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
newbase=ROOT/'experiments/E101_final_c5_solver';oldbase=ARCHIVE/'E97_gpu_hybrid_solver'
repair_changes=[p.relative_to(newbase/'repair').as_posix() for p in (newbase/'repair').rglob('*.py') if sha(p)!=sha(oldbase/'repair85_one_thread'/p.relative_to(newbase/'repair'))]
def fns(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'));return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
functions=['base_per_case','raw_terms'];a=fns(oldbase/'round2_group_freedom.py',functions);b=fns(newbase/'loss_f.py',functions)
integrity=dict(repair_source_changes=repair_changes,repair_options_equal=read(OLD/'evaluation/frozen.json')['repair']==read(newbase/'plan.json')['repair'],F_base_public_loss_AST_equal=a==b,L_loss_source_equal=all(sha(newbase/'placer'/p)==sha(ARCHIVE/'E74_collision_32k_posttrain'/p) for p in ['loss.py','losses.py','topology.py','loss_config.py','batches.py','ops.py']),network_source_equal=all(sha(p)==sha(oldbase/'nn85'/p.name) for p in (newbase/'nn').glob('*.py')),test_used=False)
save(OUT/'algorithm_integrity.json',integrity)
print(json.dumps({k:v for k,v in result.items() if k not in ('top15_weighted_regressions','soft_allocation_note')},ensure_ascii=False,indent=2));print('top5',result['top15_weighted_regressions'][:5]);print('integrity',integrity)
