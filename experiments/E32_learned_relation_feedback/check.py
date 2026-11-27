"""Five training members only: policy interface, finite gradients, structure oracle."""
from .resources import configure
RESOURCE=configure()
import json
import time
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts
from .data import load_rows,collate
from .batches import inputs_only
from .cases import WEIGHTS,predict,seed_all
from .model import GeometryModel
from .relations import RelationPolicy,objective,encode_positions,targets
from .policy import choose,LearnedRelations
from .solver import optimize
from .io import HERE,ROOT,save,sources,snapshot
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from scoring.evaluation import evaluate_case

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4,native_reduce=True)

def main():
    started=time.perf_counter();seed_all(0);warmup();reduce_warmup()
    metadata=ReferenceLayouts('train');order=sorted(range(len(metadata)),key=lambda k:metadata.records[k]['n'])
    indices=[order[k] for k in (10,4000,8000,12000,15990)];rows=load_rows('train',indices)
    base_state=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**base_state['model_config']).eval();base.load_state_dict(base_state['model'])
    g,_=collate(rows,'cuda');public=inputs_only(g);policy=RelationPolicy().cuda()
    current=g['target'].clone();current[...,:2]+=.01*torch.randn_like(current[...,:2])*g['mask'][...,:2]
    values=objective(policy(public,current,current),g);values['per_case'].mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    with torch.no_grad():
        clean=policy(public,current,current);poison=dict(g);poison['reference']=torch.randn_like(g['reference'])*1e5;poison['target']=torch.randn_like(g['target'])*1e5
        assert torch.equal(clean,policy(inputs_only(poison),current,current))
    records=[]
    for k,row in enumerate(rows):
        case=row['case'];draft,_=predict(base,case.solver_inputs);reference=row['graph']['reference'].numpy()*float(row['graph']['scale'])
        center=reference[:,:2]+reference[:,2:]/2;extent=(reference[:,None,2:]+reference[None,:,2:])/2
        margin=np.abs(center[:,None]-center[None,:])-extent;scale=np.sqrt(np.maximum(reference[:,2:].prod(1),1e-12));s=(scale[:,None]+scale[None,:])/2
        gap=margin/np.maximum(s[...,None],1e-12)
        score=np.zeros((len(reference),len(reference),3));score[...,:2]=10*(gap>=-1e-4)
        score[...,2]=10*((np.abs(gap[...,0])<1e-4)&(gap[...,1]<-1e-3)|(np.abs(gap[...,1])<1e-4)&(gap[...,0]<-1e-3))
        def oracle(state,p,full):
            edges,tree,info=choose(state,p,full,score);return edges,tree,dict(info,oracle_diagnostic=True)
        record={'member':row['member'],'n':len(reference),'reference_used_only_for_oracle_supervision':True}
        for arm in ('baseline','oracle'):
            seed_all(k);begin=time.perf_counter();x,stats=optimize(draft,case.solver_inputs,**STEADY,relation_policy=oracle if arm=='oracle' else None)
            elapsed=time.perf_counter()-begin;record[arm]={'metrics':evaluate_case(case,x.tolist(),elapsed)['metrics'],'seconds':elapsed,'positions':x.tolist(),'trace':stats['trace']}
        records.append(record);print(record['n'],*[record[a]['metrics']['cost'] for a in ('baseline','oracle')],flush=True)
    folder=HERE/'runs/20261022_interface_check5';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():raise ValueError('Completed evidence immutable')
    identity={'sources':sources(),'members':[r['member'] for r in rows],'resources':RESOURCE,'policy':STEADY}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    report={'records':records,'finite_gradients':True,'label_poisoning_input_parity':True,'parameters':sum(p.numel() for p in policy.parameters()),
        'losses':{k:v.detach().cpu().tolist() for k,v in values.items()},'seconds':time.perf_counter()-started,'test_read':False,'oracle_not_deployable':True}
    save(folder/'results.json',report);print('Interface check passed',flush=True)

if __name__=='__main__':main()
