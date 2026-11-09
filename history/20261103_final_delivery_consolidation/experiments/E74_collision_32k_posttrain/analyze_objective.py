"""Saved E42/E73 draft geometry only; no model inference or solver calls."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .io import ROOT,HERE,save,sha

def initialize():
    global DATA
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from datasets.reference_layouts import ReferenceLayouts
    DATA=ReferenceLayouts('dev_monitor500')

def terms(p,ref,inputs):
    p=np.asarray(p);ref=np.asarray(ref);n=len(p);constraints=np.asarray(inputs[5]);specs=np.asarray(inputs[6])
    area=np.asarray(inputs[1],dtype=float).copy();locked=(constraints[:,0]>0)|(constraints[:,1]>0)
    area[locked]=specs[locked,2]*specs[locked,3]
    local=np.sqrt(area);scale=(local[:,None]+local[None,:])/2
    rc=ref[:,:2]+ref[:,2:]/2;pc=p[:,:2]+p[:,2:]/2
    rd=rc[:,None]-rc[None,:];pd=pc[:,None]-pc[None,:]
    rm=np.abs(rd)-(ref[:,None,2:]+ref[None,:,2:])/2
    near=np.linalg.norm(np.maximum(rm,0),axis=-1)<=.25*scale
    mutable=(constraints[:,1]==0);eligible=near&(mutable[:,None]|mutable[None,:])&(rm.max(-1)>=-1e-6)
    np.fill_diagonal(eligible,False)
    use_x=rm[...,0]>=rm[...,1];direction=np.sign(np.where(use_x,rd[...,0],rd[...,1]));signed=np.where(use_x,pd[...,0],pd[...,1])*direction
    extent=(p[:,None,2:]+p[None,:,2:])/2
    deficit=np.maximum(np.where(use_x,extent[...,0],extent[...,1])-signed,0)/scale
    inter=np.maximum(np.minimum(p[:,None,:2]+p[:,None,2:],p[None,:,:2]+p[None,:,2:])-np.maximum(p[:,None,:2],p[None,:,:2]),0).prod(-1)
    weights=deficit**2*eligible/n;colliding=inter>1e-10
    gated=weights*inter/np.minimum(area[:,None],area[None,:])
    wrong=(signed<0)&eligible
    return {'T':float(weights.sum()),'T_noncolliding':float((weights*~colliding).sum()),'T_colliding':float((weights*colliding).sum()),'C':float(gated.sum()),'wrong_order_pairs':int(wrong.sum()/2),'near_pairs':int(eligible.sum()/2),'intersection_fraction':float(np.triu(inter,1).sum()/area.sum())}

def analyze(payload):
    index,member,current,previous=payload;item=DATA[index]
    if list(item['member'])!=member:raise ValueError('Member mismatch')
    inputs=item['case'].solver_inputs;ref=item['positions']
    return {'index':index,'member':member,'E42':terms(previous,ref,inputs),'E73':terms(current,ref,inputs)}

def main():
    base=ROOT/'experiments/E73_wide_robust_posttrain/runs/20261029_e42_topology_seed0'
    records=json.loads((base/'terminal/results.json').read_text(encoding='utf-8'))['records']
    gallery=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'
    payload=[]
    for row in records:
        path=gallery/f"case_{row['index']:04d}.json"
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            assert old['member']==row['member'];payload.append((row['index'],row['member'],row['draft_positions'],old['positions']))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:rows=list(pool.map(analyze,payload))
    summary={model:{k:float(np.mean([r[model][k] for r in rows])) for k in rows[0][model]} for model in ('E42','E73')}
    for model in summary:summary[model]['pooled_T_noncolliding_fraction']=sum(r[model]['T_noncolliding'] for r in rows)/sum(r[model]['T'] for r in rows)
    review=json.loads((ROOT/'experiments/E73_wide_robust_posttrain/runs/review.json').read_text(encoding='utf-8'))
    initial=review['initial_monitor']['dev_monitor'];best=review['best_monitor']['dev_monitor']
    delta={k:best[k]-initial[k] for k in ('total','common_original_G','topology_separation')}
    save(HERE/'runs/objective_audit/saved100.json',{'summary':summary,'dev500_best_minus_initial':delta,'rows':rows,
        'count':len(rows),'model_inference_calls':0,'solver_calls':0,'test_read':False,'four_workers':True,
        'source_terminal_sha256':sha(base/'terminal/results.json'),'interpretation':'Post hoc objective support audit. Gated C is a prospective loss change, not a trained result.'})
    print(summary,flush=True);print('dev500 best-initial',delta,flush=True)
if __name__=='__main__':main()
