"""Saved replay only: relation/contact retrieval diagnostics, never a solver call."""
from .resources import configure
configure()
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_rows
from .train import batch
from .prepare import load_states
from .relations import RelationPolicy,targets
from .batches import inputs_only,decode
from .io import HERE,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Diagnostic evidence immutable')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model=RelationPolicy(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    rows=load_rows('dev_monitor500');states,_=load_states('dev');records=[]
    with torch.inference_mode():
        for start in range(0,500,32):
            selected=rows[start:start+32]
            for slot in range(3):
                g,d,c=batch(selected,states,'cuda',[slot]*len(selected));logits=model(inputs_only(g),d,c)
                allowed,usable,near,contact,grouped=targets(g);p=decode(c,g);center=p[...,:2]+p[...,2:]/2
                extent=(p[:,:,None,2:]+p[:,None,:,2:])/2;delta=(center[:,:,None]-center[:,None,:]).abs();clear=(delta-extent)/extent.clamp_min(1e-8)
                heuristic=clear.argmax(-1);predicted=logits[...,:2].argmax(-1);near=near&usable
                correct0=allowed.gather(-1,heuristic[...,None]).squeeze(-1);correct1=allowed.gather(-1,predicted[...,None]).squeeze(-1)
                uncertain=(logits[...,:2].softmax(-1).amax(-1)>=.8)&((clear[...,0]-clear[...,1]).abs()<.5)&(clear.amax(-1)<.5)&near
                local=(g['area'].sqrt()[:,:,None]+g['area'].sqrt()[:,None,:])/2
                geometry=(delta-extent).clamp_min(0).sum(-1)/local.clamp_min(1e-8)
                for k,row in enumerate(selected):
                    edge_records=[];constraints=g['constraints'][k,:,3]
                    for group in constraints.unique().tolist():
                        if group<=0:continue
                        ids=torch.where(constraints==group)[0];ii,jj=torch.tril_indices(len(ids),len(ids),-1,device='cuda');a,b=ids[ii],ids[jj]
                        if not len(a):continue
                        top=min(len(ids)-1,len(a));truth=contact[k,a,b]
                        baseline=geometry[k,a,b].argsort()[:top];learned=logits[k,a,b,2].argsort(descending=True)[:top]
                        edge_records.append({'group':group,'pairs':len(a),'top_k':top,'heuristic_true_contacts':int(truth[baseline].sum()),'learned_true_contacts':int(truth[learned].sum())})
                    records.append({'member':row['member'],'n':row['graph']['n'],'slot':slot,'near_pairs':int(near[k].sum()),'heuristic_axis_correct':int((correct0[k]&near[k]).sum()),
                        'learned_axis_correct':int((correct1[k]&near[k]).sum()),'confident_override_pairs':int(uncertain[k].sum()),'confident_correct':int((correct1[k]&uncertain[k]).sum()),'contacts':edge_records})
    den=sum(r['near_pairs'] for r in records);edge=[e for r in records for e in r['contacts']];total=sum(e['top_k'] for e in edge)
    summary={'near_axis_accuracy_heuristic':sum(r['heuristic_axis_correct'] for r in records)/max(1,den),'near_axis_accuracy_learned':sum(r['learned_axis_correct'] for r in records)/max(1,den),
        'top_k_contact_precision_heuristic':sum(e['heuristic_true_contacts'] for e in edge)/max(1,total),'top_k_contact_precision_learned':sum(e['learned_true_contacts'] for e in edge)/max(1,total),
        'states':len(records),'cases':500,'checkpoint_sha256':sha(args.checkpoint),'postprocessing_calls':0,'test_read':False}
    save(args.output,{'summary':summary,'records':records});print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
