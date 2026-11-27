"""Saved-state diagnostic only; reference order is not a deployable target input.

Registered after the E32 endpoint. A mismatch to one reference can be an equally
valid alternative topology, so this analysis cannot establish failure causality.
No solver calls, fitting, selection, or additional confirmation cases.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from experiments.E32_learned_relation_feedback.resources import configure
configure()
import json
import torch
from experiments.E32_learned_relation_feedback.data import load_rows
from experiments.E32_learned_relation_feedback.prepare import load_states
from experiments.E32_learned_relation_feedback.train import batch
from experiments.E32_learned_relation_feedback.relations import RelationPolicy,targets
from experiments.E32_learned_relation_feedback.batches import inputs_only,decode
from experiments.E32_learned_relation_feedback.io import save,sha


def main():
    output=Path(__file__).with_name('signed_order_diagnostic.json')
    if output.exists():raise ValueError('Completed diagnosis is immutable')
    root=ROOT/'experiments/E32_learned_relation_feedback'
    rows=load_rows('dev_monitor500');states,_=load_states('dev');result={}
    for arm in ('replay','draft_only'):
        path=root/'runs'/f'20261022_{arm}_seed0/checkpoints/best.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=RelationPolicy(**state['model_config']).cuda().eval()
        model.load_state_dict(state['model']);counts={}
        def add(name,mask):counts[name]=counts.get(name,0)+int(mask.sum())
        with torch.inference_mode():
            for start in range(0,len(rows),32):
                selected=rows[start:start+32]
                for slot in range(3):
                    g,d,c=batch(selected,states,'cuda',[slot]*len(selected))
                    logits=model(inputs_only(g),d,c)
                    allowed,usable,near,_,_=targets(g)
                    p=decode(c,g);center=p[...,:2]+p[...,2:]/2
                    r=g['reference'];rc=r[...,:2]+r[...,2:]/2
                    signed=center[:,:,None]-center[:,None,:]
                    ref_signed=rc[:,:,None]-rc[:,None,:]
                    extent=(p[:,:,None,2:]+p[:,None,:,2:])/2
                    clear=(signed.abs()-extent)/extent.clamp_min(1e-8)
                    predicted=logits[...,:2].argmax(-1)
                    heur=clear.argmax(-1)
                    valid=usable&near
                    eligible=(logits[...,:2].softmax(-1).amax(-1)>=.8)&((clear[...,0]-clear[...,1]).abs()<.5)&(clear.amax(-1)<.5)&valid
                    eligible=eligible&(predicted!=heur)
                    for name,axis,mask in [('learned_near',predicted,valid),('heuristic_near',heur,valid),('candidate_override',predicted,eligible)]:
                        correct=allowed.gather(-1,axis[...,None]).squeeze(-1)
                        s=signed.gather(-1,axis[...,None]).squeeze(-1)
                        rs=ref_signed.gather(-1,axis[...,None]).squeeze(-1)
                        matching=(s*rs)>0
                        add(name+'_pairs',mask)
                        add(name+'_axis_correct',mask&correct)
                        add(name+'_axis_correct_order_matches',mask&correct&matching)
                        add(name+'_axis_correct_order_differs',mask&correct&~matching)
        result[arm]={'checkpoint_sha256':sha(path),'counts':counts}
    save(output,{'scope':'post-hoc saved dev_monitor500 three-state diagnosis',
        'interpretation':'A reference-valid unsigned axis may inherit a different current order. Different order can also describe a valid alternative layout; this is not a causal ablation.',
        'candidate_override_mask':'confidence and clearance heuristic, not a replay of the exact full-relation callback',
        'pairs_counted_in_both_directions':True,'arms':result,'postprocessing_calls':0,'test_read':False})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
