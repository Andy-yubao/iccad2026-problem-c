"""Same-state interventions on the captured actual fault batch; no new training run."""
import copy
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import time
import torch
from .io import ROOT,HERE,save,sha
from .cache import load_rows
from .batches import collate
from .model import GeometryModel
from .losses import components,WEIGHTS
from .cuda_step import CapturedStep
from .replay import restore_rng

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    folder=HERE/'runs/20261027_seed0_replay';started=time.perf_counter()
    before=torch.load(folder/'previous_update_before.pt',map_location='cpu',weights_only=False)
    after=torch.load(folder/'first_abnormal_before_update.pt',map_location='cpu',weights_only=False)
    ids=sorted(set(before['batch_global_indices']+after['event']['global_train_indices']))
    rows=load_rows('train',ids);lookup=dict(zip(ids,rows))
    def batch(indices):
        rr=[lookup[i] for i in indices]
        return collate([r['graph'] for r in rr],list(range(len(rr))),'cuda',max_nodes=120)
    previous_batch=batch(before['batch_global_indices']);fault_batch=batch(after['event']['global_train_indices'])
    model=GeometryModel(**before['model_config']).cuda().train();model.load_state_dict(before['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);optimizer.load_state_dict(before['optimizer'])
    hs=[];outs=[]
    handles=[model.head.register_forward_pre_hook(lambda m,a:hs.append(a[0])),model.head.register_forward_hook(lambda m,a,o:outs.append(o))]
    restore_rng(before['rng_before_batch']);captured=CapturedStep(model,optimizer,previous_batch,'geometry160')
    for h in handles:h.remove()
    hs=hs[-13:];out=outs[-1]
    def evaluate(theta,graph,rng):
        model.load_state_dict(theta)
        for k,v in graph.items():captured.inputs[k].copy_(v)
        restore_rng(rng);captured.graph.replay()
        with torch.no_grad():
            cc=components(out,captured.inputs);per_case=sum(w*cc[k] for k,w in WEIGHTS.items())
            return {'losses':{k:float(v) for k,v in captured.values.items()},'grad_norm':float(captured.norm),
                'hidden_max':[float(t.detach().abs().max()) for t in hs],
                'case_loss':per_case.detach().cpu().tolist(),'state_max':float(out.detach().abs().max())}
    previous_metrics=evaluate(before['model'],previous_batch,before['rng_before_batch'])
    optimizer.step()
    max_update_delta=max(float((t.detach().cpu()-after['model'][k]).abs().max()) for k,t in model.state_dict().items())
    assert max_update_delta==0.,max_update_delta
    fault_metrics=evaluate(after['model'],fault_batch,after['rng_before_batch'])
    assert fault_metrics['losses']['total']==after['event']['losses']['total']
    result={'status':'complete','fault_step':after['event']['step'],'responsible_update':before['batch_event']['step'],
        'optimizer_reconstruction_max_tensor_delta':max_update_delta,'previous_batch':previous_metrics,
        'original_fault':fault_metrics,'variants':{},'test_read':False,'new_formal_training_runs':0,
        'purpose':'Exact same actual batch/Dropout/RNG; only specified saved parameter changes differ.'}
    for alpha in (0.,.25,.5,1.):
        theta={k:a+alpha*(after['model'][k]-a) for k,a in before['model'].items()}
        result['variants'][f'update_fraction_{alpha}']=evaluate(theta,fault_batch,after['rng_before_batch'])
    for name,predicate in [
        ('revert_graph_parameters',lambda k:'.local.' in k),
        ('revert_ff_parameters',lambda k:'.ff.' in k),
        ('revert_readout_feedback_parameters',lambda k:k.startswith(('head.','state_embed.'))),
        ('revert_attention_parameters',lambda k:any(q in k for q in ('.qkv.','.output.','.bias.'))),
        ('only_graph_parameter_update',lambda k:'.local.' not in k),
    ]:
        theta={k:before['model'][k] if predicate(k) else v for k,v in after['model'].items()}
        result['variants'][name]=evaluate(theta,fault_batch,after['rng_before_batch'])
    result['after_update_on_previous_batch']=evaluate(after['model'],previous_batch,before['rng_before_batch'])
    changes=[]
    for k,a in before['model'].items():
        delta=after['model'][k]-a
        changes.append({'parameter':k,'update_norm':float(delta.norm()),'old_norm':float(a.norm()),'max_abs_delta':float(delta.abs().max())})
    result['largest_parameter_updates']=sorted(changes,key=lambda x:x['update_norm'],reverse=True)[:30]
    result['total_update_norm']=sum(x['update_norm']**2 for x in changes)**.5
    result['old_parameter_norm']=sum(x['old_norm']**2 for x in changes)**.5
    result['elapsed_seconds']=time.perf_counter()-started
    save(folder/'interventions.json',result)
    state=json.loads((HERE/'state.json').read_text(encoding='utf-8'));state.update(status='awaiting_causal_repair_review',stage='same_update_interventions_complete',active_children=[],
        next_action='Read seed0 replay/interventions.json and trace.json. Exact historical trajectory and responsible update reconstructed. Select the smallest non-prenorm repair supported by same-state evidence; verify continuation through fault before launching one comparable 64k run.')
    save(HERE/'state.json',state)
    print(json.dumps({'step':result['fault_step'],'update_parity':max_update_delta,'before':previous_metrics['losses']['total'],
        'fault':fault_metrics['losses']['total'],'variants':{k:v['losses']['total'] for k,v in result['variants'].items()},
        'after_on_previous':result['after_update_on_previous_batch']['losses']['total']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
