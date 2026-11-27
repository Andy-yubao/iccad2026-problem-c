"""Resume original Adam/RNG and capture the first explosive transition."""
import argparse
import copy
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import random
import time
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,HERE,save,sha,checkpoint
from .cache import load_rows
from .resident import ResidentRows
from .model import GeometryModel
from .cuda_step import CapturedStep

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(0,1),default=0)
    ap.add_argument('--updates',type=int,default=500);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=ROOT/f'experiments/E57_capacity_data_diagnosis/runs/20261026_capacity16000_geometry160_seed{args.seed}'
    path=source/'checkpoints/best.pt';digest=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False)
    folder=HERE/f'runs/20261027_seed{args.seed}_replay';folder.mkdir(exist_ok=False)
    start=time.perf_counter();progress=copy.deepcopy(cp['progress']);start_step=progress['step']
    old=torch.load(source/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    historical={x['step']:x for x in old['progress']['batch_history']};del old
    indices=cp['identity']['subset_indices'];rows=load_rows('train',indices)
    pack=ResidentRows(rows,'cuda',120)
    model=GeometryModel(**cp['model_config']).cuda();model.load_state_dict(cp['model']);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);optimizer.load_state_dict(cp['optimizer'])
    restore_rng(cp['rng'])
    # Read captured tensors directly; these hooks add no operators to the training graph.
    # Capture exposes original hidden/readout tensors without parameter updates.
    hidden_tensors=[];pre_tensors=[]
    handles=[model.head.register_forward_pre_hook(lambda m,a:hidden_tensors.append(a[0])),
             model.head[0].register_forward_hook(lambda m,a,o:pre_tensors.append(o))]
    restore_rng(cp['rng']);captured=CapturedStep(model,optimizer,pack.batch(list(range(32))),'geometry160')
    for h in handles:h.remove()
    hidden_tensors=hidden_tensors[-13:];pre_tensors=pre_tensors[-13:]
    def clone_state():
        return {'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                'rng':rng_state(),'progress':copy.deepcopy({k:v for k,v in progress.items() if k not in ('history','batch_history','gradient_history')}),
                'model_config':cp['model_config'],'source_checkpoint_sha256':digest,'source_identity':cp['identity']}
    records=[];previous=None;status='bounded_replay_complete';parity=[]
    initial=clone_state()
    save(folder/'config.json',{'source':str(path),'source_sha256':digest,'seed':args.seed,'start_step':start_step,
        'updates':args.updates,'source_elapsed_seconds':cp['elapsed_seconds'],'source_config':cp['model_config'],
        'exact_sampler':'original E57 subset-index mapping and seed+monitor-block NumPy permutation','precision':'FP32',
        'torch':torch.__version__,'max_nodes':120,'source_hashes':{p.name:sha(p) for p in HERE.glob('*.py')},'test_read':False})
    del initial
    while progress['step']<start_step+args.updates:
        if (HERE/'runs/STOP').exists() or time.perf_counter()-start>3600:
            status='bounded_stop';break
        if progress['order'] is None:
            progress['order']=np.random.default_rng(args.seed+progress['epoch']).permutation(len(rows))
            progress['cursor']=0
        cursor=progress['cursor'];local_ids=progress['order'][cursor:cursor+32]
        pack.batch(torch.as_tensor(local_ids,device='cuda'),out=captured.inputs)
        before_rng=rng_state()
        captured.graph.replay()
        values={k:float(v.detach()) for k,v in captured.values.items()};norm=float(captured.norm)
        hidden=[float(t.detach().abs().max()) for t in hidden_tensors]
        pre=[{'min':float(t.detach().min()),'max':float(t.detach().max())} for t in pre_tensors]
        step=progress['step']+1
        record={'step':step,'losses':values,'grad_norm':norm,'hidden_max':hidden,'readout_preactivation':pre,
                'global_train_indices':[indices[int(i)] for i in local_ids],'elapsed_seconds':time.perf_counter()-start}
        records.append(record)
        if step in historical:
            h=historical[step];delta=abs(values['total']-h['losses']['total'])
            parity.append({'step':step,'replay':values['total'],'historical':h['losses']['total'],'abs_delta':delta,
                           'grad_delta':abs(norm-h['grad_norm'])})
        explosive=not bool(captured.finite) or values['total']>5 or norm>1000 or hidden[-1]>1000
        if explosive:
            current=clone_state();current['rng_before_batch']=before_rng;current['event']=record
            checkpoint(folder/'first_abnormal_before_update.pt',current)
            if previous is not None:checkpoint(folder/'previous_update_before.pt',previous)
            status='first_abnormal_captured'
            save(folder/'event.json',{'status':status,'event':record,'historical_parity':parity,
                'previous_checkpoint_step':previous['progress']['step'] if previous else None,
                'diagnostic_thresholds':{'loss':5,'grad_norm':1000,'final_hidden_max':1000},'test_read':False})
            print(json.dumps({'status':status,'step':step,'loss':values['total'],'grad':norm,'hidden':hidden[-1],'parity':parity}),flush=True)
            break
        previous=clone_state();previous['rng_before_batch']=before_rng;previous['batch_global_indices']=record['global_train_indices']
        previous['batch_event']=record
        optimizer.step();progress['step']=step;progress['cursor']+=32
        if progress['cursor']==len(progress['order']):
            progress['epoch']+=1;progress['order']=None;progress['cursor']=0
            # At this bound (seed0 6000->6500, seed1 10000->10500), no scheduler
            # change precedes the historical failure; monitoring consumes no RNG.
        if step%25==0:
            print(json.dumps({'step':step,'loss':values['total'],'grad':norm,'hidden':hidden[-1]}),flush=True)
            save(folder/'trace.json',{'records':records,'historical_parity':parity,'test_read':False})
    assert sha(path)==digest
    save(folder/'trace.json',{'records':records,'historical_parity':parity,'test_read':False})
    save(folder/'results.json',{'status':status,'start_step':start_step,'last_computed_step':records[-1]['step'] if records else start_step,
        'optimizer_updates':progress['step']-start_step,'source_unchanged':True,'historical_parity':parity,
        'elapsed_seconds':time.perf_counter()-start,'test_read':False})
    state=json.loads((HERE/'state.json').read_text(encoding='utf-8'));state.update(status='awaiting_root_cause_review',stage='replay_captured',active_children=[],
        replay_results=str((folder/'results.json').relative_to(HERE)),next_action='Inspect first abnormal and preceding optimizer update; establish trigger with same-state single-variable intervention before choosing repair or large training.')
    save(HERE/'state.json',state)

if __name__=='__main__':main()
