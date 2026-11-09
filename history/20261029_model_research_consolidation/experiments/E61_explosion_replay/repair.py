"""One targeted continuation through the real fault, with graph-step projection."""
import copy
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import time
import numpy as np
import torch
from .io import ROOT,HERE,save,sha,checkpoint
from .cache import load_rows
from .resident import ResidentRows
from .model import GeometryModel
from .cuda_step import CapturedStep
from .step_control import GraphStepControl
from .replay import restore_rng,rng_state

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();source=ROOT/'experiments/E57_capacity_data_diagnosis/runs/20261026_capacity16000_geometry160_seed0/checkpoints/best.pt'
    digest=sha(source);cp=torch.load(source,map_location='cpu',weights_only=False)
    folder=HERE/'runs/20261027_graph_step_projection_seed0';folder.mkdir(exist_ok=False)
    progress=copy.deepcopy(cp['progress']);first_step=progress['step'];indices=cp['identity']['subset_indices']
    rows=load_rows('train',indices);pack=ResidentRows(rows,'cuda',120)
    model=GeometryModel(**cp['model_config']).cuda().train();model.load_state_dict(cp['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);optimizer.load_state_dict(cp['optimizer'])
    hidden=[];handle=model.head.register_forward_pre_hook(lambda m,a:hidden.append(a[0]))
    restore_rng(cp['rng']);captured=CapturedStep(model,optimizer,pack.batch(list(range(32))),'geometry160');handle.remove();last_hidden=hidden[-1]
    control=GraphStepControl(model,.01)
    fault=torch.load(HERE/'runs/20261027_seed0_replay/first_abnormal_before_update.pt',map_location='cpu',weights_only=False)
    reverse={v:i for i,v in enumerate(indices)};fault_ids=torch.as_tensor([reverse[i] for i in fault['event']['global_train_indices']],device='cuda')
    records=[];probes=[];status='running'
    def probe():
        rng=rng_state();pack.batch(fault_ids,out=captured.inputs);restore_rng(fault['rng_before_batch']);captured.graph.replay()
        result={'step':progress['step'],'loss':float(captured.values['total']),'hidden_max':float(last_hidden.detach().abs().max()),'grad_norm':float(captured.norm)}
        restore_rng(rng);probes.append(result);return result
    def store(reason):
        checkpoint(folder/'latest.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'model_config':cp['model_config'],
            'rng':rng_state(),'progress':progress,'scheduler':cp['scheduler'],'source_identity':cp['identity'],
            'source_checkpoint_sha256':digest,'source_elapsed_seconds':cp['elapsed_seconds'],
            'elapsed_seconds':time.perf_counter()-started,'status':reason,'graph_step_control':control.summary()})
        save(folder/'progress.json',{'status':reason,'step':progress['step'],'continued_updates':progress['step']-first_step,
            'trace':records,'fixed_fault_probes':probes,'control':control.summary(),'elapsed_seconds':time.perf_counter()-started,'test_read':False})
    config={'source':str(source),'source_sha256':digest,'start_step':first_step,'target_step':8000,'original_fault_step':6278,
        'change':'Only graph-local joint weight/bias AdamW displacement is projected to <=1% of pre-step parameter norm',
        'reason':'Exact fault interventions show graph-parameter displacement dominates amplification; gradient clip1 does not bound Adam displacement.',
        'predeclared_ratio':.01,'ratio_selection':'One fixed conservative relative-step trust region; no dev sweep',
        'no_graph_input_normalization':True,'architecture_loss_batch_precision_lr_unchanged':True,
        'lr_scope':'Replay ends at original monitor-block16; scheduler first becomes eligible at15 and cannot make3 bad checks by16. LR stays original1e-3.',
        'purpose':'Targeted stability continuation, not checkpoint selection or deployed-model quality evaluation','max_wall_seconds':3600,
        'test_read':False,'source_hashes':{p.name:sha(p) for p in HERE.glob('*.py')}}
    save(folder/'config.json',config)
    state=json.loads((HERE/'state.json').read_text(encoding='utf-8'));state.update(status='running',stage='graph_step_projection_validation',
        active_children=[{'pid':os.getpid(),'stage':'repair_validation','output':str(folder.relative_to(HERE))}],
        next_action='Do not duplicate running repair. Once results.json exists, review whether original fault and later2000 updates stay healthy. No large training until accepted.')
    save(HERE/'state.json',state)
    print(json.dumps({'initial_fault_probe':probe()}),flush=True);store('prepared')
    while progress['step']<8000:
        if (HERE/'runs/STOP').exists() or time.perf_counter()-started>3600:
            status='resource_or_user_stop';break
        if progress['order'] is None:
            progress['order']=np.random.default_rng(progress['epoch']).permutation(len(rows));progress['cursor']=0
        ids=progress['order'][progress['cursor']:progress['cursor']+32]
        pack.batch(torch.as_tensor(ids,device='cuda'),out=captured.inputs);captured.graph.replay()
        loss=float(captured.values['total']);norm=float(captured.norm);hidden_max=float(last_hidden.detach().abs().max())
        if not bool(captured.finite) or loss>5 or norm>1000 or hidden_max>1000:
            status='repair_failed';records.append({'step':progress['step']+1,'loss':loss,'grad_norm':norm,'hidden_max':hidden_max});break
        control.before();optimizer.step();control.after()
        progress['seen'][ids]=True;progress['step']+=1;progress['cursor']+=32
        if progress['cursor']==len(progress['order']):progress['epoch']+=1;progress['cursor']=0;progress['order']=None
        if progress['step']%100==0:
            records.append({'step':progress['step'],'loss':loss,'grad_norm':norm,'hidden_max':hidden_max})
            q=probe();print(json.dumps({'train':records[-1],'fixed_fault':q}),flush=True)
        if progress['step']%500==0:store('running')
    if status=='running':status='repair_validation_complete'
    probe();store(status);assert sha(source)==digest
    summary={'status':status,'source_step':first_step,'final_step':progress['step'],'updates':progress['step']-first_step,
        'original_fault_step_passed':progress['step']>=6278,'fixed_fault_probes':probes,'control':control.summary(),
        'max_logged_training_loss':max([r['loss'] for r in records],default=None),'elapsed_seconds':time.perf_counter()-started,
        'source_unchanged':True,'test_read':False}
    save(folder/'results.json',summary)
    state=json.loads((HERE/'state.json').read_text(encoding='utf-8'));state.update(status='awaiting_repair_review',stage=status,active_children=[],
        repair_results=str((folder/'results.json').relative_to(HERE)),next_action='Read fault_probe.json and repair results. Validate the root mechanism and absence of forward-input normalization. If repair accepted, freeze one comparable64k/160x12 seed0 run using the same fixed1% graph Adam-step projection; keep all other E59 original-G settings. Failure requires inspecting its exact event, not blind extra candidates.')
    state['diagnostic_wall_seconds']=state.get('diagnostic_wall_seconds',0)+time.perf_counter()-started
    save(HERE/'state.json',state);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
