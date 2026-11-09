"""One accepted-repair large run followed by its unique-best finalization."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from .io import ROOT,HERE,save,exclusive

CONTROL=HERE/'runs/20261027_control'
FOLDER=HERE/'runs/20261027_large64000_graph_step_seed0'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def health():
    import torch
    from .cache import load_rows
    from .data import collate
    from .batches import inputs_only
    from .training_model import GeometryModel
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    state=torch.load(FOLDER/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    ids=read(HERE/'runs/20261027_preflight/results.json')['train_indices'];rows=load_rows('train',ids);g,_=collate(rows,'cpu')
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model']);zeros=[]
    free=g['valid']&g['mask'][...,0].bool()
    handle=model.head[1].register_forward_hook(lambda m,a,o:zeros.append(float((o[free]==0).float().mean())))
    with torch.no_grad():out=model(inputs_only(g))
    handle.remove();hidden=float(model.last_hidden_max)
    result={'status':'passed' if hidden<1000 and zeros[-1]<.99 and bool(torch.isfinite(out).all()) else 'failed',
        'hidden_max':hidden,'final_silu_zero_fraction':zeros[-1],'unique_train_cases':5,'graph_prenorm':False,
        'train_indices':ids,'test_read':False,'limit':'Same-five-case health check plus per-update guards; not universal stability proof'}
    save(FOLDER/'health.json',result)
    if result['status']!='passed':raise RuntimeError('Completed training failed numerical-health gate')

def main():
    CONTROL.mkdir(parents=True,exist_ok=True)
    with exclusive(CONTROL/'large_controller.lock'):
        state=read(HERE/'state.json')
        if not state.get('repair_accepted'):raise RuntimeError('Root-cause repair not accepted')
        if read(HERE/'runs/20261027_preflight/results.json')['status']!='passed':raise RuntimeError('Five-case full-entry preflight required')
        deadline=datetime.fromisoformat(state['calendar_deadline_utc']).timestamp()
        def update(status,stage,child=None):
            state.update(status=status,stage=stage,controller_pid=os.getpid(),active_children=[child] if child else [],
                updated_utc=datetime.now(timezone.utc).isoformat())
            save(HERE/'state.json',state)
        def execute(stage,args):
            command=[sys.executable,'-u','-B','-m',*args]
            log_path=CONTROL/(stage+'.log')
            with log_path.open('a',encoding='utf-8') as log:
                log.write('\n'+subprocess.list2cmdline(command)+'\n');log.flush()
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update('running',stage,{'pid':child.pid,'command':command,'log':str(log_path.relative_to(HERE))})
                while child.poll() is None:
                    if time.time()>deadline or (HERE/'runs/STOP').exists():
                        FOLDER.mkdir(parents=True,exist_ok=True);(FOLDER/'STOP').touch()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError(stage+' failed; inspect '+str(log_path))
        try:
            training=read(FOLDER/'results.json') if (FOLDER/'results.json').exists() else None
            if not training or training['status'] not in ('early_stop','max_epochs','budget'):
                state['large_training_started']=True;state['large_run']=str(FOLDER.relative_to(HERE))
                command=['experiments.E61_explosion_replay.train','--output',str(FOLDER),'--arm','large_original160',
                    '--dataset-size','64000','--seed','0','--max-hours','4','--max-epochs','200','--batch-size','32',
                    '--block-steps','500','--engine','staged_cuda_graph','--deadline',str(deadline)]
                if (FOLDER/'checkpoints/latest.pt').exists():command.append('--resume')
                execute('large_train',command)
            training=read(FOLDER/'results.json')
            if training['status'] not in ('early_stop','max_epochs','budget'):raise RuntimeError('Training did not finish normally')
            health()
            for stage,command in [
                ('large_raw2000',['experiments.E61_explosion_replay.final','--output',str(FOLDER),'--stage','raw']),
                ('large_flash500',['experiments.E61_explosion_replay.final','--output',str(FOLDER),'--stage','terminal']),
                ('large_draft100',['experiments.E61_explosion_replay.render_drafts','--output',str(FOLDER)])]:
                if time.time()>deadline or (HERE/'runs/STOP').exists():raise RuntimeError('Calendar/STOP boundary reached')
                execute(stage,command)
            state['completed']=list(dict.fromkeys(state.get('completed',[])+['large_training_complete','large_health_passed','unique_best_raw2000','unique_best_flash500','same_draft100_gallery']))
            state['next_action']='Read large results/full_dev/terminal summary/health and generate final comparison withE42/E59 using saved data. Inspect3-5 existing same-case images, preserve limits and protected baselines, clean only redundant failed preflight artifacts, then deletee61 heartbeat and mark complete. Do not rerun evaluation or start extra training.'
            update('awaiting_final_review','large_outputs_complete')
        except BaseException as error:
            state['last_error']=repr(error);state['next_action']='Inspect failed stage/log and exact preserved state; do not duplicate the controller or silently restart/retrain. Keep budgets and no-prenorm requirement.'
            update('needs_diagnosis',state.get('stage'));raise

if __name__=='__main__':main()
