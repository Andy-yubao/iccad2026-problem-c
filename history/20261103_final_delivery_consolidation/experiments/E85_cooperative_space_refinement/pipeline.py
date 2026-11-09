"""Detached sequential two-arm controller; no unattended architecture search."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from .io import HERE,ROOT,save,sources,exclusive
from .initialization import verify_protected
ARMS=('control','cooperative')
OUTPUTS={arm:HERE/('runs/20261030_P74_F81_'+arm+'_seed0') for arm in ARMS}
STATE=HERE/'state.json'
def now():return datetime.now(timezone.utc).isoformat()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    commands=[('initial_audit',[sys.executable,'-u','-B','-m','experiments.E85_cooperative_space_refinement.initial_audit'])]
    for arm,folder in OUTPUTS.items():
        cmd=[sys.executable,'-u','-B','-m','experiments.E85_cooperative_space_refinement.train','--arm',arm,'--output',str(folder),'--max-hours','2','--max-epochs','200','--batch-size','32','--seed','0','--engine','cuda_graph']
        if args.resume and (folder/'checkpoints/latest.pt').exists():cmd.append('--resume')
        commands.append(('train_'+arm,cmd))
    for arm,folder in OUTPUTS.items():
        commands.append(('final_'+arm,[sys.executable,'-u','-B','-m','experiments.E85_cooperative_space_refinement.final','--output',str(folder),'--stage','all']))
    for arm,folder in OUTPUTS.items():
        commands.append(('gallery_'+arm,[sys.executable,'-u','-B','-m','experiments.E85_cooperative_space_refinement.render_drafts','--output',str(folder)]))
    commands.append(('report',[sys.executable,'-u','-B','-m','experiments.E85_cooperative_space_refinement.report']))
    if args.dry_run:print(json.dumps(commands));return
    with exclusive(HERE/'runs/pipeline.lock'):
        verify_protected();check=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
        if args.resume and (HERE/'runs/lr_policy_change/validation.json').exists():
            check=json.loads((HERE/'runs/lr_policy_change/validation.json').read_text(encoding='utf-8'))
        if check['status']!='passed' or check['sources']!=sources():raise ValueError('Passing current-source preflight required')
        events=json.loads(STATE.read_text(encoding='utf-8')).get('events',[]) if STATE.exists() else []
        def state(status,stage,child=None,**extra):
            events.append(dict(utc=now(),status=status,stage=stage))
            save(STATE,dict(status=status,stage=stage,controller_pid=os.getpid(),child_pid=child,updated_utc=now(),events=events,test_read=False,**extra))
            ledger=[]
            for arm,folder in OUTPUTS.items():
                result=folder/'results.json';budget=folder/'budget.json'
                info=json.loads(result.read_text(encoding='utf-8')) if result.exists() else {}
                charged=max(info.get('elapsed_seconds',0),json.loads(budget.read_text(encoding='utf-8')).get('elapsed_seconds',0) if budget.exists() else 0)
                ledger.append(dict(arm=arm,status=info.get('status','pending_or_running'),charged_seconds=charged,output=str(folder)))
            total=sum(row['charged_seconds'] for row in ledger)
            save(HERE/'study.json',dict(phase='20261030 cooperative space',prior_training_seconds=34604.39659300004,max_new_training_seconds=14400,global_cap_seconds=64800,phase_charged_seconds=total,global_charged_seconds=34604.39659300004+total,preflight_seconds=check['seconds'],scheduler_policy_change='runs/lr_policy_change/migration.json',comparison_limit='Completed control used original protection; cooperative resumed without protection; not a pure operator-only training-policy comparison',runs=ledger,status=status,stage=stage,test_read=False))
        stage='initial'
        try:
            for stage,cmd in commands:
                if (HERE/'runs/STOP').exists():state('stopped',stage);return
                if stage.startswith('train_'):
                    folder=OUTPUTS[stage[len('train_'):]]
                    if (folder/'results.json').exists():
                        old=json.loads((folder/'results.json').read_text(encoding='utf-8'))
                        if old['status'] in ('early_stop','budget','max_epochs'):continue
                with (HERE/('runs/'+stage+'.log')).open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    state('running',stage,child.pid);print(stage,child.pid,flush=True);code=child.wait()
                if code:raise RuntimeError(stage+' failed; see its stage log')
                if stage.startswith('train_'):
                    result=json.loads((OUTPUTS[stage[len('train_'):]]/'results.json').read_text(encoding='utf-8'))
                    if result['status'] not in ('early_stop','budget','max_epochs'):state('stopped',stage);return
                if stage.startswith('final_') and not (OUTPUTS[stage[len('final_'):]]/'terminal/results.json').exists():state('stopped','evaluation_incomplete');return
                state('stage_complete',stage)
            state('complete','complete',report='runs/comparison.md',baseline_promoted=False)
        except BaseException as error:state('failed',stage,error=repr(error));raise
        finally:verify_protected()
if __name__=='__main__':main()
