"""Single-candidate resumable pipeline; never reselects by terminal Q."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from .io import HERE,ROOT,save,sources,exclusive
from .initialization import verify_protected
OUTPUT=HERE/'runs/20261029_e42_topology_seed0'
STATE=HERE/'state.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    commands=[('train',[sys.executable,'-u','-B','-m','experiments.E73_wide_robust_posttrain.train','--arm','topology','--output',str(OUTPUT),'--max-hours','2','--max-epochs','200','--batch-size','32','--seed','0','--engine','cuda_graph']),
        ('final',[sys.executable,'-u','-B','-m','experiments.E73_wide_robust_posttrain.final','--output',str(OUTPUT),'--stage','all']),
        ('draft100',[sys.executable,'-u','-B','-m','experiments.E73_wide_robust_posttrain.render_drafts','--output',str(OUTPUT)]),
        ('report',[sys.executable,'-u','-B','-m','experiments.E73_wide_robust_posttrain.report'])]
    if args.resume and (OUTPUT/'checkpoints/latest.pt').exists():commands[0][1].append('--resume')
    if args.dry_run:print(json.dumps(commands,indent=2));return
    with exclusive(HERE/'runs/pipeline.lock'):
        verify_protected()
        check=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
        if check['status']!='passed' or check['sources']!=sources():raise ValueError('Passing preflight for current sources required')
        events=json.loads(STATE.read_text(encoding='utf-8')).get('events',[]) if STATE.exists() else []
        def state(status,stage,child=None,**extra):
            utc=datetime.now(timezone.utc).isoformat();events.append({'utc':utc,'status':status,'stage':stage})
            save(STATE,{'status':status,'stage':stage,'controller_pid':os.getpid(),'child_pid':child,'updated_utc':utc,'output':str(OUTPUT),'events':events,'test_read':False,**extra})
        stage='preflight'
        try:
            for stage,cmd in commands:
                if (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists():state('stopped',stage);return
                if stage=='train' and (OUTPUT/'results.json').exists():
                    old=json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                    if old['status'] in ('early_stop','budget','max_epochs'):continue
                with (HERE/f'runs/{stage}.log').open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    state('running',stage,child.pid);print(stage,child.pid,flush=True)
                    code=child.wait()
                if code:raise RuntimeError(f'{stage} failed ({code}); see runs/{stage}.log')
                if stage=='train':
                    result=json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                    if result['status'] not in ('early_stop','budget','max_epochs'):state('stopped',stage);return
                if stage=='final' and not (OUTPUT/'terminal/results.json').exists():state('stopped','evaluation_incomplete');return
                state('stage_complete',stage)
            state('complete','complete',baseline_promoted=False,report='runs/review.md')
        except BaseException as error:state('failed',stage,error=repr(error));raise
        finally:verify_protected()
if __name__=='__main__':main()
