"""User-started combined seed0 candidate; no hourly wakeup or other continuation."""
import argparse
from datetime import datetime,timezone
import json
import os
import signal
import subprocess
import sys
from .io import ROOT,HERE,save,exclusive,sources
from .initialization import verify_protected

ARMS=('full_ports',)
OUTPUTS={arm:HERE/f'runs/20261028_{arm}_nosummary_v2_seed0' for arm in ARMS}


def commands(arm,resume=False):
    output=OUTPUTS[arm]
    train=[sys.executable,'-u','-B','-m','experiments.E68_port_tokens.train','--arm',arm,
           '--output',str(output),'--seed','0','--batch-size','32','--max-hours','3',
           '--max-epochs','200','--device','cuda','--engine','cuda_graph']
    if resume and (output/'checkpoints/latest.pt').exists():train.append('--resume')
    return [('train',train),('final',[sys.executable,'-u','-B','-m','experiments.E68_port_tokens.final',
                                    '--output',str(output),'--stage','all']),
            ('draft100',[sys.executable,'-u','-B','-m','experiments.E68_port_tokens.render_drafts','--output',str(output)])]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    if args.dry_run:
        print(json.dumps({arm:commands(arm,args.resume) for arm in ARMS},indent=2));return
    with exclusive(HERE/'runs/pipeline.lock'):
        preflight=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
        if preflight['status']!='passed' or preflight['sources']!=sources():
            raise ValueError('Current E68 source requires a passing five-case preflight')
        protected=verify_protected();state_path=HERE/'state.json'
        prior=json.loads(state_path.read_text()) if state_path.exists() else {};events=prior.get('events',[])
        def state(status,arm,stage,pid=None,**extra):
            utc=datetime.now(timezone.utc).isoformat();events.append({'status':status,'arm':arm,'stage':stage,'utc':utc})
            save(state_path,{'status':status,'arm':arm,'stage':stage,'updated_utc':utc,'controller_pid':os.getpid(),
                 'child_pid':pid,'outputs':{k:str(v.relative_to(HERE)) for k,v in OUTPUTS.items()},'events':events,
                 'formal_seeds':[0],'automation_enabled':False,'test_read':False,**extra})
        def stopped(output=None):return (HERE/'runs/STOP').exists() or (output is not None and (output/'STOP').exists())
        old_handler=signal.signal(signal.SIGINT,lambda *_:(HERE/'runs/STOP').touch())
        try:
            for arm,output in OUTPUTS.items():
                for stage,command in commands(arm,args.resume):
                    if stopped(output):state('stopped',arm,stage);return
                    if stage=='train' and (output/'results.json').exists():
                        result=json.loads((output/'results.json').read_text())
                        if result['status'] in ('early_stop','budget','max_epochs'):continue
                    log=HERE/f'runs/{arm}_{stage}.log'
                    with log.open('a',encoding='utf-8') as stream:
                        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                        state('running',arm,stage,child.pid);print(f'{arm}/{stage}: pid={child.pid}; log={log}',flush=True)
                        code=child.wait()
                    if code:raise RuntimeError(f'{arm}/{stage} exited {code}; see {log}')
                    if stopped(output):state('stopped',arm,stage);return
                    if stage=='train':
                        result=json.loads((output/'results.json').read_text())
                        if result['status'] not in ('early_stop','budget','max_epochs'):state('stopped',arm,stage);return
                    if stage=='final' and not (output/'terminal/results.json').exists():raise RuntimeError('Final evaluation incomplete')
                    state('stage_complete',arm,stage)
            command=[sys.executable,'-u','-B','-m','experiments.E68_port_tokens.report']
            with (HERE/'runs/report.log').open('a',encoding='utf-8') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);state('running','all','report',child.pid)
                if child.wait():raise RuntimeError('E68 report failed; see runs/report.log')
            state('complete','all','complete',report='runs/review.md',baseline_promoted=False,
                  next_action='User review; no additional runs, seeds, automation, or experiment continuation')
        except BaseException as error:
            state('failed',locals().get('arm','all'),locals().get('stage','controller'),error=repr(error));raise
        finally:
            signal.signal(signal.SIGINT,old_handler)
            assert verify_protected()==protected


if __name__=='__main__':main()
