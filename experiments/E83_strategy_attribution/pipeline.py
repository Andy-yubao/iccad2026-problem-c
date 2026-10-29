"""Budgeted two-arm inference study with OS lock and immutable completed legs."""
import os,threading,time,traceback
from datetime import datetime,timezone
from .io import HERE,save,exclusive,sources
from .prepare import read,freeze

def now():return datetime.now(timezone.utc).isoformat()
def main():
    plan=read(HERE/'plan.json');state_path=HERE/'state.json'
    with exclusive(HERE/'runs/pipeline.lock'):
        state=read(state_path) if state_path.exists() else {}
        if state.get('status') in ('complete_pending_visual_audit','complete_audited'):
            print('Completed E83 reused; no repeated inference or scoring.',flush=True);return
        if (HERE/'runs/STOP').exists():raise RuntimeError('E83 STOP present; inspect authorization before any restart')
        started=state.get('started_utc',now());elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(started)).total_seconds()
        if elapsed>=plan['max_total_seconds']:raise RuntimeError('Original study wall budget exhausted')
        state.update(status='preflight',pid=os.getpid(),started_utc=started,last_update_utc=now(),training_updates=0,test_read=False)
        state.setdefault('arms',{});save(state_path,state)
        def expire_total():
            save(HERE/'runs/deadline.json',{'reason':'study_wall_budget','utc':now(),'max_seconds':plan['max_total_seconds']});(HERE/'runs/STOP').touch()
        timer=threading.Timer(plan['max_total_seconds']-elapsed,expire_total);timer.daemon=True;timer.start()
        try:
            p=HERE/'runs/preflight.json'
            if p.exists():
                if read(p)['status']!='passed' or read(p)['sources']!=sources():raise ValueError('Existing preflight does not match sources')
            else:
                from .preflight import main as preflight
                preflight()
            from .final import run,stopped
            for arm in plan['arms']:
                folder,snapshot,identity=freeze(arm)
                current=state['arms'].get(arm['name'],{})
                if current.get('status')=='terminal_complete':continue
                arm_started=current.get('started_utc',now());arm_elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(arm_started)).total_seconds()
                if arm_elapsed>=plan['max_seconds_per_arm']:raise RuntimeError('Original arm wall budget exhausted')
                current.update(status='evaluating',started_utc=arm_started,pid=os.getpid(),checkpoint_sha256=identity['checkpoint_sha256'])
                state['arms'][arm['name']]=current;state.update(status='evaluating',current_arm=arm['name'],last_update_utc=now());save(state_path,state)
                def expire_arm(folder=folder):
                    save(folder/'deadline.json',{'reason':'arm_wall_budget','utc':now(),'max_seconds':plan['max_seconds_per_arm']});(folder/'STOP').touch()
                arm_timer=threading.Timer(plan['max_seconds_per_arm']-arm_elapsed,expire_arm);arm_timer.daemon=True;arm_timer.start()
                try:complete=run(folder,snapshot,identity)
                finally:arm_timer.cancel()
                current.update(status='terminal_complete' if complete else 'stopped_partial',checked_utc=now(),
                    accounted_wall_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(arm_started)).total_seconds())
                save(state_path,state)
                if not complete or stopped(folder):state['status']='stopped_partial';return
            state.update(status='analysis_and_gallery',last_update_utc=now());save(state_path,state)
            from .analysis import main as analyze
            from .render_drafts import main as render
            analyze()
            if (HERE/'runs/STOP').exists():state['status']='stopped_after_evaluation';return
            render();state['status']='complete_pending_visual_audit'
        except Exception as error:
            state.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
        finally:
            timer.cancel();state.update(last_update_utc=now(),accounted_wall_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(started)).total_seconds())
            save(state_path,state)

if __name__=='__main__':main()
