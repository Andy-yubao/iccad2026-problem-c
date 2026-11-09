"""One inference-only raw/flash/gallery evaluation of a frozen module swap."""
from datetime import datetime,timezone
import json,os,threading,time,traceback
from .io import HERE,save,exclusive

OUTPUT=HERE/'runs/20261029_e42_frozen_e79_transfer'
def main():
    from .preflight import main as preflight
    from .final import run
    from .render_drafts import run as gallery
    begin=time.perf_counter();OUTPUT.mkdir(parents=True,exist_ok=True)
    def expired():
        save(OUTPUT/'deadline.json',{'reason':'evaluation_wall_budget','max_seconds':3600})
        (OUTPUT/'STOP').touch()
    with exclusive(HERE/'runs/pipeline.lock'):
        previous=json.loads((HERE/'state.json').read_text(encoding='utf-8')) if (HERE/'state.json').exists() else {}
        spent=previous.get('evaluation_seconds',0)
        timer=threading.Timer(max(0,3600-spent),expired);timer.daemon=True;timer.start()
        state={'status':'running_frozen_diagnostic','pid':os.getpid(),'training_updates':0,
            'started_utc':datetime.now(timezone.utc).isoformat(),'output':OUTPUT.relative_to(HERE).as_posix(),
            'test_read':False,'evaluation_seconds':spent}
        save(HERE/'state.json',state)
        try:
            if not (HERE/'runs/preflight.json').exists():preflight()
            run(OUTPUT)
            if (OUTPUT/'terminal/results.json').exists():
                state['status']='terminal_complete_gallery_running';save(HERE/'state.json',state)
                gallery(OUTPUT);state['status']='complete_pending_audit'
            else:state['status']='stopped_partial'
        except Exception as error:
            state['status']='failed';state['error']=repr(error);state['traceback']=traceback.format_exc();raise
        finally:
            timer.cancel();state['evaluation_seconds']=spent+time.perf_counter()-begin
            state['checked_utc']=datetime.now(timezone.utc).isoformat();save(HERE/'state.json',state)

if __name__=='__main__':main()
