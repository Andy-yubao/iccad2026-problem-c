"""Authorized train-only controller; no automatic evaluation or other experiment."""
import json,os,traceback
from .io import HERE,ROOT,save,exclusive
from .train import RUN,run,parser
from .preflight import verify
from .budget import global_charged_seconds

def main():
    args=parser().parse_args()
    verify()
    if (HERE/'runs/STOP').exists() or (RUN/'STOP').exists():raise ValueError('STOP present')
    if global_charged_seconds()>=64800:raise ValueError('Historical18h budget exhausted')
    with exclusive(ROOT/'experiments/E89_spatial_neural_refinement/runs/pipeline.lock'),exclusive(HERE/'runs/pipeline.lock'):
        save(HERE/'state.json',{'status':'running','pid':os.getpid(),'stage':'train_only','test_read':False})
        try:
            run(args)
            result=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
            save(HERE/'state.json',{'status':result['status'],'pid':None,'phase':result['phase'],
                 'stage':'training ended; evaluation requires explicit request','test_read':False})
        except BaseException as error:
            save(HERE/'state.json',{'status':'failed','pid':None,'error':repr(error),'traceback':traceback.format_exc(),'test_read':False})
            raise

if __name__=='__main__':main()
