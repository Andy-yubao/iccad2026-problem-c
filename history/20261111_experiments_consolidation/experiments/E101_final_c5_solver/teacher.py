"""Frozen original C5 teachers; new frozen L supplies the F input drafts."""
import argparse,gc,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
from .common import HERE,RUN,PLAN,read,save,sha,seed,check_stop,stopped,utc
from .data import Data,Packer,public,manifest,resolve
from .model import DeploymentModel
from .nn.base_model import decode
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks

def worker_init():
    seed()
    from .repair.canonical_native import warmup
    warmup()

def worker(job):
    index,inputs,draft=job
    from .repair.solver import optimize
    positions,trace=optimize(draft,inputs,**PLAN['repair'])
    hard=independent_hard_checks(positions.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
    if not hard['is_feasible'] or not (positions[:,:2]>=0).all():raise RuntimeError('Hard-illegal teacher')
    return index,positions,bool(trace['row_retained']),trace['seconds']

class PublicRows:
    def __init__(self,parent):
        self.m=manifest(parent);self.lookup={i:e for e in self.m['parts'] for i in e['indices']};self.path=None;self.rows=None
    def get(self,index):
        entry=self.lookup[index]
        if self.path!=entry['path']:
            p=torch.load(resolve(entry['path']),map_location='cpu',weights_only=False,mmap=True)
            assert p['indices']==entry['indices'] and p['identity']==self.m['part_identities'][entry['identity_key']]
            self.rows=dict(zip(p['indices'],p['rows']));self.path=entry['path']
        inputs=self.rows[index]['case'].solver_inputs
        return (int(inputs[0]),*[np.asarray(v).copy() for v in inputs[1:]])

@torch.inference_mode()
def run(resume=False):
    seed();begin=time.perf_counter();baseline=HERE/'assets/c5_baseline.pt';assert sha(baseline)==PLAN['source_sha256']
    original=torch.load(baseline,map_location='cpu',weights_only=False);model=DeploymentModel(**original['model_config']).cuda().eval();model.load_state_dict(original['model']);del original
    new_l=DeploymentModel(**model_config(model)).placer.cuda().eval()
    lpath=RUN/'L/checkpoints/best.pt';saved=torch.load(lpath,map_location='cpu',weights_only=False);new_l.load_state_dict(saved['model']);del saved
    binding=dict(baseline_sha256=sha(baseline),new_L_sha256=sha(lpath),plan_sha256=sha(HERE/'plan.json'),sources=read(RUN/'sources.json'),labels='C5 full public-only solver -> center/logshape nominal-area reconstruction as original C5; actual XYWH also saved',test_read=False)
    root=RUN/'teacher';root.mkdir(exist_ok=True)
    if (root/'config.json').exists():assert read(root/'config.json')==binding
    else:save(root/'config.json',binding)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for role in ('train','dev_monitor500'):
            check_stop();data=Data(role);parent=data.parent;folder=root/parent;folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            full=len(data.arrays['n']);arrays={}
            for k,shape,dtype in [('base',(full,120,3),np.float32),('teacher',(full,120,3),np.float32),('usable',(full,),np.float32),('actual_positions',(full,120,4),np.float64),('done',(full,),np.bool_)]:
                path=folder/(k+'.npy');arrays[k]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
            rows=PublicRows(parent);batch=Packer(data,supervision=False);pending={};count=int(arrays['done'][data.indices].sum());seconds=0.;fallbacks=0
            def collect():
                nonlocal count,seconds,fallbacks
                done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for f in done:
                    j,scale,n=pending.pop(f);index,p,fb,sec=f.result();assert index==j
                    arrays['teacher'][j]=0;arrays['actual_positions'][j]=0
                    arrays['teacher'][j,:n]=np.c_[(p[:,:2]+p[:,2:]/2)/scale,np.log(p[:,2]/p[:,3])].astype(np.float32)
                    arrays['actual_positions'][j,:n]=p;arrays['usable'][j]=not fb;arrays['done'][j]=True;count+=1;seconds+=sec;fallbacks+=int(fb)
                    if count%128==0 or count==len(data):
                        for a in arrays.values():a.flush()
                        save(root/'progress.json',dict(role=role,completed=count,total=len(data),worker_solve_seconds_this_invocation=seconds,fallbacks_this_invocation=fallbacks,elapsed_seconds=time.perf_counter()-begin,test_read=False));print('Teacher',role,count,'/',len(data),flush=True)
            for start in range(0,len(data),32):
                check_stop();ids=[i for i in range(start,min(start+32,len(data))) if not arrays['done'][data.indices[i]]]
                if not ids:continue
                host,count_batch=batch.gpu(ids);g=public(host);old=model(g);physical=(decode(old,g)*g['scale'][:,None,None]).cpu().double().numpy();base=new_l(g).cpu().numpy();scales=g['scale'].cpu().numpy()
                for k,i in enumerate(ids):
                    check_stop()
                    while len(pending)>=8:collect();check_stop()
                    j=int(data.indices[i]);n=int(data.arrays['n'][j]);inputs=rows.get(j);p=physical[k,:n].copy();c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
                    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
                    arrays['base'][j]=base[k];future=pool.submit(worker,(j,inputs,p));pending[future]=(j,float(scales[k]),n)
                del host,g,old,physical,base;gc.collect()
            while pending:check_stop();collect()
            for a in arrays.values():a.flush()
            assert bool(arrays['done'][data.indices].all());save(folder/'complete.json',dict(count=len(data),binding=binding,usable=int(arrays['usable'][data.indices].sum()),test_read=False))
    save(root/'results.json',dict(status='complete',elapsed_seconds=time.perf_counter()-begin,train=64000,dev_monitor=500,test_read=False))

def model_config(model):
    # Immutable selected C5 architecture; no widening or new token inputs.
    return dict(width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=192,message_width=64,max_correction=.15,cooperative=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    try:run(args.resume)
    except BaseException as e:save(RUN/'teacher/error.json',dict(error=repr(e),updated_utc=utc(),no_automatic_retry=True));raise
