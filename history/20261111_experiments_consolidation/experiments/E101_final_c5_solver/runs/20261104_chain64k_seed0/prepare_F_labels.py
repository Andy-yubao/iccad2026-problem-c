"""Original E85 targets16k + same-source extension48k, and new E74 bases64k."""
from pathlib import Path
import sys,time,multiprocessing as mp
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from experiments.E101_final_c5_solver.common import HERE,RUN,PLAN,read,save,sha,seed,check_stop,stopped,snapshot,sources,utc
from experiments.E101_final_c5_solver.data import Data,Packer,public,manifest,resolve
from experiments.E101_final_c5_solver.train import baseline
from experiments.E101_final_c5_solver.teacher_nn.model import DeploymentModel as TeacherModel
from experiments.E101_final_c5_solver.teacher_nn.base_model import decode

BUNDLE=Path(__file__).resolve().parent
WORKSPACE=BUNDLE.parents[3]
ORIGINAL=WORKSPACE/'history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/20261101_public_teacher'
SOURCE=read(HERE/'teacher_source.json')

class PublicRows:
    def __init__(self):
        self.m=manifest('train');self.lookup={i:e for e in self.m['parts'] for i in e['indices']};self.path=None
    def get(self,index):
        e=self.lookup[index]
        if self.path!=e['path']:
            p=torch.load(resolve(e['path']),map_location='cpu',weights_only=False,mmap=True)
            assert p['identity']==self.m['part_identities'][e['identity_key']] and p['indices']==e['indices']
            self.rows=dict(zip(p['indices'],p['rows']));self.path=e['path']
        row=self.rows[index];assert list(row['member'])==self.m['members'][index]
        inputs=row['case'].solver_inputs
        return (int(inputs[0]),*[np.asarray(v).copy() for v in inputs[1:]])

def init_worker():
    torch.set_num_threads(1)
    from experiments.E101_final_c5_solver.teacher_repair.canonical_native import warmup
    warmup()

def solve(job):
    index,inputs,draft,scale=job
    from threadpoolctl import threadpool_limits
    from experiments.E101_final_c5_solver.teacher_repair.solver import optimize
    from experiments.E101_final_c5_solver.teacher_repair.backend.frozen.packing import prepare_problem
    from experiments.E101_final_c5_solver.teacher_repair.backend.frozen.solver import hard_valid
    np.random.seed(index);torch.manual_seed(index)
    with threadpool_limits(SOURCE['worker_solver_threads']):
        positions,stats=optimize(draft,inputs,**SOURCE['FULL'])
    assert hard_valid(positions,prepare_problem(*inputs)),'Original teacher failed its hard validation'
    target=np.c_[(positions[:,:2]+positions[:,2:]/2)/scale,np.log(positions[:,2]/positions[:,3])].astype(np.float32)
    assert np.isfinite(target).all()
    return index,target,float(not stats['row_retained'])

def crop(g,n):
    return {k:(v if k=='scale' else v[:,:n,:n] if k in ('adjacency','mib','group') else v[:,:n]) for k,v in g.items()}

def teacher_model(device='cuda'):
    path=HERE/'assets/teacher_E85.pt';assert sha(path)==SOURCE['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False);model=TeacherModel(**state['model_config']);model.load_state_dict(state['model'],strict=True)
    return model.to(device).eval()

def old_labels(role,data,arrays):
    m=read(ORIGINAL/role/'manifest.json');expected=16000 if role=='train' else 500
    assert m['count']==m['completed']==expected;seen=set()
    for entry in m['parts']:
        check_stop();path=ORIGINAL/entry['path'];assert sha(path)==entry['sha256']
        p=torch.load(path,map_location='cpu',weights_only=False);assert p['indices']==entry['indices']
        for i,row in zip(p['indices'],p['rows']):
            assert i not in seen and row['index']==i and row['member']==data.members[i];seen.add(i)
            j=int(data.indices[i]);target=row['target'];n=int(data.arrays['n'][j])
            assert target.dtype==torch.float32 and n<=len(target)<=120 and bool(torch.isfinite(target).all())
            arrays['teacher'][j]=0;arrays['teacher'][j,:len(target)]=target.numpy();arrays['usable'][j]=float(row['usable']);arrays['target_done'][j]=True
    assert seen==set(range(expected))

@torch.inference_mode()
def main():
    seed();check_stop();begin=time.perf_counter()
    if not (RUN/'sources.json').exists():snapshot()
    expected=read(RUN/'sources.json');current=sources();assert all(current.get(n)==h for n,h in expected.items())
    model,_=baseline();lp=RUN/'L/checkpoints/best.pt';model.placer.load_state_dict(torch.load(lp,map_location='cpu',weights_only=False)['model']);model=model.cuda().eval()
    teacher=teacher_model();assert read(ORIGINAL/'results.json')['status']=='complete'
    root=RUN/'teacher';root.mkdir(exist_ok=True)
    binding=dict(new_L_sha256=sha(lp),teacher_E85_sha256=SOURCE['checkpoint_sha256'],F_start_sha256=PLAN['source_sha256'],sources=expected,source_config_sha256=sha(ORIGINAL/'config.json'),original_manifests={r:sha(ORIGINAL/r/'manifest.json') for r in ('train','dev_monitor500')},extension='same frozen E85 and FULL on train16000..63999; never C5 teacher',teacher_settings=SOURCE,test_read=False)
    if (root/'config.json').exists():assert read(root/'config.json')==binding
    else:save(root/'config.json',binding)
    for role in ('train','dev_monitor500'):
        check_stop();data=Data(role);folder=root/data.parent;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():assert read(folder/'complete.json')['binding']==binding;continue
        full=len(data.arrays['n']);arrays={}
        for k,shape,dtype in [('base',(full,120,3),np.float32),('teacher',(full,120,3),np.float32),('usable',(full,),np.float32),('target_done',(full,),np.bool_),('base_done',(full,),np.bool_)]:
            p=folder/(k+'.npy');arrays[k]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=dtype,shape=shape)
        batch=Packer(data,supervision=False)
        def flush():
            for a in arrays.values():a.flush()
            save(root/'progress.json',dict(role=role,total=len(data),base_done=int(arrays['base_done'][data.indices].sum()),target_done=int(arrays['target_done'][data.indices].sum()),elapsed_seconds=time.perf_counter()-begin,test_read=False))
        try:
            old_labels(role,data,arrays);flush()
            # The model trained in this run supplies base only; original E85
            # supplies all newly generated supervision targets.
            for start in range(0,len(data),32):
                check_stop();ids=[i for i in range(start,min(start+32,len(data))) if not arrays['base_done'][data.indices[i]]]
                if not ids:continue
                g,count=batch.gpu(ids);base=model.placer(public(g))[:count].cpu().numpy();assert np.isfinite(base).all()
                j=data.indices[ids];arrays['base'][j]=base;arrays['base_done'][j]=True
                if (start+32)%1024==0 or start+32>=len(data):flush();print('New L bases',role,start+len(ids),'/',len(data),flush=True)
            if role=='train':
                rows=PublicRows();pending={};collected=0
                def collect():
                    nonlocal collected
                    ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                    for future in ready:
                        expected_index=pending.pop(future);index,target,usable=future.result();assert index==expected_index
                        arrays['teacher'][index]=0;arrays['teacher'][index,:len(target)]=target;arrays['usable'][index]=usable;arrays['target_done'][index]=True;collected+=1
                    if collected and collected%128==0:flush()
                with ProcessPoolExecutor(max_workers=SOURCE['workers'],mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
                    for start in range(16000,len(data),32):
                        check_stop();ids=[i for i in range(start,min(start+32,len(data))) if not arrays['target_done'][data.indices[i]]]
                        if not ids:continue
                        host,count=batch.gpu(ids);nmax=int(data.arrays['n'][data.indices[ids]].max());g=crop(public(host),nmax)
                        base=teacher.placer(g);source=teacher.refiner(base,g);physical=(decode(source,g)*g['scale'][:,None,None]).cpu().double().numpy();scales=g['scale'].cpu().numpy()
                        for k,index in enumerate(ids):
                            check_stop()
                            while len(pending)>=16:collect();check_stop()
                            j=int(data.indices[index]);inputs=rows.get(j);n=int(inputs[0]);p=physical[k,:n].copy();c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
                            p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
                            pending[pool.submit(solve,(j,inputs,p,float(scales[k])))]=j
                        if start%1024==0:flush();print('Original E85 teacher extension',int(arrays['target_done'][data.indices].sum()),'/64000',flush=True)
                    while pending:check_stop();collect()
            assert bool(arrays['target_done'][data.indices].all()) and bool(arrays['base_done'][data.indices].all())
            flush();save(folder/'complete.json',dict(count=len(data),binding=binding,usable=int(arrays['usable'][data.indices].sum()),test_read=False))
        finally:flush()
    save(root/'results.json',dict(status='complete',train=64000,dev_monitor=500,binding=binding,elapsed_seconds=time.perf_counter()-begin,test_read=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:save(RUN/'teacher/error.json',dict(error=repr(e),at_utc=utc(),no_automatic_retry=True));raise
