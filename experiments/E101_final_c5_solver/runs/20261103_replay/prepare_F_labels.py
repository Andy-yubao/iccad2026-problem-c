"""Keep historical E85 targets immutable; regenerate only the new L bases."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from experiments.E101_final_c5_solver.common import HERE,RUN,PLAN,read,save,sha,seed,check_stop,snapshot,sources,utc
from experiments.E101_final_c5_solver.data import Data,Packer,public
from experiments.E101_final_c5_solver.train import baseline,configure,Engine
from experiments.E101_final_c5_solver import loss_f

BUNDLE=Path(__file__).resolve().parent
WORKSPACE=BUNDLE.parents[3]
TEACHER=WORKSPACE/'history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/20261101_public_teacher'

def rows(role,data,limit=None):
    manifest=read(TEACHER/role/'manifest.json')
    assert manifest['count']==manifest['completed']==len(data)
    seen=set()
    for entry in manifest['parts']:
        path=TEACHER/entry['path'];assert path.resolve().is_relative_to(TEACHER.resolve())
        assert sha(path)==entry['sha256'],'Historical teacher part changed'
        part=torch.load(path,map_location='cpu',weights_only=False)
        assert part['indices']==entry['indices'] and len(part['rows'])==len(entry['indices'])
        for index,row in zip(part['indices'],part['rows']):
            assert index not in seen and row['index']==index and row['member']==data.members[index]
            seen.add(index)
            if limit is None or index<limit:
                j=int(data.indices[index]);n=int(data.arrays['n'][j]);target=row['target']
                assert target.dtype==torch.float32 and target.ndim==2 and target.shape[1]==3 and n<=len(target)<=120
                assert bool(torch.isfinite(target).all())
                yield index,j,row
        if limit is not None and len(seen)>=limit:return
    assert seen==set(range(len(data)))

def load(lpath):
    model,cfg=baseline();state=torch.load(lpath,map_location='cpu',weights_only=False)
    model.placer.load_state_dict(state['model'],strict=True)
    return model.cuda().eval(),cfg

def short_check():
    """One discarded real F update, using only five train members; no resume test."""
    seed();data=Data('train');dev=Data('dev_monitor500')
    loss_f.MAX_GROUP_SIZE=max(data.meta['max_public_group'],dev.meta['max_public_group'])
    lpath=BUNDLE/'historical/E74/best.pt';model,_=load(lpath);part=configure('F',model)
    ids=list(range(5));pad=ids+[ids[-1]]*27;g,_=Packer(data,supervision=False).gpu(ids)
    with torch.no_grad():g['base']=model.placer(public(g)).detach()
    targets=torch.zeros((5,120,3));usable=torch.zeros(5)
    for i,_,row in rows('train',data,limit=5):targets[i,:len(row['target'])]=row['target'];usable[i]=float(row['usable'])
    g['teacher']=targets[pad].cuda();g['usable']=usable[pad].cuda()
    opt=torch.optim.AdamW(part.parameters(),lr=PLAN['stages'][0]['lr_levels'][0],weight_decay=.0003)
    before=[p.detach().clone() for p in part.parameters()]
    engine=Engine('F',part,opt,g);value=engine.step(g)
    delta=max(float((p.detach()-b).abs().max()) for p,b in zip(part.parameters(),before));assert np.isfinite(value) and delta>0
    save(BUNDLE/'F_bridge_short_check.json',dict(status='passed',members=data.members[:5],unique_members=5,actual_optimizer_updates=1,discarded_update=True,loss=value,max_parameter_change=delta,target_source='original E85 tensors, SHA and member verified',L_for_check_sha256=sha(lpath),test_read=False,checked_utc=utc()))
    print('F bridge: five train members, one actual discarded update passed',flush=True)

@torch.inference_mode()
def main():
    seed();check_stop();begin=time.perf_counter()
    if not (RUN/'sources.json').exists():snapshot()
    else:
        expected=read(RUN/'sources.json');current=sources()
        assert all(current.get(n)==h for n,h in expected.items()),'Registered F runtime changed'
    assert read(TEACHER/'results.json')['status']=='complete'
    lpath=RUN/'L/checkpoints/best.pt';model,_=load(lpath)
    binding=dict(new_L_sha256=sha(lpath),F_start_sha256=PLAN['source_sha256'],plan_sha256=sha(HERE/'plan.json'),historical_teacher_config_sha256=sha(TEACHER/'config.json'),historical_teacher_manifests={r:sha(TEACHER/r/'manifest.json') for r in ('train','dev_monitor500')},sources=read(RUN/'sources.json'),targets='Original E85 target and usable copied exactly, no solver regeneration',base='New unique E74 best; eval/no_grad, public inputs only',test_read=False)
    root=RUN/'teacher';root.mkdir(exist_ok=True)
    if (root/'config.json').exists():assert read(root/'config.json')==binding
    else:save(root/'config.json',binding)
    for role in ('train','dev_monitor500'):
        check_stop();data=Data(role);folder=root/data.parent;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            assert read(folder/'complete.json')['binding']==binding
            continue
        full=16000 if data.parent=='train' else len(data.arrays['n'])
        arrays={}
        for k,shape,dtype in [('base',(full,120,3),np.float32),('teacher',(full,120,3),np.float32),('usable',(full,),np.float32),('done',(full,),np.bool_)]:
            path=folder/(k+'.npy');arrays[k]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
        try:
            # Recheck and retain original label payloads, independent of base progress.
            for i,j,row in rows(role,data):
                check_stop();arrays['teacher'][j]=0;arrays['teacher'][j,:len(row['target'])]=row['target'].numpy();arrays['usable'][j]=float(row['usable'])
            batch=Packer(data,supervision=False)
            for start in range(0,len(data),32):
                check_stop();ids=[i for i in range(start,min(start+32,len(data))) if not arrays['done'][data.indices[i]]]
                if not ids:continue
                g,count=batch.gpu(ids);base=model.placer(public(g))[:count].cpu().numpy()
                assert np.isfinite(base).all()
                actual=data.indices[ids];arrays['base'][actual]=base;arrays['done'][actual]=True
                if (start+32)%512==0 or start+32>=len(data):
                    for a in arrays.values():a.flush()
                    completed=int(arrays['done'][data.indices].sum())
                    save(root/'progress.json',dict(role=role,completed=completed,total=len(data),elapsed_seconds=time.perf_counter()-begin,test_read=False));print('New L bases / original E85 targets',role,completed,'/',len(data),flush=True)
            assert bool(arrays['done'][data.indices].all())
            save(folder/'complete.json',dict(count=len(data),binding=binding,usable=int(arrays['usable'][data.indices].sum()),test_read=False))
        finally:
            for a in arrays.values():a.flush()
    save(root/'results.json',dict(status='complete',train=16000,dev_monitor=500,elapsed_seconds=time.perf_counter()-begin,binding=binding,test_read=False))

if __name__=='__main__':
    try:
        if '--short-check' in sys.argv:short_check()
        else:main()
    except BaseException as e:save(RUN/'teacher/error.json',dict(error=repr(e),at_utc=utc(),no_automatic_retry=True));raise
