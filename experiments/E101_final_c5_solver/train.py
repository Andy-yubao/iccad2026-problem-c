"""Two explicit fixed-Loss stages, exact member order, full checkpoint state."""
import argparse,gc,random,time
import numpy as np
import torch
from .common import HERE,RUN,PLAN,read,save,checkpoint,sha,seed,rng,restore_rng,stopped,sources,utc
from .data import Data,Packer,public,PUBLIC
from .model import DeploymentModel
from .placer.loss import per_case as loss_l
from .placer.step_control import GraphStepControl
from . import loss_f

def baseline():
    p=HERE/'assets/c5_baseline.pt';assert sha(p)==PLAN['source_sha256']
    s=torch.load(p,map_location='cpu',weights_only=False);m=DeploymentModel(**s['model_config']);m.load_state_dict(s['model'],strict=True)
    return m,s['model_config']

def configure(stage,model):
    model.requires_grad_(False)
    if stage=='L':
        model.placer.requires_grad_(True)
        for layer in model.placer.layers:layer.local.requires_grad_(False)
        return model.placer
    model.refiner.requires_grad_(True);return model.refiner

def terms(stage,model,g):
    if stage=='L':return loss_l(model(public(g)),g,'topology')
    return loss_f.per_case(model(g['base'],public(g)),g,g['teacher'],g['usable'])

class Engine:
    def __init__(self,stage,model,opt,sample,fast_update=False):
        self.stage,self.model,self.opt=stage,model,opt;self.g={k:v.clone() for k,v in sample.items()};self.parameters=[p for p in model.parameters() if p.requires_grad]
        self.projection=GraphStepControl(model,.005) if stage=='L' else None
        random_state=torch.cuda.get_rng_state_all();s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream());model.train()
        def forward():
            opt.zero_grad(set_to_none=True);v=terms(stage,model,self.g);total=v['total'].mean();total.backward();norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,foreach=True)
            finite=torch.isfinite(torch.stack([x.mean() for x in v.values()])).all()&torch.isfinite(norm)
            return v,norm,finite
        with torch.cuda.stream(s):
            for _ in range(2):v,norm,finite=forward();del v,norm,finite
        torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=s):self.values,self.norm,self.finite=forward()
        torch.cuda.synchronize();torch.cuda.set_rng_state_all(random_state)
        self.update=None
        if fast_update:
            from .update_engine import CapturedUpdate
            self.update=CapturedUpdate(opt,self.parameters,self.projection)
    def step(self,batch):
        for k,v in batch.items():self.g[k].copy_(v,non_blocking=True)
        self.graph.replay();gate=torch.stack((self.finite.float(),self.values['total'].mean(),self.norm,self.values['total'].amax())).detach().cpu().tolist()
        if gate[0]!=1 or (gate[1]>=10 if self.stage=='L' else gate[3]>=1000) or gate[2]>=10000:raise RuntimeError('Finite/scale gate rejected before AdamW')
        if self.update:self.update.replay()
        else:
            if self.projection:self.projection.before()
            self.opt.step()
            if self.projection:self.projection.after()
            if not bool(torch.isfinite(torch.stack(torch._foreach_norm(self.parameters))).all()):raise RuntimeError('Nonfinite parameters after AdamW')
        return gate[1]

class Batch:
    def __init__(self,data,stage):
        self.data,self.stage=data,stage;self.public=Packer(data,supervision=stage=='L');self.labels=None
        if stage=='F':
            folder=RUN/'teacher'/data.parent;self.labels={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ('base','teacher','usable')}
            self.extra={k:torch.empty((32,*v.shape[1:]),dtype=torch.float32,pin_memory=True) for k,v in self.labels.items()}
    def gpu(self,ids):
        g,count=self.public.gpu(ids)
        if self.labels is not None:
            pad=list(ids)+[ids[-1]]*(32-count);actual=self.data.indices[pad]
            for k,v in self.extra.items():
                np.take(self.labels[k],actual,axis=0,out=v.numpy());g[k]=v.cuda(non_blocking=True)
        return g,count

@torch.no_grad()
def monitor(stage,model,batch):
    model.eval();vectors=[]
    for start in range(0,len(batch.data),32):
        if stopped():return None
        g,count=batch.gpu(list(range(start,min(start+32,len(batch.data)))));v=terms(stage,model,g)
        values=v['total'][:count].detach().cpu();assert bool(torch.isfinite(values).all());vectors.extend(values.tolist())
    return float(np.mean(vectors))

def run(stage,resume=False):
    begin=time.perf_counter();seed();folder=RUN/stage;cfg=next(x for x in PLAN['stages'] if x['name']==stage);folder.mkdir(parents=True,exist_ok=True)
    train=Data('train');dev=Data('dev_monitor500');mon=Data('train_monitor500')
    loss_f.MAX_GROUP_SIZE=max(train.meta['max_public_group'],dev.meta['max_public_group'])
    model,model_config=baseline()
    if stage=='F':
        l=torch.load(RUN/'L/checkpoints/best.pt',map_location='cpu',weights_only=False);model.placer.load_state_dict(l['model'],strict=True);del l
    model=model.cuda();part=configure(stage,model);levels=cfg['lr_levels'];opt=torch.optim.AdamW([p for p in part.parameters() if p.requires_grad],lr=levels[0],weight_decay=.0003)
    batch=Batch(train,stage);devbatch=Batch(dev,stage);monbatch=Batch(mon,stage)
    identity=dict(revision=PLAN['revision'],stage=stage,plan_sha256=sha(HERE/'plan.json'),source_sha256=read(RUN/'sources.json'),registry_sha256=sha(HERE.parents[1]/'protocol/reference_workset_v6.json'),max_public_group=loss_f.MAX_GROUP_SIZE,train_count=len(train),selection='same fixed totalLoss dev500 mean',engine='v2.guarded_backward_then_captured_AdamW_projection' if stage=='L' else 'v1.guarded_graph_backward_eager_AdamW')
    prior=0.;state=dict(epoch=0,cursor=0,step=0,seen=np.zeros(len(train),dtype=bool),order=None,best=None,best_epoch=None,history=[],tier=0,bad=0,plateau_best=None,last_drop=-1,pending_monitor=True)
    projection=None
    if resume:
        saved=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:
            extension_path=RUN/'budget_extension.json'
            if extension_path.exists():
                extension=read(extension_path)
                assert stage=='L' and sha(folder/'checkpoints/latest.pt')==extension['from_checkpoint_sha256']
                assert saved['identity']==extension['from_identity']
                assert identity['source_sha256']==extension['new_sources'] and identity['plan_sha256']==extension['new_plan_sha256']
                assert {k:v for k,v in saved['identity'].items() if k not in ('source_sha256','plan_sha256')}=={k:v for k,v in identity.items() if k not in ('source_sha256','plan_sha256')}
            else:
                migration=read(RUN/'engineering/migration_v2.json')
                assert sha(folder/'checkpoints/latest.pt')==migration['from_checkpoint_sha256']
                assert saved['identity']['source_sha256']==migration['old_sources'] and identity['source_sha256']==migration['new_sources']
                assert {k:v for k,v in saved['identity'].items() if k not in ('source_sha256','engine')}=={k:v for k,v in identity.items() if k not in ('source_sha256','engine')}
        if saved['finished']:return saved['status']
        part.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['progress'];prior=saved['elapsed_seconds'];restore_rng(saved['rng']);projection=saved.get('projection')
        if (folder/'heartbeat.json').exists():prior=max(prior,read(folder/'heartbeat.json')['elapsed_seconds'])
        migration_path=RUN/'engineering/migration_v2.json'
        if stage=='L' and migration_path.exists():prior=max(prior,read(migration_path)['preserved_elapsed_seconds'])
        extension_path=RUN/'budget_extension.json'
        if stage=='L' and extension_path.exists():prior=max(prior,read(extension_path)['preserved_elapsed_seconds'])
    else:save(folder/'config.json',dict(identity=identity,initialization='weights-only C5 migration; fresh optimizer/schedule',configuration=cfg,started_utc=utc()))
    elapsed=lambda:prior+time.perf_counter()-begin
    sample,_=batch.gpu([0]*32);engine=Engine(stage,part,opt,sample,fast_update=stage=='L')
    if projection and engine.projection:
        engine.projection.steps=projection['steps'];engine.projection.clipped.copy_(torch.tensor(projection['clipped_steps_per_group'],device='cuda'))
    print(stage,'FP32 graph captured; no warmup optimizer updates',flush=True)
    lastsave=0.;lastbeat=0.
    def store(status,finished=False,best=False):
        nonlocal lastsave
        value=dict(model=part.state_dict(),model_config=model_config,optimizer=opt.state_dict(),rng=rng(),progress=state,identity=identity,elapsed_seconds=elapsed(),status=status,finished=finished,projection=engine.projection.summary() if engine.projection else None)
        checkpoint(folder/'checkpoints/latest.pt',value)
        if best:checkpoint(folder/'checkpoints/best.pt',value)
        lastsave=elapsed();save(folder/'progress.json',dict(status=status,finished=finished,epoch=state['epoch'],cursor=state['cursor'],steps=state['step'],seen_count=int(state['seen'].sum()),best_epoch=state['best_epoch'],best_loss=state['best'],lr=levels[state['tier']],history=state['history'],elapsed_seconds=elapsed(),test_read=False))
    while True:
        if stopped():store('user_stop');return 'user_stop'
        if state['pending_monitor']:
            tr=monitor(stage,part,monbatch);dv=monitor(stage,part,devbatch)
            if tr is None or dv is None:store('user_stop');return 'user_stop'
            best=state['best'] is None or dv<state['best']
            if best:state['best'],state['best_epoch']=dv,state['epoch']
            delta=max(1e-5,abs(state['plateau_best'])*.001) if state['plateau_best'] is not None else 0
            if state['plateau_best'] is None or dv<state['plateau_best']-delta:state['plateau_best']=dv;state['bad']=0
            else:state['bad']+=1
            if state['bad']>=3 and state['tier']<3:
                state['tier']+=1;state['bad']=0;state['last_drop']=state['epoch']
                for group in opt.param_groups:group['lr']=levels[state['tier']]
            state['history'].append(dict(epoch=state['epoch'],train_loss=tr,dev_loss=dv,elapsed_seconds=elapsed(),tier=state['tier'],steps=state['step']));state['pending_monitor']=False;store('training',best=best)
            print(stage,'epoch',state['epoch'],'train',tr,'dev',dv,'best',state['best_epoch'],'seconds',round(elapsed(),1),flush=True)
            if state['tier']==3 and state['bad']>=3 and state['epoch']-state['last_drop']>=2:
                store('early_stop',True);save(folder/'results.json',dict(status='early_stop',best_epoch=state['best_epoch'],elapsed_seconds=elapsed(),steps=state['step'],test_read=False));return 'early_stop'
        if elapsed()>=cfg['max_seconds'] or state['epoch']>=cfg['max_epochs']:
            store('budget',True);save(folder/'results.json',dict(status='budget',best_epoch=state['best_epoch'],elapsed_seconds=elapsed(),steps=state['step'],test_read=False));return 'budget'
        part.train()
        if state['order'] is None:state['order']=np.random.permutation(len(train));state['cursor']=0
        ids=state['order'][state['cursor']:state['cursor']+32].tolist();assert len(ids)==32
        start=time.perf_counter();g,_=batch.gpu(ids);packed=time.perf_counter();loss=engine.step(g);wall=time.perf_counter()-start
        state['cursor']+=32;state['step']+=1;state['seen'][ids]=True
        if state['cursor']==len(train):state['epoch']+=1;state['order']=None;state['cursor']=0;state['pending_monitor']=True
        if state['step']==1 or elapsed()-lastbeat>=5:
            save(folder/'heartbeat.json',dict(pid=__import__('os').getpid(),stage=stage,epoch=state['epoch'],steps=state['step'],cursor=state['cursor'],seen_count=int(state['seen'].sum()),last_loss=loss,step_seconds=wall,batch_prepare_seconds=packed-start,guarded_GPU_update_seconds=wall-(packed-start),samples_per_second=32/wall,elapsed_seconds=elapsed(),updated_utc=utc(),actual_optimizer_updates=True,test_read=False));lastbeat=elapsed()
        if elapsed()-lastsave>=60 or state['step']==1:store('training')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['L','F']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    from .common import exclusive
    try:
        with exclusive(RUN/'training.lock'):
            result=run(a.stage,a.resume)
            if result=='user_stop':raise SystemExit(2)
    except Exception as e:save(RUN/a.stage/'error.json',dict(error=repr(e),at_utc=utc(),no_automatic_retry=True));raise
