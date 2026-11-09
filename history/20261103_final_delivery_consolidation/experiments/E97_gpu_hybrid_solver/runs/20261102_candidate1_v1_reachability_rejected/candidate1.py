"""Fixed public-solver imitation objective and fully resumable small-F training."""
import argparse,copy,datetime,gc,json,random,signal,subprocess,time,zipfile
from pathlib import Path
import numpy as np
from .latency import torch,ROOT,HERE,IDS,DeploymentModel,decode85,sha
from .teacher import checkpoint
from .train_io.io import save,exclusive
from .train_io.data import load_rows,LEGACY
from .train_io.dataset_view import registry,REGISTRY
from .candidate_packer import PublicPacker

TEACHER=HERE/'runs/20261101_public_teacher'
OUTPUT=HERE/'runs/20261102_candidate1_distill_seed0'
PREFLIGHT=HERE/'runs/20261102_candidate1_preflight'
LEVELS=(.001,.00025,.0001,.00005)

def configuration():
    return dict(revision='E97.candidate1.public_solver_imitation.v1',center=1.,shape=.25,relative=.25,beta=.1,near=.25,
        weighting='teacher usable; all frozen members retained in mean',reference=False,selection='same dev500 arithmetic mean for best/LR/early-stop')

def per_case(state,g,target,usable):
    mask=g['mask']*g['valid'][...,None];local=g['area'].clamp_min(1e-12).sqrt()
    center=torch.nn.functional.smooth_l1_loss(state[...,:2]/local[...,None],target[...,:2]/local[...,None],reduction='none',beta=.1)
    center=(center*mask[...,:2]).sum((1,2))/mask[...,:2].sum((1,2)).clamp_min(1)
    shape=((state[...,2]-target[...,2]).square()*mask[...,2]).sum(1)/mask[...,2].sum(1).clamp_min(1)
    delta=state[:,:,:2][:,:,None]-state[:,:,:2][:,None,:];wanted=target[:,:,:2][:,:,None]-target[:,:,:2][:,None,:]
    scale=(local[:,:,None]+local[:,None,:])/2;n=state.shape[1];ix=torch.arange(n,device=state.device)
    pairs=g['valid'][:,:,None]&g['valid'][:,None,:]&(ix[:,None]!=ix[None,:])[None]
    pairs=pairs&(mask[...,0].bool()[:,:,None]|mask[...,0].bool()[:,None,:])
    # Labels contain center/log-aspect; near geometry uses public nominal area
    # and locked WH, rather than an unrecorded teacher area-tolerance variable.
    wh=decode85(target,g)[...,2:]
    gap=wanted.abs()-(wh[:,:,None]+wh[:,None,:])/2
    near=(gap.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)&pairs
    error=torch.nn.functional.smooth_l1_loss(delta/scale.clamp_min(1e-8)[...,None],wanted/scale.clamp_min(1e-8)[...,None],reduction='none',beta=.1).mean(-1)
    relative=(error*near).sum((1,2))/near.sum((1,2)).clamp_min(1)
    return dict(teacher_center=center,teacher_shape=shape,teacher_relative=relative,total=usable*(center+.25*shape+.25*relative))

def seed():
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(v):
    random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch']);torch.cuda.set_rng_state_all(v['cuda'])
def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return copy.deepcopy(v)
def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def source_identity():
    names=[Path(__file__),HERE/'candidate_packer.py',HERE/'candidate1_plan.json',ROOT/'docs/protocol.md']
    names+=list((HERE/'nn85').glob('*.py'))+list((HERE/'train_io').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(names)}

def teacher_rows(role):
    if role not in ('train','dev_monitor500'):raise PermissionError('Only registered teacher panels')
    result=json.loads((TEACHER/'results.json').read_text(encoding='utf-8'));assert result['finished'] and result['status']=='complete'
    config=json.loads((TEACHER/'config.json').read_text(encoding='utf-8'));manifest=json.loads((TEACHER/role/'manifest.json').read_text(encoding='utf-8'))
    found={}
    for part in manifest['parts']:
        path=TEACHER/part['path'];assert sha(path)==part['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['indices']==part['indices']
        for i,row in zip(saved['indices'],saved['rows']):
            assert i not in found and row['member']==config['members'][role][i]
            assert all(bool(torch.isfinite(row[k]).all()) for k in ('base','source','target'))
            found[i]=row
    assert sorted(found)==list(range(manifest['count']))
    return found

def attach(rows,labels,indices=None):
    selected=list(range(len(rows))) if indices is None else indices
    for row,index in zip(rows,selected):
        label=labels[index];assert list(row['member'])==label['member']
        row.update(draft=label['base'],teacher=label['target'],teacher_usable=float(label['usable']))
    return rows

class Packer:
    def __init__(self,row):
        self.public=PublicPacker(row,p=480);self.target=torch.zeros(32,120,3,pin_memory=True);self.usable=torch.zeros(32,pin_memory=True)
    def pack(self,rows):
        batch,draft,count=self.public.pack(rows);self.target.zero_();self.usable.zero_()
        selected=list(rows)+[rows[-1]]*(32-count)
        for i,row in enumerate(selected):self.target[i,:row['graph']['n']].copy_(row['teacher']);self.usable[i]=row['teacher_usable']
        return {k:batch[k] for k in LEGACY},draft,self.target,self.usable,count
    def gpu(self,rows):
        g,d,t,u,count=self.pack(rows)
        return {k:v.cuda(non_blocking=True) for k,v in g.items()},d.cuda(non_blocking=True),t.cuda(non_blocking=True),u.cuda(non_blocking=True),count

def model_and_optimizer():
    plan=json.loads((HERE/'candidate1_plan.json').read_text(encoding='utf-8'));path=ROOT/plan['source_checkpoint'];assert sha(path)==plan['source_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda();model.load_state_dict(saved['model'],strict=True)
    model.refiner.requires_grad_(True);model.placer.requires_grad_(False);model.placer.eval()
    assert sum(p.numel() for p in model.refiner.parameters())==241547
    return model,torch.optim.AdamW(model.refiner.parameters(),lr=.001,weight_decay=.0003),saved['model_config']

def guard(model,opt,values,prediction,norm):
    finite=torch.isfinite(torch.stack(list(values.values()))).all()&torch.isfinite(norm)&torch.isfinite(prediction).all()
    gate=torch.stack((finite.to(torch.float32),values['total'].amax(),norm)).detach().cpu().tolist()
    if gate[0]!=1 or gate[1]>=1000 or gate[2]>=10000:raise RuntimeError('Nonfinite/scale gate before AdamW')
    opt.step()
    if not bool(torch.isfinite(torch.stack(torch._foreach_norm(list(model.refiner.parameters())))).all()):raise RuntimeError('Nonfinite parameters after AdamW')

def eager(model,opt,g,d,t,u):
    model.train();opt.zero_grad(set_to_none=True);prediction=model.refiner(d,g);values=per_case(prediction,g,t,u)
    values['total'].mean().backward();norm=torch.nn.utils.clip_grad_norm_(model.refiner.parameters(),1.,foreach=True)
    guard(model,opt,values,prediction,norm);return values

class Engine:
    def __init__(self,model,opt,g,d,t,u):
        self.model,self.opt=model,opt;self.g={k:v.clone() for k,v in g.items()};self.d=d.clone();self.t=t.clone();self.u=u.clone()
        opt.zero_grad(set_to_none=True);gc.collect();random_state=torch.cuda.get_rng_state_all();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());model.train()
        with torch.cuda.stream(stream):
            for _ in range(2):
                opt.zero_grad(set_to_none=True);prediction=model.refiner(self.d,self.g);v=per_case(prediction,self.g,self.t,self.u);v['total'].mean().backward();torch.nn.utils.clip_grad_norm_(model.refiner.parameters(),1.,foreach=True)
                del prediction,v
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            opt.zero_grad(set_to_none=True);self.prediction=model.refiner(self.d,self.g);self.values=per_case(self.prediction,self.g,self.t,self.u);self.values['total'].mean().backward()
            self.norm=torch.nn.utils.clip_grad_norm_(model.refiner.parameters(),1.,error_if_nonfinite=False,foreach=True)
        torch.cuda.synchronize();torch.cuda.set_rng_state_all(random_state)
    def replay(self,g=None,d=None,t=None,u=None):
        if g is not None:
            for k,v in g.items():self.g[k].copy_(v,non_blocking=True)
        for target,value in ((self.d,d),(self.t,t),(self.u,u)):
            if value is not None:target.copy_(value,non_blocking=True)
        self.graph.replay();guard(self.model,self.opt,self.values,self.prediction,self.norm);return self.values

def preflight():
    begin=time.perf_counter();PREFLIGHT.mkdir(exist_ok=False);seed();labels=teacher_rows('train');rows=attach(load_rows('train',IDS),labels,IDS)
    packer=Packer(rows[0]);g,d,t,u,_=packer.gpu([rows[i%5] for i in range(32)])
    model,opt,cfg=model_and_optimizer();model.eval();initial=cpu(model.state_dict());initial_rng=rng()
    prediction=model.refiner(d,g);values=per_case(prediction,g,t,u)
    torch.testing.assert_close(values['total'],u*(values['teacher_center']+.25*values['teacher_shape']+.25*values['teacher_relative']),rtol=0,atol=0)
    initial_loss=float(values['total'].mean().detach().cpu());del prediction,values
    eager(model,opt,g,d,t,u);eager_model=cpu(model.state_dict());eager_opt=cpu(opt.state_dict())
    del model,opt;gc.collect();torch.cuda.empty_cache()
    model,opt,_=model_and_optimizer();model.load_state_dict(initial);restore(initial_rng);engine=Engine(model,opt,g,d,t,u)
    exact(initial,model.state_dict());assert not opt.state;exact(initial_rng,rng());engine.replay()
    error=max(float((v.detach().cpu()-eager_model[k]).abs().max()) for k,v in model.state_dict().items())
    assert error<3e-5,error
    moment_error=0.
    for k,v in cpu(opt.state_dict())['state'].items():
        for name,value in v.items():
            torch.testing.assert_close(value,eager_opt['state'][k][name],rtol=1e-4,atol=1e-7)
            moment_error=max(moment_error,float((value-eager_opt['state'][k][name]).abs().max()))
    first=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),rng=rng(),cursor=32,order=np.random.default_rng(0).permutation(16000))
    checkpoint(PREFLIGHT/'step1.pt',first);engine.replay();whole=cpu(model.state_dict());moments=cpu(opt.state_dict());random_state=rng()
    assert all(p.grad is None for p in model.placer.parameters())
    for k,v in initial.items():
        if k.startswith('placer.'):assert torch.equal(v,whole[k])
    del model,opt,engine;gc.collect();torch.cuda.empty_cache()
    first=torch.load(PREFLIGHT/'step1.pt',map_location='cpu',weights_only=False);model,opt,_=model_and_optimizer();model.load_state_dict(first['model']);opt.load_state_dict(first['optimizer'])
    engine=Engine(model,opt,g,d,t,u);restore(first['rng']);engine.replay();exact(whole,model.state_dict());exact(moments,opt.state_dict());exact(random_state,rng())
    before=cpu(model.state_dict());before_opt=cpu(opt.state_dict());bad=t.clone();bad[0,0,0]=float('nan')
    try:engine.replay(t=bad);raise AssertionError('Nonfinite teacher accepted')
    except RuntimeError:pass
    exact(before,model.state_dict());exact(before_opt,opt.state_dict())
    value=dict(status='passed',ids=IDS,source_sha256=source_identity(),model_config=cfg,loss=configuration(),initial_loss=initial_loss,eager_graph_update_max_abs=error,
        eager_graph_Adam_moment_max_abs=moment_error,capture_no_Adam_or_parameter_update=True,restored_next_model_Adam_RNG_bit_exact=True,nonfinite_rejected_before_Adam=True,frozen_L_unchanged=True,formal_training_started=False,seconds=time.perf_counter()-begin,test_read=False)
    save(PREFLIGHT/'results.json',value);print(json.dumps({k:v for k,v in value.items() if k!='source_sha256'},indent=2),flush=True)

class Scheduler:
    def __init__(self):self.best=None;self.bad=0;self.early_bad=0;self.drops=0;self.last_drop=None
    def step(self,value,epoch,opt):
        delta=0. if self.best is None else max(1e-5,abs(self.best)*.001)
        if self.best is None or value<self.best-delta:self.best=value;self.bad=0;self.early_bad=0
        else:self.bad+=1;self.early_bad+=1
        if self.bad>=3 and self.drops<3:
            self.drops+=1;self.bad=0;self.last_drop=epoch
            for group in opt.param_groups:group['lr']=LEVELS[self.drops]
    def stop(self,epoch):return self.drops==3 and self.early_bad>=3 and epoch-self.last_drop>=2

@torch.no_grad()
def monitor(model,rows,packer,halt):
    model.eval();sums={}
    for start in range(0,len(rows),32):
        if halt():return None
        g,d,t,u,count=packer.gpu(rows[start:start+32]);values=per_case(model.refiner(d,g),g,t,u)
        keys=list(values);flat=torch.stack([values[k][:count] for k in keys]).cpu()
        if not bool(torch.isfinite(flat).all()):raise RuntimeError('Nonfinite monitor')
        for key,x in zip(keys,flat):sums[key]=sums.get(key,0.)+float(x.sum())
    return {k:v/len(rows) for k,v in sums.items()}|dict(count=len(rows))

def train(resume=False):
    check=json.loads((PREFLIGHT/'results.json').read_text(encoding='utf-8'));assert check['status']=='passed' and check['source_sha256']==source_identity()
    with exclusive(HERE/'runs/candidate1_training.lock'):
        begin=time.perf_counter();seed();study=json.loads((HERE/'study.json').read_text(encoding='utf-8'));deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
        plan=json.loads((HERE/'candidate1_plan.json').read_text(encoding='utf-8'))
        if (HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise RuntimeError('STOP/deadline before start')
        model,opt,cfg=model_and_optimizer();sched=Scheduler();prior=0.;saved=None
        identity=dict(revision='E97.candidate1.teacher_imitation.v1',source_sha256=source_identity(),teacher_config_sha256=sha(TEACHER/'config.json'),teacher_manifests={role:sha(TEACHER/role/'manifest.json') for role in ('train','dev_monitor500')},loss=configuration(),model_config=cfg,seed=0,batch_size=32,precision='FP32 no AMP/TF32',registry_sha256=sha(REGISTRY),max_seconds=3600,max_epochs=100,global_deadline=study['deadline_utc'],test_read=False)
        progress=dict(epoch=0,step=0,cursor=0,order=None,pending_monitor=True,best_loss=None,best_epoch=None,seen=np.zeros(16000,dtype=bool),history=[])
        if resume:
            saved=torch.load(OUTPUT/'checkpoints/latest.pt',map_location='cpu',weights_only=False);assert saved['identity']==identity
            if saved['finished']:print('Already finished');return
            if saved['status']=='error':raise RuntimeError('Error requires diagnosis, no automatic retry')
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.__dict__.update(saved['scheduler']);progress=saved['progress'];prior=saved['elapsed_seconds'];restore(saved['rng'])
        else:
            OUTPUT.mkdir(exist_ok=False)
            if study['formal_candidates_started']>=study['max_formal_training_candidates']:raise RuntimeError('Candidate cap reached')
            study['formal_candidates_started']+=1;study.update(status='candidate1_training',current_stage='candidate1 fixed imitation training')
            for item in study['candidate_ledger']:
                if item['candidate']==1:item.update(status='training',formal_training_started=True,output=OUTPUT.relative_to(HERE).as_posix())
            save(HERE/'study.json',study);save(OUTPUT/'config.json',dict(identity=identity,plan=plan,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),uncommitted_source_snapshot=True))
            with zipfile.ZipFile(OUTPUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
                for name in identity['source_sha256']:z.write(ROOT/name,name)
        requested=[False];previous=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        elapsed=lambda:prior+time.perf_counter()-begin
        halt=lambda:requested[0] or (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() or elapsed()>=3600 or datetime.datetime.now(datetime.timezone.utc)>=deadline
        def stop_reason():return 'user_stop' if requested[0] or (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() else 'budget'
        last=[elapsed()]
        def store(status,finished=False,best=False):
            value=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=dict(sched.__dict__),rng=rng(),progress=progress,identity=identity,model_config=cfg,elapsed_seconds=elapsed(),status=status,finished=finished)
            checkpoint(OUTPUT/'checkpoints/latest.pt',value)
            if best:checkpoint(OUTPUT/'checkpoints/best.pt',value)
            view={k:v for k,v in progress.items() if k not in ('order','seen')};view.update(status=status,finished=finished,seen_count=int(progress['seen'].sum()),scheduler=dict(sched.__dict__),elapsed_seconds=elapsed(),test_read=False)
            save(OUTPUT/'progress.json',view);save(HERE/'state.json',dict(status='candidate1_'+status,active_run=OUTPUT.relative_to(HERE).as_posix(),step=progress['step'],epoch=progress['epoch'],formal_training_started=True,formal_candidates_started=1,max_formal_training_candidates=8,test_read=False));last[0]=elapsed()
        try:
            labels=teacher_rows('train');devlabels=teacher_rows('dev_monitor500');rows=attach(load_rows('train'),labels);dev=attach(load_rows('dev_monitor500'),devlabels)
            lookup={tuple(r['member']):i for i,r in enumerate(rows)};monitor_ids=[lookup[tuple(r['member'])] for r in registry()['sets']['train_monitor500']['records']]
            packer=Packer(rows[0]);g,d,t,u,_=packer.gpu(rows[:32]);engine=Engine(model,opt,g,d,t,u)
            if saved:restore(saved['rng'])
            print('Candidate1 engine ready; trainable241547; teacher train/dev',len(rows),len(dev),flush=True)
            status='running'
            while status=='running':
                if halt():status=stop_reason();break
                if progress['pending_monitor']:
                    val=monitor(model,dev,packer,halt);tr=monitor(model,[rows[i] for i in monitor_ids],packer,halt)
                    if val is None or tr is None:status=stop_reason();break
                    improved=progress['best_loss'] is None or val['total']<progress['best_loss']
                    if improved:progress['best_loss']=val['total'];progress['best_epoch']=progress['epoch']
                    lr=opt.param_groups[0]['lr'];sched.step(val['total'],progress['epoch'],opt)
                    progress['history'].append(dict(epoch=progress['epoch'],step=progress['step'],train_monitor=tr,dev_monitor=val,lr=lr,lr_next=opt.param_groups[0]['lr'],train_monitor_seen_count=int(progress['seen'][monitor_ids].sum())))
                    progress['pending_monitor']=False;store('running',best=improved);print('epoch',progress['epoch'],'trainLoss',tr['total'],'devLoss',val['total'],'lr',opt.param_groups[0]['lr'],flush=True)
                    if sched.stop(progress['epoch']):status='early_stop';break
                    if progress['epoch']>=100:status='max_epochs';break
                if progress['order'] is None:progress['order']=np.random.default_rng(progress['epoch']).permutation(16000);progress['cursor']=0
                while progress['cursor']<16000:
                    if halt():break
                    indices=progress['order'][progress['cursor']:progress['cursor']+32];g,d,t,u,_=packer.pack([rows[int(i)] for i in indices]);model.train();values=engine.replay(g,d,t,u)
                    progress['seen'][indices]=True;progress['cursor']+=len(indices);progress['step']+=1
                    if elapsed()-last[0]>=30:store('running')
                    if progress['step']%250==0:print('step',progress['step'],'Loss',float(values['total'].mean().detach().cpu()),flush=True)
                if progress['cursor']<16000:status=stop_reason();break
                progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True
            finished=status in ('early_stop','max_epochs','budget');store(status,finished)
            save(OUTPUT/'results.json',dict(status=status,finished=finished,steps=progress['step'],epochs=progress['epoch'],best_epoch=progress['best_epoch'],elapsed_seconds=elapsed(),test_read=False))
        except BaseException:store('error');raise
        finally:signal.signal(signal.SIGINT,previous)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.preflight:preflight()
    elif a.execute:train(a.resume)
    else:print('Explicit --preflight or --execute required; no work started')
