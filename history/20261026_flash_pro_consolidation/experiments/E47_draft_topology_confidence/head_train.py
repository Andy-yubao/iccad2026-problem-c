"""Bounded frozen-encoder learnability probe; never invokes an optimizer solver."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='4'
os.environ['OMP_MAX_ACTIVE_LEVELS']='1'
import argparse,hashlib,json,random,signal,time,uuid
from pathlib import Path
import numpy as np
import torch
from .io import HERE,WEIGHTS,WEIGHT_SHA,sha,save,sources,snapshot,exclusive,task_expired
from .data import load_rows,collate as base_collate
from .batches import inputs_only
from .model import GeometryModel,decode
from .head import RelationHead,collate,labels,loss
from datasets.reference_layouts import ReferenceLayouts,REGISTRY

LR=(1e-3,2.5e-4,1e-4,5e-5)


def checkpoint(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(value,temp)
    for attempt in range(20):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def rng_state():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}
def restore_rng(s):
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])


def read_finished_cache(folder,identity):
    manifest=json.loads((folder/'cache/manifest.json').read_text())
    if not manifest['finished'] or manifest['identity']!=identity:raise ValueError('Prepared cache identity/phase')
    result={}
    for role in ('train','dev'):
        expected=[list(r['member']) for r in ReferenceLayouts(role).records];rows=[];members=[]
        for entry in manifest['roles'][role]:
            path=folder/'cache'/entry['name']
            if path.stat().st_size!=entry['size']:raise ValueError('Prepared chunk changed')
            part=torch.load(path,map_location='cpu',weights_only=False)
            if part['identity']!=identity or len(part['rows'])!=entry['count']:raise ValueError('Prepared chunk identity')
            if part['members']!=[r['member'] for r in part['rows']]:raise ValueError('Prepared member payload')
            rows+=part['rows'];members+=part['members']
        if members!=expected:raise ValueError('Prepared logical member ordering')
        result[role]=rows
    return result


def cache(folder,base,identity,stopping):
    manifest_path=folder/'cache/manifest.json';(folder/'cache').mkdir(exist_ok=True)
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'identity':identity,'roles':{},'finished':False}
    if manifest['identity']!=identity:raise ValueError('Cache identity changed')
    result={};started=time.perf_counter()
    for role in ('train','dev'):
        originals=load_rows(role);parts=manifest['roles'].setdefault(role,[]);result[role]=[]
        for start in range(0,len(originals),128):
            if stopping():save(manifest_path,manifest);return None
            selected=originals[start:start+128];name=f'{role}_{start:05d}.pt';path=folder/'cache'/name
            if start//128<len(parts):
                part=torch.load(path,map_location='cpu',weights_only=False)
                if part['members']!=[list(r['member']) for r in selected] or part['identity']!=identity:raise ValueError('Cache member mismatch')
                result[role]+=part['rows'];continue
            rows=[]
            for offset in range(0,len(selected),32):
                batch=selected[offset:offset+32];g=base_collate(batch,'cuda')[0];holder={}
                handle=base.head.register_forward_pre_hook(lambda module,args:holder.__setitem__('latent',args[0].detach()))
                with torch.no_grad():
                    state=base(inputs_only(g));draft=decode(state,g).cpu();hidden=holder['latent'].cpu()
                handle.remove()
                for at,row in enumerate(batch):
                    n=row['graph']['n'];reference=row['graph']['reference']
                    rows.append({'n':n,'member':list(row['member']),'latent':hidden[at,:n].clone(),'draft':draft[at,:n].clone(),
                        'adjacency':row['graph']['adjacency'].clone(),'constraints':row['graph']['constraints'].clone(),'labels':labels(reference)})
            checkpoint(path,{'identity':identity,'members':[list(r['member']) for r in selected],'rows':rows})
            parts.append({'name':name,'count':len(rows),'size':path.stat().st_size});result[role]+=rows
            manifest['preparation_seconds_this_invocation']=time.perf_counter()-started;save(manifest_path,manifest)
            if start%1024==0:print(f'CACHE {role} {start+len(rows)}/{len(originals)}',flush=True)
    manifest['finished']=True;save(manifest_path,manifest);return result


@torch.no_grad()
def monitor(head,rows,stopping):
    head.eval();total=0.;correct=0;entries=0
    for start in range(0,len(rows),32):
        if stopping():return None
        batch=rows[start:start+32];g,y,mask=collate(batch,'cuda');z=head(g);total+=float(loss(z,y,mask))*len(batch)
        valid=mask>0;correct+=int(((z>0)==y.bool())[valid].sum());entries+=int(valid.sum())*4
    return {'BCE':total/len(rows),'direction_accuracy':correct/entries,'count':len(rows)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--recover-state-json',action='store_true',help='Audited same-feature recovery after initial infinite JSON state')
    ap.add_argument('--max-train-seconds',type=float,default=1200);ap.add_argument('--max-epochs',type=int,default=200);a=ap.parse_args()
    if '/' in a.tag or '\\' in a.tag:raise ValueError('Bad tag')
    if task_expired() or not torch.cuda.is_available():raise RuntimeError('Deadline or CUDA unavailable')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0);random.seed(0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=HERE/'runs'/a.tag;folder.mkdir(exist_ok=True);(folder/'checkpoints').mkdir(exist_ok=True)
    identity={'revision':'E47.frozen_reference_probe.v1','sources':sources(),'checkpoint_sha256':sha(WEIGHTS),'registry_sha256':sha(REGISTRY),
        'seed':0,'batch_size':32,'lr_levels':LR,'max_train_seconds':a.max_train_seconds,'max_epochs':a.max_epochs,
        'calibration':'whole train shards with sha256(E47calib+shard) modulo 10 equal 0; never gradient',
        'target':'four non-exclusive reference separation validity labels; not optimization benefit or MIB legality confidence',
        'selection':'fixed dev_monitor500 per-case BCE; no postprocessing during training or monitoring','test_read':False}
    # JSON normalizes tuples on disk; compare only normalized identities.
    identity=json.loads(json.dumps(identity));path=folder/'training.json'
    if identity['checkpoint_sha256']!=WEIGHT_SHA:raise ValueError('Frozen model changed')
    migration_path=folder/'state_fix_migration.json';migration=json.loads(migration_path.read_text()) if migration_path.exists() else None
    recovery_old=None
    if a.resume and a.recover_state_json and not path.exists() and (folder/'checkpoints/latest.pt').exists():
        prior_checkpoint=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False);old_identity=prior_checkpoint['identity']
        old_config={k:v for k,v in old_identity.items() if k!='sources'};new_config={k:v for k,v in identity.items() if k!='sources'}
        changed={name for name in set(old_identity['sources'])|set(identity['sources']) if old_identity['sources'].get(name)!=identity['sources'].get(name)}
        if old_config!=new_config or changed!={'experiments/E47_draft_topology_confidence/head_train.py'}:raise ValueError('State-only recovery scope changed')
        if prior_checkpoint['state']['history'] or prior_checkpoint['state']['best_epoch'] is not None:raise ValueError('Recovery is only before the first completed validation')
        recovery_old=old_identity;recovery=folder/'state_fix';recovery.mkdir(exist_ok=True);(recovery/'checkpoints').mkdir(exist_ok=True)
        old_latest=folder/'checkpoints/latest.pt';(recovery/'checkpoints/latest_before_fix.pt').write_bytes(old_latest.read_bytes());snapshot(recovery,identity)
        migration={'old_identity':old_identity,'new_identity':identity,'reason':'Initial best inf cannot be JSON; finite sentinel and read-only finished-cache reuse; feature/label/model sources unchanged',
            'preserved_cursor':prior_checkpoint['state']['cursor'],'preserved_train_seconds':prior_checkpoint['state']['cumulative_train_seconds'],
            'old_checkpoint_sha256':sha(recovery/'checkpoints/latest_before_fix.pt'),'budget_reset':False}
        save(migration_path,migration)
    if path.exists():
        old=json.loads(path.read_text())
        if not a.resume or old['identity']!=identity:raise ValueError('Use same-identity resume')
        if old.get('finished'):print('Training already finished');return
    elif a.resume and recovery_old is None:raise FileNotFoundError(path)
    else:snapshot(folder,identity)
    stop={'value':False};signal.signal(signal.SIGINT,lambda sig,frame:stop.__setitem__('value',True))
    def external_stop():return stop['value'] or task_expired() or (HERE/'runs/STOP').exists() or (folder/'STOP').exists()
    with exclusive(folder/'.train.lock'):
        frozen=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**frozen['model_config']).eval().cuda();base.load_state_dict(frozen['model'])
        for param in base.parameters():param.requires_grad_(False)
        cache_source=migration['old_identity'] if migration else identity
        cache_identity={k:v for k,v in cache_source.items() if k not in ('max_train_seconds','max_epochs')}
        manifest_path=folder/'cache/manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()).get('finished'):prepared=read_finished_cache(folder,cache_identity)
        else:prepared=cache(folder,base,cache_identity,external_stop)
        if prepared is None:save(path,{'identity':identity,'finished':False,'phase':'cache_stopped'});return
        del base;torch.cuda.empty_cache()
        train,dev=prepared['train'],prepared['dev'];assert len(train)==16000 and len(dev)==2000
        cal=[i for i,r in enumerate(train) if int(hashlib.sha256(('E47calib'+r['member'][0]).encode()).hexdigest(),16)%10==0]
        cal_set=set(cal);gradient=[i for i in range(len(train)) if i not in cal_set]
        by_train={tuple(r['member']):r for r in train};by_dev={tuple(r['member']):r for r in dev}
        train_monitor=[by_train[tuple(r['member'])] for r in ReferenceLayouts('train_monitor500').records]
        dev_monitor=[by_dev[tuple(r['member'])] for r in ReferenceLayouts('dev_monitor500').records]
        head=RelationHead().cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=LR[0],weight_decay=0.)
        state={'epoch':1,'cursor':0,'order':None,'cumulative_train_seconds':0.,'best':1e9,'best_epoch':None,'drops':0,'scheduler_best':None,
            'lr_bad':0,'early_bad':0,'last_drop_epoch':None,'history':[],'gradient_cases':len(gradient),'calibration_cases':len(cal),
            'train_monitor_gradient_cases':sum(tuple(r['member']) in {tuple(train[i]['member']) for i in gradient} for r in train_monitor)}
        latest=folder/'checkpoints/latest.pt'
        if a.resume and latest.exists():
            ck=torch.load(latest,map_location='cpu',weights_only=False)
            if ck['identity']!=identity and (migration is None or ck['identity']!=migration['old_identity']):raise ValueError('Checkpoint identity')
            head.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);state=ck['state'];restore_rng(ck['rng'])
            if not np.isfinite(state['best']):state['best']=1e9
            if ck.get('unclean_reserve'):state['cumulative_train_seconds']+=ck['unclean_reserve']
        prior=state['cumulative_train_seconds'];started=time.perf_counter()
        def stopping():return external_stop() or prior+time.perf_counter()-started>=a.max_train_seconds or task_remaining()<2700
        def persist(clean=False):
            state['cumulative_train_seconds']=prior+time.perf_counter()-started
            checkpoint(latest,{'identity':identity,'model':head.state_dict(),'optimizer':optimizer.state_dict(),'state':state,'rng':rng_state(),'unclean_reserve':0 if clean else 30})
            save(path,{'identity':identity,'state':state,'finished':False,'phase':'training','parameters':sum(p.numel() for p in head.parameters())})
        reason='max_epochs'
        while state['epoch']<=a.max_epochs:
            if stopping():reason='resource_or_user_stop';break
            if state['order'] is None:state['order']=torch.randperm(len(gradient)).tolist();state['cursor']=0
            head.train()
            while state['cursor']<len(gradient):
                if stopping():break
                ids=state['order'][state['cursor']:state['cursor']+32];rows=[train[gradient[k]] for k in ids]
                g,y,mask=collate(rows,'cuda');optimizer.zero_grad(set_to_none=True);value=loss(head(g),y,mask)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite head loss')
                value.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();state['cursor']+=len(ids)
                if state['cursor']%1024<32:persist()
            if state['cursor']<len(gradient):reason='resource_or_user_stop';break
            tm=monitor(head,train_monitor,stopping);dm=monitor(head,dev_monitor,stopping)
            if tm is None or dm is None:reason='resource_or_user_stop';break
            epoch=state['epoch'];entry={'epoch':epoch,'train_monitor':tm,'dev_monitor':dm,'lr':optimizer.param_groups[0]['lr']};state['history'].append(entry)
            if dm['BCE']<state['best']:
                state['best']=dm['BCE'];state['best_epoch']=epoch
                checkpoint(folder/'checkpoints/best.pt',{'identity':identity,'model':head.state_dict(),'epoch':epoch,'validation':dm})
            if epoch>=15:
                old=state['scheduler_best'];meaningful=old is None or dm['BCE']<old-max(1e-5,abs(old)*1e-3)
                if meaningful:state['scheduler_best']=dm['BCE'];state['lr_bad']=state['early_bad']=0
                else:state['lr_bad']+=1;state['early_bad']+=1
                if state['lr_bad']>=3 and state['drops']<3:
                    state['drops']+=1;state['lr_bad']=0;state['last_drop_epoch']=epoch
                    for group in optimizer.param_groups:group['lr']=LR[state['drops']]
            print(f"EPOCH {epoch} train={tm['BCE']:.6f} dev={dm['BCE']:.6f} best={state['best']:.6f} drops={state['drops']}",flush=True)
            state['epoch']+=1;state['cursor']=0;state['order']=None;persist()
            if state['drops']==3 and state['early_bad']>=3 and epoch-state['last_drop_epoch']>=2:reason='early_stop';break
        persist(clean=True)
        if not (folder/'checkpoints/best.pt').exists():save(path,{'identity':identity,'state':state,'finished':False,'phase':'no_completed_epoch','stop_reason':reason});return
        chosen=torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False);head.load_state_dict(chosen['model']);head.eval()
        full_dev=monitor(head,dev,external_stop)
        # One scalar calibration on whole-shard held-out train only, fixed best.
        logits=[];targets=[];weights=[]
        with torch.no_grad():
            for start in range(0,len(cal),32):
                g,y,mask=collate([train[i] for i in cal[start:start+32]],'cuda');z=head(g);valid=mask>0
                logits.append(z[valid]);targets.append(y[valid]);weights.append(mask[valid])
        z,y,w=torch.cat(logits),torch.cat(targets),torch.cat(weights);temperature=torch.nn.Parameter(torch.zeros((),device='cuda'))
        calibration_optimizer=torch.optim.LBFGS([temperature],lr=.5,max_iter=30,line_search_fn='strong_wolfe')
        def closure():
            calibration_optimizer.zero_grad();v=(torch.nn.functional.binary_cross_entropy_with_logits(z/temperature.exp().clamp(.25,4),y,reduction='none').mean(-1)*w).sum()/len(cal)
            v.backward();return v
        calibration_optimizer.step(closure);scale=float(temperature.exp().clamp(.25,4).detach())
        save(folder/'calibration.json',{'temperature':scale,'source':'whole-shard train holdout only','cases':len(cal),'head_best_sha256':sha(folder/'checkpoints/best.pt'),
            'confidence_semantics':'reference-direction validity, not Q gain probability','test_read':False})
        save(path,{'identity':identity,'state':state,'finished':True,'phase':'complete','stop_reason':reason,'full_dev_reference_proxy':full_dev,
            'frozen_checkpoint_unchanged':sha(WEIGHTS)==WEIGHT_SHA,'temperature':scale,'trained_parameters':sum(p.numel() for p in head.parameters()),'test_read':False})
        print(f"PROBE COMPLETE best={state['best_epoch']} reason={reason} temperature={scale:.4f}",flush=True)


def task_remaining():
    from datetime import datetime,timezone
    budget=json.loads((HERE/'runs/budget.json').read_text());return (datetime.fromisoformat(budget['deadline_utc'].replace('Z','+00:00'))-datetime.now(timezone.utc)).total_seconds()


if __name__=='__main__':main()
