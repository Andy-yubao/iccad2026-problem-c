"""Fresh random 36k/2k/2k MIB-compatible members and train/dev CPU caches."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import collections
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import pathlib
import time
import traceback
import numpy as np
import torch
from datasets.paths import ROOT,dataset_paths
from .transform import compatible_areas,supervision
from .batches import sample
from .ports import public_ports
from .role_features import build

HERE=pathlib.Path(__file__).resolve().parent
SEED=2026111901
TARGETS=dict(train=36000,dev=2000,test=2000)
RAW=dataset_paths()['training_root']


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):
            h.update(b)
    return h.hexdigest()


def save(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(tmp,p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def inspect(task):
    torch.set_num_threads(1)
    if (HERE/'STOP').exists():raise RuntimeError('User STOP')
    role,shard,requests=task
    path=RAW/shard;before=path.stat()
    raw=torch.load(path,map_location='cpu',weights_only=False)
    assert len(raw[0])==112
    accepted=[];rejected=collections.Counter();hist=collections.Counter()
    for rank,index in requests:
        f=raw[0][index].detach().cpu().float()
        n=len(f);areas=f[:,0].numpy();flags=f[:,1:].numpy()
        locked=(flags[:,:2]!=0).any(1)
        # These WH are public specifications. Free WHXY and metrics are not indexed.
        public_wh=np.ones((n,2),np.float32)
        if locked.any():
            public_wh[locked]=raw[5][index][torch.from_numpy(locked),:2].detach().cpu().float().numpy()
        adjusted,reason=compatible_areas(areas,flags,public_wh)
        hist[n]+=1
        if reason:
            rejected[reason]+=1
        else:
            accepted.append(dict(rank=rank,member=[shard,index],n=n))
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return dict(role=role,shard=shard,accepted=accepted,rejected=dict(rejected),scanned=len(requests),n_counts=dict(hist),
                source=dict(size=after.st_size,mtime_ns=after.st_mtime_ns,sha256=sha(path)))


def baseline(reference,b2b,p2b,pins,metrics):
    p=reference.numpy().astype(float);center=p[:,:2]+p[:,2:]/2
    b=np.asarray(b2b);q=np.asarray(p2b);ports=np.asarray(pins)
    b=b[~(b==-1).all(1)];q=q[~(q==-1).all(1)]
    wire_b=float((np.abs(center[b[:,0].astype(int)]-center[b[:,1].astype(int)]).sum(1)*b[:,2]).sum()) if len(b) else 0.
    wire_p=float((np.abs(ports[q[:,0].astype(int),:2]-center[q[:,1].astype(int)]).sum(1)*q[:,2]).sum()) if len(q) else 0.
    area=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    m=np.asarray(metrics)
    if len(m)>=8:
        if m[0]>0:area=float(m[0])
        if m[-2]>0:wire_b=float(m[-2])
        if m[-1]>=0:wire_p=float(m[-1])
    return dict(area_baseline=area,hpwl_baseline=wire_b+wire_p)


def cache_part(task):
    torch.set_num_threads(1)
    if (HERE/'STOP').exists():raise RuntimeError('User STOP')
    role,part_id,groups,identity=task
    assert role in ('train','dev'), 'New test free labels and caches remain sealed'
    output=[];indices=[];audit=collections.Counter();profile=dict(max_modules=0,max_ports=0,max_pins=0)
    for shard,requests,source in groups:
        if (HERE/'STOP').exists():raise RuntimeError('User STOP')
        path=RAW/shard;st=path.stat()
        assert (st.st_size,st.st_mtime_ns)==(source['size'],source['mtime_ns']) and sha(path)==source['sha256']
        raw=torch.load(path,map_location='cpu',weights_only=False)
        for i,index,expected_n in requests:
            f=raw[0][index].detach().cpu().float();flags=f[:,1:].clone();n=len(f)
            assert n==expected_n
            original=raw[5][index][:,[2,3,0,1]].detach().cpu().float().clone()
            locked=flags[:,:2].bool().any(1);pre=flags[:,1]!=0
            specs=torch.full((n,4),-1.,dtype=torch.float32)
            specs[locked,2:]=original[locked,2:];specs[pre]=original[pre]
            areas,reason=compatible_areas(f[:,0],flags,specs[:,2:])
            assert reason is None
            inputs=(n,torch.from_numpy(areas),*[raw[k][index].detach().cpu().float().clone() for k in (1,2,3)],flags,specs)
            teacher,checks=supervision(original,inputs[1],flags,specs)
            graph=sample(inputs,teacher);graph.update(public_ports(inputs,graph['scale']))
            roles=build(graph)
            assert graph['features'].shape==(n,17) and bool((graph['mask']==1).all())
            for label in torch.unique(flags[:,2]):
                if label>0:
                    mask=flags[:,2]==label
                    assert bool((inputs[1][mask]==inputs[1][mask][0]).all())
                    assert float(graph['target'][mask,2].max()-graph['target'][mask,2].min())<2e-6
            audit.update(checks);audit['cases_with_corrected_overlap']+=checks['corrected_overlap_pairs']>0
            audit['cases_with_corrected_negative_coordinates']+=checks['corrected_negative_modules']>0
            audit['cases_with_original_mib_shape_disagreement']+=checks['original_mib_shape_disagreement_groups']>0
            audit['cases']+=1
            profile['max_modules']=max(profile['max_modules'],n)
            profile['max_ports']=max(profile['max_ports'],len(roles['port_features']))
            profile['max_pins']=max(profile['max_pins'],graph['pin_coordinates'].shape[1])
            output.append(dict(member=(shard,index),graph=graph,role=roles,inputs=inputs,
                reference_xywh=original,supervision_xywh=teacher,reference_audit=checks,
                baseline_metrics=baseline(original,inputs[2],inputs[3],inputs[4],raw[6][index])))
            indices.append(i)
    destination=HERE/f'cache/{role}/{part_id:04d}.pt';destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.pt.tmp');torch.save(dict(identity=identity,indices=indices,rows=output),tmp);os.replace(tmp,destination)
    result=dict(path=str(destination.relative_to(HERE)),sha256=sha(destination),indices=indices,count=len(indices),audit=dict(audit),profile=profile)
    save(destination.with_suffix('.json'),result)
    return result


def stratified(records,count,seed):
    rng=np.random.default_rng(seed);groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[r['n']].append(i)
    desired={n:count*len(ids)/len(records) for n,ids in groups.items()}
    quota={n:int(v) for n,v in desired.items()}
    while sum(quota.values())<count:
        n=max(groups,key=lambda n:(desired[n]-quota[n],n));quota[n]+=1
    chosen=[]
    for n in sorted(groups):chosen.extend(rng.choice(groups[n],quota[n],replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def main():
    started=time.perf_counter();torch.set_num_threads(1)
    gate=read(HERE/'preflight.json')
    assert gate['status']=='passed' and gate['sources']=={p.name:sha(p) for p in HERE.glob('*.py')}, 'Preflight/source mismatch'
    assert not (HERE/'state.json').exists(), 'Preserve prior run; no implicit overwrite/retry'
    shards=sorted(str(p.relative_to(RAW)).replace('\\','/') for p in RAW.glob('worker_*/layouts_*.th'))
    assert len(shards)==9000 and len(set(shards))==9000
    rng=np.random.default_rng(SEED);rng.shuffle(shards)
    pools=dict(train=shards[:8100],dev=shards[8100:8550],test=shards[8550:])
    plan=dict(version='random_mib40k.v1',seed=SEED,targets=TARGETS,shard_partitions=pools,
        historical_overlap='Explicitly allowed by user; no exclusion by historical membership',
        compatibility='Public MIB mean-area; reject immutable WH disagreement or >=0.9999% mean-area vs locked-shape mismatch; retain public specs',
        target_policy='Preserve original centers; same-MIB shared mean original log-aspect, or public immutable WH if present; original reference separate; no legality claim',
        label_access='New train/dev only. New test public features and immutable WH only; no test free WHXY/metrics indexing or cache',
        training_authorized=False,workers=4,selection_uses_loss=False,selection_uses_Q=False)
    save(HERE/'plan.json',plan)
    producer_hashes={p.name:sha(p) for p in HERE.glob('*.py')}
    state=dict(status='running',phase='public_compatibility_scan',pid=os.getpid(),scanned=0,compatible_seen=0,rejections={},selected={},test_free_labels_read=False,training_started=False)
    save(HERE/'state.json',state)
    all_sources={};sets={};scan_stats={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for role,target in TARGETS.items():
            candidates=np.random.default_rng(SEED+{'train':11,'dev':22,'test':33}[role]).permutation(len(pools[role])*112)
            cursor=0;accepted=[];rejections=collections.Counter();hist=collections.Counter()
            block=100000 if role=='train' else 6000
            while len(accepted)<target:
                end=min(len(candidates),cursor+block)
                if end==cursor:raise RuntimeError('Compatible candidate pool exhausted: '+role)
                grouped=collections.defaultdict(list)
                for rank in range(cursor,end):
                    flat=int(candidates[rank]);grouped[pools[role][flat//112]].append((rank,flat%112))
                jobs=[(role,s,v) for s,v in sorted(grouped.items())]
                pending=[pool.submit(inspect,t) for t in jobs]
                for future in as_completed(pending):
                    result=future.result();accepted.extend(result['accepted']);rejections.update(result['rejected']);hist.update(result['n_counts'])
                    source=result['source']
                    if result['shard'] in all_sources:assert all_sources[result['shard']]==source
                    all_sources[result['shard']]=source
                    state['scanned']+=result['scanned'];state['compatible_seen']+=len(result['accepted'])
                    state['role']=role;state['rejections']=dict(rejections)
                    if state['scanned']//2000!=(state['scanned']-result['scanned'])//2000:
                        save(HERE/'state.json',dict(state,elapsed_seconds=time.perf_counter()-started))
                        print(f'[SCAN] role={role} total_scanned={state["scanned"]} compatible_in_role={len(accepted)}',flush=True)
                    if (HERE/'STOP').exists():raise RuntimeError('User STOP; do not continue')
                cursor=end;block=10000 if role=='train' else 3000
            accepted.sort(key=lambda r:r['rank'])
            records=[dict(id=f'{role}_{i:05d}',member=r['member'],n=r['n']) for i,r in enumerate(accepted[:target])]
            sets[role]=dict(count=target,records=records,n_counts=dict(collections.Counter(r['n'] for r in records)))
            scan_stats[role]=dict(scanned=cursor,compatible_seen=len(accepted),rejections=dict(rejections),scanned_n_counts=dict(hist))
            state['selected'][role]=target;save(HERE/'state.json',state)
            print(f'[SELECTED] {role}={target} scanned={cursor} rejected={dict(rejections)}',flush=True)
        members=[{tuple(r['member']) for r in sets[role]['records']} for role in TARGETS]
        assert not any(members[a]&members[b] for a,b in ((0,1),(0,2),(1,2)))
        panels={role:stratified(sets[parent]['records'],500,SEED+offset) for role,parent,offset in [('train_monitor500','train',101),('dev_monitor500','dev',202)]}
        manifest=dict(**plan,sets=sets,panels=panels,scan_statistics=scan_stats,source_shards=all_sources,producer_hashes=producer_hashes,
            task_version='MIB.mean_area.shared_shape_supervision.v1',test_labels_sealed=True,created_unix=time.time())
        identity=digest(manifest);manifest['manifest_sha256']=identity
        save(HERE/'manifest.json',manifest);save(ROOT/'protocol/random_mib40k_v1.json',manifest)
        tasks=[]
        for role in ('train','dev'):
            groups=collections.defaultdict(list)
            for i,r in enumerate(sets[role]['records']):groups[r['member'][0]].append((i,r['member'][1],r['n']))
            ordered=[(s,v,all_sources[s]) for s,v in sorted(groups.items())]
            for j in range(0,len(ordered),16):tasks.append((role,j//16,ordered[j:j+16],identity))
        pending={pool.submit(cache_part,t):t[0] for t in tasks};results=dict(train=[],dev=[])
        state.update(phase='train_dev_cache',completed=0,total=38000,rejections={r:s['rejections'] for r,s in scan_stats.items()})
        save(HERE/'state.json',state)
        for future in as_completed(pending):
            part=future.result();role=pending[future];results[role].append(part);state['completed']+=part['count']
            save(HERE/'state.json',dict(state,elapsed_seconds=time.perf_counter()-started))
            print(f'[CACHE] {state["completed"]}/38000 {role} {part["path"]}',flush=True)
            if (HERE/'STOP').exists():raise RuntimeError('User STOP; do not continue')
    totals=collections.Counter();profile=collections.Counter()
    for role,parts in results.items():
        assert sorted(i for p in parts for i in p['indices'])==list(range(TARGETS[role]))
        for p in parts:
            totals.update(p['audit'])
            for k,v in p['profile'].items():profile[k]=max(profile[k],v)
        save(HERE/f'cache/{role}/manifest.json',dict(identity=identity,members=[r['member'] for r in sets[role]['records']],parts=sorted(parts,key=lambda p:p['path'])))
    assert not (HERE/'cache/test').exists()
    assert producer_hashes=={p.name:sha(p) for p in HERE.glob('*.py')}, 'Producers changed during preparation'
    save(HERE/'cache/profile.json',dict(profile));save(HERE/'reference_audit.json',dict(totals))
    report=dict(status='complete',identity=identity,counts=TARGETS,cache_counts=dict(train=36000,dev=2000,test=0),
        scanned=sum(v['scanned'] for v in scan_stats.values()),scan_statistics=scan_stats,reference_audit=dict(totals),profile=dict(profile),
        elapsed_seconds=time.perf_counter()-started,training_started=False,test_free_labels_read=False)
    save(HERE/'report.json',report)
    state.update(status='complete',phase='complete',elapsed_seconds=time.perf_counter()-started,identity=identity)
    save(HERE/'state.json',state)
    print('[COMPLETE] '+json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(HERE/'error.json',dict(error=repr(error),traceback=traceback.format_exc()))
        save(HERE/'state.json',dict(status='error',error=repr(error),pid=os.getpid()))
        raise
