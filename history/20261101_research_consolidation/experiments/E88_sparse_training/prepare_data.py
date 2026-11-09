"""User-run public-only sparse selector; independent cached-data copies, four CPU threads."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,time,signal,threading,sys,subprocess,zipfile
from datetime import datetime,timezone
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import torch
from .io import HERE,ROOT,sha,save,sources,exclusive
from .cache import verify,resolve_part

OUTPUT=HERE/'runs/20261031_sparse_pool'

def public_statistics(index,row):
    n,_,bb,pb,_,c,_=row['case'].solver_inputs;n=int(n)
    bb=np.asarray(bb);pb=np.asarray(pb);c=np.asarray(c)
    bb=bb[bb[:,2]>0] if len(bb) else bb
    pairs={tuple(sorted((int(e[0]),int(e[1])))) for e in bb if int(e[0])!=int(e[1])}
    ports={int(e[1]) for e in pb if float(e[2])>0}
    return dict(index=index,member=list(row['member']),n=n,density=2*len(pairs)/(n*(n-1)),
        port_coverage=len(ports)/n,preplaced_fraction=float((c[:,1]!=0).mean()),
        locked_fraction=float(c[:,:2].any(1).mean()),boundary_fraction=float((c[:,4]!=0).mean()))

def stratum(record):
    # Match exact N and weak-port coverage bins, not development performance.
    return record['n'],min(2,int(record['port_coverage']/.1000000001))

def rank(record):
    return hashlib.sha256(('E88.seed0:'+repr(record['member'])).encode()).hexdigest()

def select_profiles(records,maximum):
    base=[r for r in records if r['index']<16000]
    cut={n:float(np.median([r['density'] for r in base if r['n']==n])) for n in sorted({r['n'] for r in base})}
    old=defaultdict(list);new=defaultdict(list)
    for row in records:
        if 16000<=row['index']<32000:continue
        if row['port_coverage']>.30 or row['density']>cut[row['n']]:continue
        (old if row['index']<16000 else new)[stratum(row)].append(row)
    support=sorted(set(old)&set(new));picked={key:[] for key in support}
    queues={key:sorted(new[key],key=rank) for key in support}
    # Round-robin exact-N/coverage strata, bounded by available genuine new cases.
    while sum(map(len,picked.values()))<maximum:
        changed=False
        for key in support:
            if len(picked[key])<len(queues[key]) and sum(map(len,picked.values()))<maximum:
                picked[key].append(queues[key][len(picked[key])]);changed=True
        if not changed:break
    selected=[row for key in support for row in picked[key]]
    if not selected:raise ValueError('No matched sparse support; do not move dev/test members')
    compact=lambda r:{k:r[k] for k in ('index','member','n','density','port_coverage')}
    return dict(train_density_cutoffs={str(k):v for k,v in cut.items()},selected_count=len(selected),
        new_members=[compact(r) for r in selected],
        matched_strata=[dict(n=k[0],coverage_bin=k[1],slot_weight=len(picked[k]),
            old_members=[compact(r) for r in sorted(old[k],key=rank)],new_indices=[r['index'] for r in picked[k]])
            for k in support if picked[k]],
        base_sparse_count=sum(map(len,old.values())),candidate_sparse_count=sum(map(len,new.values())),
        unsupported_old_strata=[list(k) for k in sorted(set(old)-set(new))],
        matched_features=['exact module count','weak-port coverage bin'],
        caveat='Other public constraint burden is recorded, not guaranteed identical')

def preparation_context():
    # Data profiles/selection do not depend on an as-yet unfinished training run.
    return dict(mode='public_data_preparation_only',future_training_parent=None,
        known_pretrained_source='E74 lineage',excluded_train_prefix=32000,
        authorization='2026-10-31 user explicitly requested background data scan before training')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--max-new',type=int,default=4000);a=p.parse_args()
    if not 1<=a.max_new<=4000:raise ValueError('At most4000 new members')
    if a.dry_run:
        print(json.dumps(dict(stage='data preparation independent of E87 completion',public_profile_cases=48000,base_indices=[0,15999],new_indices=[32000,63999],
            excludes_parent_seen_prefix32000=True,threads=4,max_new=a.max_new,test_read=False),indent=2));return
    with exclusive(HERE/'runs/data_preparation.lock'):run(a)

def run(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    begin=time.perf_counter();event=threading.Event();old_handler=signal.signal(signal.SIGINT,lambda *_:event.set())
    stopped=lambda:event.is_set() or (OUTPUT/'STOP').exists() or (HERE/'runs/STOP').exists()
    progress=dict(status='running',phase='verifying_inputs',pid=os.getpid(),profile_cases=0,total_profile_cases=48000,
        threads=4,numeric_threads_per_worker=1,training_started=False,test_read=False)
    def state(**values):
        progress.update(values);progress.update(elapsed_seconds=time.perf_counter()-begin,updated_utc=datetime.now(timezone.utc).isoformat())
        save(OUTPUT/'progress.json',progress)
    try:_run_scan(a,begin,stopped,event,state)
    except BaseException as error:
        event.set();state(status='failed',error=repr(error));raise
    finally:signal.signal(signal.SIGINT,old_handler)

def _run_scan(a,begin,stopped,event,state):
    state();manifest=verify('train')
    identity=dict(sources=sources(),context=preparation_context(),max_new=a.max_new,threads=4,train_only=True,dev_read=False,test_read=False)
    config=OUTPUT/'config.json'
    if config.exists():
        if json.loads(config.read_text(encoding='utf-8'))!=identity:raise ValueError('Sparse preparation identity changed')
    else:save(config,identity)
    snapshot=OUTPUT/'source_snapshot.zip'
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(HERE.glob('*.py')):archive.write(path,path.relative_to(ROOT).as_posix())
        save(OUTPUT/'invocation.json',dict(command=[sys.executable,*sys.argv],cwd=str(ROOT),pid=os.getpid(),
            started_utc=datetime.now(timezone.utc).isoformat(),source_snapshot_sha256=sha(snapshot),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            uncommitted_sources_preserved=True,training_started=False,test_read=False))
    final=OUTPUT/'sparse_members.json'
    if final.exists():
        if json.loads(final.read_text(encoding='utf-8'))['identity']!=identity:raise ValueError('Frozen sparse manifest changed')
        existing=json.loads(final.read_text(encoding='utf-8'))
        state(status='complete',phase='complete',profile_cases=48000,selected_count=existing['selected_count'])
        print('Existing sparse manifest verified');return
    needed=lambda i:i<16000 or i>=32000
    def process(payload):
        if stopped():return None
        number,entry=payload;dest=OUTPUT/f'public_profiles/part_{number:04d}.json'
        if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))
        chunk=torch.load(resolve_part(entry['path']),map_location='cpu',weights_only=False,mmap=True)
        if chunk['identity']!=manifest['part_identities'][entry['identity_key']] or chunk['indices']!=entry['indices']:raise ValueError('Cache identity mismatch')
        if len(chunk['rows'])!=len(chunk['indices']):raise ValueError('Cache row count mismatch')
        for index,row in zip(chunk['indices'],chunk['rows']):
            if list(row['member'])!=manifest['members'][index]:raise ValueError('Cached member mismatch')
        values=[]
        for i,row in zip(chunk['indices'],chunk['rows']):
            if stopped():return None
            if needed(i):values.append(public_statistics(i,row))
        save(dest,values);return values
    tasks=[(i,e) for i,e in enumerate(manifest['parts']) if any(needed(j) for j in e['indices'])]
    records=[];completed=0
    state(phase='scanning',total_parts=len(tasks))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(process,payload) for payload in tasks]
        try:
            for f in as_completed(futures):
                values=f.result()
                if values is not None:records.extend(values);completed+=1
                state(profile_cases=len(records),completed_parts=completed)
                print('public profiles',len(records),'/48000',flush=True)
                if stopped():
                    for pending in futures:pending.cancel()
                    break
        except BaseException:
            event.set()
            for pending in futures:pending.cancel()
            raise
    if stopped():state(status='stopped',phase='stopped');return
    if len(records)!=48000 or len({r['index'] for r in records})!=48000:raise ValueError('Public profile coverage mismatch')
    state(phase='selecting');result=select_profiles(records,a.max_new)
    profile_hashes={p.relative_to(OUTPUT).as_posix():sha(p) for p in sorted((OUTPUT/'public_profiles').glob('*.json'))}
    save(final,dict(identity=identity,**result,profile_hashes=profile_hashes,
        sampler='shared 80% base round-robin +20% matched stratum slots; repeat old versus genuine new; fixed update budget',
        parent_exposure='E74 known32k prefix excluded from genuinely new candidates',seconds=time.perf_counter()-begin,
        no_free_reference_fields_analyzed=True,test_read=False))
    state(status='complete',phase='complete',profile_cases=48000,selected_count=result['selected_count'])
    print('Frozen new sparse members:',result['selected_count'],flush=True)

if __name__=='__main__':main()
