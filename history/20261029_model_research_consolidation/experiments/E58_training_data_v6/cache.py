"""Resumable four-process CPU tensor cache; v5 train/dev parts are referenced once."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import torch
from datasets.expanded_reference import ExpandedReferenceLayouts,registry
from .io import ROOT,HERE,RUN,CACHE,REGISTRY,read,save,sha,replace,verify_frozen,exclusive


def identity():
    names=['features.py','batches.py','data.py','cache.py','io.py']
    return {'format':'E58.expanded_reference_tensors.v6','registry_sha256':sha(REGISTRY),
        'paths_sha256':sha(ROOT/'protocol/paths.json'),
        'feature_sources':{n:sha(HERE/n) for n in names},
        'dataset_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in [
            ROOT/'datasets/expanded_reference/__init__.py',ROOT/'datasets/reference_layouts.py',ROOT/'datasets/worksets.py',ROOT/'datasets/cases.py']}}


def initialize():
    global DATA,IDENTITY
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    DATA=ExpandedReferenceLayouts('train');IDENTITY=identity()


def prepare_part(ids,path_string):
    from .data import prepare
    path=Path(path_string);rows=[]
    audit={'cases':0,'nontrivial_groups':0,'changed_groups':0,'grouped_blocks':0,'max_group_total_relative_error':0.}
    for index in ids:
        item=DATA[index];a=item['mib_audit'];rows.append(prepare(item));audit['cases']+=1
        for key in ('nontrivial_groups','changed_groups','grouped_blocks'):audit[key]+=a[key]
        audit['max_group_total_relative_error']=max(audit['max_group_total_relative_error'],a['max_group_total_relative_error'])
    temp=path.with_suffix('.tmp');torch.save({'identity':IDENTITY,'indices':ids,'rows':rows},temp);replace(temp,path)
    st=path.stat();record={'path':path.relative_to(ROOT).as_posix(),'indices':ids,'count':len(ids),
        'identity_key':'v6','size':st.st_size,'mtime_ns':st.st_mtime_ns,'mib_audit':audit}
    save(path.with_suffix('.json'),record)
    return record


def old_parts(role):
    old_root=ROOT/read(HERE/'provenance.json')['existing_cache'];m=read(old_root/role/'manifest.json')
    expected=registry()['sets'][role]['records'][:16000] if role=='train' else registry()['sets']['dev']['records']
    if m['members']!=[x['member'] for x in expected]:raise ValueError('v5 reused membership mismatch')
    for name in ('features.py','batches.py'):
        if sha(HERE/name)!=m['identity']['sources'][name]:raise ValueError('Reused feature semantics differ')
    order=sorted(range(len(expected)),key=lambda i:expected[i]['member']);entries=[]
    for part in m['parts']:
        path=old_root/role/part['name'];st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Reused v5 chunk changed')
        start=int(path.stem);ids=order[start:start+128]
        entries.append({'path':path.relative_to(ROOT).as_posix(),'indices':ids,'count':len(ids),
            'identity_key':'v5','size':st.st_size,'mtime_ns':st.st_mtime_ns})
    return m['identity'],entries


def verify(role):
    if role not in ('train','dev'):raise PermissionError('Test cache forbidden')
    manifest=read(CACHE/role/'manifest.json')
    if manifest['identity']!=identity():raise ValueError('Cache data/source identity changed')
    rows=registry()['sets'][role]['records']
    if manifest['members']!=[x['member'] for x in rows]:raise ValueError('Cache members changed')
    covered=[]
    for part in manifest['parts']:
        path=ROOT/part['path'];st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Cache part changed: '+str(path))
        covered.extend(part['indices'])
    if sorted(covered)!=list(range(len(rows))):raise ValueError('Missing or duplicate cache indices')
    return manifest


def load_rows(role,indices=None,*,train_size=64000):
    if role not in ('train','dev','train_monitor500','dev_monitor500'):raise PermissionError('Tests sealed')
    if train_size not in (16000,32000,64000):raise ValueError('Unknown frozen training view')
    data=registry();parent='train' if role.startswith('train') else 'dev'
    manifest=verify(parent);all_records=data['sets'][parent]['records']
    role_records=data['sets'][role]['records']
    if role=='train':role_records=role_records[:train_size]
    lookup={tuple(x['member']):i for i,x in enumerate(all_records)}
    selected=list(range(len(role_records))) if indices is None else list(indices)
    if any(i<0 or i>=len(role_records) for i in selected):raise IndexError('Outside frozen view')
    requested=[lookup[tuple(role_records[i]['member'])] for i in selected];needed=set(requested);found={}
    for entry in manifest['parts']:
        if not needed.intersection(entry['indices']):continue
        part=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False)
        if part['identity']!=manifest['part_identities'][entry['identity_key']] or part['indices']!=entry['indices']:
            raise ValueError('Chunk identity/indices changed')
        if len(part['rows'])!=len(part['indices']):raise ValueError('Chunk row count changed')
        for index,row in zip(part['indices'],part['rows']):
            if list(row['member'])!=all_records[index]['member']:raise ValueError('Cached logical member mismatch')
            if index in needed:found[index]=row
    if set(found)!=needed:raise ValueError('Requested cache incomplete')
    return [found[i] for i in requested]


def build():
    if (CACHE/'results.json').exists():
        verify('train');verify('dev');print('Cache already ready; no rebuild',flush=True);return
    CACHE.mkdir(parents=True,exist_ok=True);(CACHE/'train').mkdir(exist_ok=True);(CACHE/'dev').mkdir(exist_ok=True)
    current=identity();stamp=CACHE/'identity.json'
    if stamp.exists() and read(stamp)!=current:raise ValueError('Interrupted cache identity changed')
    save(stamp,current);data=registry();old_identity,existing=old_parts('train')
    tasks=[]
    for lo,hi,label in [(16000,32000,'add32'),(32000,64000,'add64')]:
        order=sorted(range(lo,hi),key=lambda i:data['sets']['train']['records'][i]['member'])
        for start in range(0,len(order),128):
            tasks.append((order[start:start+128],str(CACHE/'train'/f'{label}_{start:06d}.pt')))
    complete=[];todo=[]
    for ids,path_string in tasks:
        path=Path(path_string);sidecar=path.with_suffix('.json')
        if path.exists() and sidecar.exists():
            record=read(sidecar);st=path.stat()
            if record['indices']!=ids or (st.st_size,st.st_mtime_ns)!=(record['size'],record['mtime_ns']):
                raise ValueError('Completed expansion chunk changed')
            complete.append(record)
        else:todo.append((ids,path_string))
    requested=[False];old_handler=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    stopped=lambda:requested[0] or (RUN/'STOP').exists() or (CACHE/'STOP').exists()
    started=time.perf_counter()
    def progress(status):
        save(RUN/'cache_progress.json',{'status':status,'reused_train_cases':16000,'new_completed':sum(x['count'] for x in complete),
            'new_total':48000,'new_parts_completed':len(complete),'new_parts_total':len(tasks),
            'workers':4,'library_threads_per_worker':1,'seconds_this_invocation':time.perf_counter()-started,'test_read':False})
    try:
        progress('building')
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            iterator=iter(todo);pending={}
            def submit():
                if stopped():return
                item=next(iterator,None)
                if item:pending[pool.submit(prepare_part,*item)]=item
            for _ in range(4):submit()
            while pending:
                done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future);complete.append(future.result());progress('building');submit()
                    if len(complete)%10==0:print('New train cases cached:',sum(x['count'] for x in complete),'/48000',flush=True)
        if stopped() or len(complete)!=len(tasks):progress('stopped');return
        parts=existing+sorted(complete,key=lambda x:x['path'])
        save(CACHE/'train/manifest.json',{'identity':current,'count':64000,'members':[x['member'] for x in data['sets']['train']['records']],
            'parts':parts,'part_identities':{'v5':old_identity,'v6':current},'nested_train_views':[16000,32000,64000],'test_read':False})
        dev_identity,dev_parts=old_parts('dev')
        save(CACHE/'dev/manifest.json',{'identity':current,'count':2000,'members':[x['member'] for x in data['sets']['dev']['records']],
            'parts':dev_parts,'part_identities':{'v5':dev_identity},'test_read':False})
        verify('train');verify('dev');verify_frozen()
        audit={'audited_new_cases':48000,'nontrivial_groups':sum(x['mib_audit']['nontrivial_groups'] for x in complete),
            'changed_groups':sum(x['mib_audit']['changed_groups'] for x in complete),
            'max_group_total_relative_error':max(x['mib_audit']['max_group_total_relative_error'] for x in complete),
            'locked_specs_unchanged':True,'reused_old_cases':16000,'reused_dev_cases':2000}
        save(CACHE/'results.json',{'status':'complete','identity':current,'train_count':64000,'dev_count':2000,
            'new_train_cases':48000,'no_duplicate_old_tensor_payloads':True,'MIB_audit':audit,'test_read':False})
        progress('complete');print('v6 cache complete: train64000/dev2000; no test cache',flush=True)
    finally:signal.signal(signal.SIGINT,old_handler)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    with exclusive(CACHE.with_suffix('.lock')):build()
