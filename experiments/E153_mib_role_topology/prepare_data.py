"""Re-express trusted E58 CPU caches under predict-all MIB/role semantics."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import json,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from datasets.reference_layouts import digest
from .io import ROOT,HERE,sha,save,checkpoint,exclusive
from .batches import sample
from .ports import public_ports
from .role_features import build
OLD=ROOT/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6'
SOURCE=OLD/'runs/20261026_reference_v6_64k/cache'
REGISTRY=ROOT/'protocol/reference_workset_v6.json'
CACHE=HERE/'data_cache_v1'
PRODUCERS=('prepare_data.py','features.py','batches.py','ports.py','role_features.py')
def identity():return dict(schema='E153.mib32k.predict_all.roles.v1',registry_sha256=sha(REGISTRY),
    source_manifests={r:sha(SOURCE/r/'manifest.json') for r in ('train','dev')},
    producers={n:sha(HERE/n) for n in PRODUCERS},precision='FP32; original cached reference labels; no rounding',
    area='Existing frozen E14.mean_area.v1 inputs, MIB groups equal area; immutable specs retained',
    predict_all=True,labels_outside_inputs=True,test_read=False)
def resolve(name):
    if name.startswith('experiments/E58_training_data_v6/'):
        p=OLD/name[len('experiments/E58_training_data_v6/'):]
    elif name.startswith('experiments/E42_joint_regularization/'):
        p=ROOT/'history/20261103_final_delivery_consolidation'/name
    else:raise ValueError('Unregistered archived cache path')
    p=p.resolve();assert p.is_relative_to(ROOT);return p
def process(task):
    torch.set_num_threads(1)
    role,part,expected_identity,wanted,members,key=task
    p=resolve(part['path']);st=p.stat();assert (st.st_size,st.st_mtime_ns)==(part['size'],part['mtime_ns'])
    content=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    assert content['identity']==expected_identity and content['indices']==part['indices']
    output=[];ids=[];groups=0
    for index,row in zip(content['indices'],content['rows']):
        if index not in wanted:continue
        assert list(row['member'])==members[index]
        inputs=tuple(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in row['case'].solver_inputs)
        n,areas,_,_,_,flags,spec=inputs
        for group in torch.unique(flags[:,2]):
            if group<=0:continue
            use=flags[:,2]==group;assert bool((areas[use]==areas[use][0]).all())
            locked=use & flags[:,:2].bool().any(1)
            if locked.any():
                wh=spec[locked,2:];assert torch.allclose(wh,wh[:1].expand_as(wh),rtol=0,atol=1e-10)
                assert bool((abs(wh.prod(-1)/areas[locked]-1)<.009999).all())
            groups+=int(use.sum()>1)
        old=row['graph'];reference=old['reference']*old['scale']
        graph=sample(inputs,reference);graph.update(public_ports(inputs,graph['scale']))
        # Preserve cached center/log-aspect labels without introducing WH roundtrips.
        graph['target']=old['target'].clone();graph['target'][:,:2]*=old['scale']/graph['scale']
        assert graph['features'].shape==(n,17) and bool((graph['mask']==1).all())
        assert torch.equal(graph['constraints'][:,2],flags[:,2].long())
        role_fields=build(graph)
        output.append(dict(member=tuple(members[index]),graph=graph,role=role_fields,inputs=inputs,
            reference_xywh=reference,baseline_metrics=row['case'].baseline_metrics))
        ids.append(index)
    destination=CACHE/role/f'{min(ids):06d}.pt'
    checkpoint(destination,dict(identity=key,indices=ids,rows=output))
    return dict(path=destination.relative_to(HERE).as_posix(),sha256=sha(destination),indices=ids,
                source=part['path'],count=len(ids),nontrivial_mib_groups=groups)
def main():
    start=time.perf_counter();state=dict(status='preparing',pid=os.getpid(),completed=0,total=32500,training_started=False,test_read=False)
    with exclusive(HERE/'data_preparation.lock'):
        ident=identity();key=digest(ident);registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
        checked=dict(registry);assert digest({k:v for k,v in checked.items() if k!='manifest_sha256'})==checked['manifest_sha256']
        # Register the physical-path migration, preserving every historical manifest byte.
        migrations={}
        for name,digest_old in registry['frozen_existing_protocol'].items():
            actual=sha(ROOT/name)
            if name=='protocol/paths.json':migrations[name]=dict(historical_sha256=digest_old,current_sha256=actual,reason='Previously recorded dataset relocation and random40k path addition; old inputs reused from verified cache')
            else:assert actual==digest_old,name
        tasks=[];wanted_by_role={};source_manifests={}
        for role in ('train','dev'):
            records=registry['sets'][role]['records'];members=[r['member'] for r in records]
            wanted=set(range(32000)) if role=='train' else {members.index(r['member']) for r in registry['sets']['dev_monitor500']['records']}
            m=json.loads((SOURCE/role/'manifest.json').read_text(encoding='utf-8'));assert m['members']==members
            wanted_by_role[role]=wanted;source_manifests[role]=m
            for part in m['parts']:
                selected=wanted.intersection(part['indices'])
                if selected:tasks.append((role,part,m['part_identities'][part['identity_key']],selected,members,key))
        train_members={tuple(r['member']) for r in registry['sets']['train']['records'][:32000]}
        dev_members={tuple(r['member']) for r in registry['sets']['dev_monitor500']['records']}
        assert not train_members & dev_members and not {s for s,i in train_members}&{s for s,i in dev_members}
        save(CACHE/'identity.json',ident);save(HERE/'data_preparation_state.json',state)
        results={r:[] for r in ('train','dev')}
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(process,t):t[0] for t in tasks}
            for future in as_completed(pending):
                result=future.result();role=pending[future];results[role].append(result)
                state.update(completed=state['completed']+result['count'],elapsed_seconds=time.perf_counter()-start,part=result['path'])
                save(HERE/'data_preparation_state.json',state)
                print(f'[DATA] {state["completed"]}/32500 {result["path"]}',flush=True)
                if (HERE/'STOP').exists():raise RuntimeError('User STOP; no further processing authorized')
        for role,parts in results.items():
            covered=[i for p in parts for i in p['indices']];assert len(covered)==len(set(covered)) and set(covered)==wanted_by_role[role]
            records=registry['sets'][role]['records']
            save(CACHE/role/'manifest.json',dict(identity=key,members=[r['member'] for r in records],selected_indices=sorted(covered),parts=sorted(parts,key=lambda p:p['indices'][0])))
        assert identity()==ident,'Data producers changed while processing'
        save(CACHE/'panels.json',{role:[r['member'] for r in registry['sets'][role]['records']] for role in ('train_monitor500','dev_monitor500')})
        state.update(status='complete',identity=key,train_count=32000,dev_count=500,source_cache_reused=True,
            masks_all_ones=True,mib_preserved=True,immutable_specs_unchanged=True,path_migrations=migrations,elapsed_seconds=time.perf_counter()-start)
        save(HERE/'data_preparation_state.json',state);print('[DATA] COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(HERE/'data_preparation_error.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
