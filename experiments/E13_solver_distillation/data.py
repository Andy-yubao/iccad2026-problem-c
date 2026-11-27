"""Frozen bank membership; public features and supervision remain separate."""
import hashlib,json
import numpy as np
import torch
from .io import ROOT,sha
from .backend.frozen.inference import prepare_inference
from .targets import SCHEMA,validate
from .student import CHECKPOINT_SHA256
from .backend.frozen.packing import prepare_problem
from datasets.worksets import LayoutReader,workset_manifest,member_pairs
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_bank(path,split,smoke=False):
    path=path.resolve();config=json.loads((path/'config.json').read_text(encoding='utf-8'))['identity']
    state=json.loads((path/'results.json').read_text(encoding='utf-8'))
    for name,key in [('protocol/split_v1.json','split_sha256'),('protocol/workset_v2.json','workset_sha256')]:
        if sha(ROOT/name)!=config[key]:raise ValueError('Frozen protocol mismatch')
    if config['schema']!=SCHEMA or state['error_count'] or state['status']!='completed':raise ValueError('Incomplete/failed teacher bank')
    if config['checkpoint_sha256']!=CHECKPOINT_SHA256:raise ValueError('Unexpected teacher checkpoint')
    if config['pilot']!=smoke:raise ValueError('Pilot cannot be used as a full training bank')
    if config['split']!=split:raise ValueError('Teacher bank split mismatch')
    records=state['records'];reader=LayoutReader(workset_manifest(),'train_monitor' if smoke else split)
    if not smoke and config['members']!=[list(p) for p in reader.members]:raise ValueError('Teacher members differ from frozen workset')
    if len(records)!=len(config['members']):raise ValueError('Missing teacher record')
    if len({tuple(r['member']) for r in records})!=len(records):raise ValueError('Duplicate teacher member')
    if {tuple(r['member']) for r in records}!={tuple(r) for r in config['members']}:raise ValueError('Teacher record/member mismatch')
    identity=digest({'identity':config,'labels':sorted((r['case_id'],r['label_sha256']) for r in records)})
    if smoke:records=[r for r in records if int(r['member'][0].split('/')[0].split('_')[1]) in (0,24,49,74,99)]
    lookup={pair:i for i,pair in enumerate(reader.members)};samples=[];quality=[]
    for k,r in enumerate(records):
        member=tuple(r['member']);raw=reader.case(lookup[member]);assert raw.case_id==r['case_id']
        label_path=path/r['label_file']
        if not label_path.resolve().is_relative_to(path) or sha(label_path)!=r['label_sha256']:raise ValueError('Teacher label hash/path mismatch')
        p=prepare_problem(*raw.solver_inputs)
        with np.load(label_path,allow_pickle=False) as z:
            validate(z,p);target=torch.tensor(z['geometry']);positions=z['positions']
            ov=np.minimum(positions[:,None,:2]+positions[:,None,2:],positions[None,:,:2]+positions[None,:,2:])-np.maximum(positions[:,None,:2],positions[None,:,:2])
            axis=np.argmin(np.abs(ov),axis=-1);centers=positions[:,:2]+positions[:,2:]/2
            delta=centers[None,:,:]-centers[:,None,:]
            forward=np.take_along_axis(delta,axis[...,None],axis=-1)[...,0]>=0
            contact=np.where(z['group_contact_positive'],axis+2*(~forward),4)
            supervision={'geometry':target,'contact':torch.tensor(contact,dtype=torch.long),
               'boundary':torch.tensor((z['boundary_satisfied'][:,None]&np.array([1,2,4,8]))!=0,dtype=torch.float32),
               'mib':torch.tensor(z['mib_same'],dtype=torch.float32)}
        public={key:value[0] for key,value in prepare_inference(raw.solver_inputs,'cpu').items()}
        samples.append({'public':public,'targets':supervision,'member':member,'case_id':raw.case_id})
        case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
        ref=evaluate_case(case,raw.reference_positions,0.)['metrics'];teacher=r['evaluation']['metrics']
        quality.append({'case_id':raw.case_id,'reference_Q':ref['cost'],'teacher_Q':teacher['cost'],
          'delta':teacher['cost']-ref['cost'],'reference_legal':ref['is_feasible'],'teacher_legal':teacher['is_feasible'],
          'rescue':r['teacher_stats'].get('rescue_used',False)})
        if (k+1)%1000==0:print('Prepared',split,k+1,len(records),flush=True)
    return samples,identity,quality

def collate(samples,indices,device):
    rows=[samples[int(i)] for i in indices];n=max(len(r['public']['features']) for r in rows);b=len(rows)
    public={};targets={}
    for destination,keyset in [(public,'public'),(targets,'targets')]:
        for key,first in rows[0][keyset].items():
            if first.ndim==0:destination[key]=torch.stack([r[keyset][key] for r in rows]);continue
            pair=key in (('adjacency','mib','group') if keyset=='public' else ('contact','mib'))
            shape=(b,n,n)+first.shape[2:] if pair else (b,n)+first.shape[1:]
            value=torch.zeros(shape,dtype=first.dtype)
            if keyset=='targets' and key=='contact':value.fill_(4)
            for j,r in enumerate(rows):
                count=len(r['public']['features'])
                if pair:value[j,:count,:count]=r[keyset][key]
                else:value[j,:count]=r[keyset][key]
            destination[key]=value
    return ({k:v.to(device) for k,v in public.items()},{k:v.to(device) for k,v in targets.items()})

def monitor_indices(samples):
    monitor=set(member_pairs(workset_manifest(),'train_monitor'))
    result=[i for i,s in enumerate(samples) if s['member'] in monitor]
    if len(result)!=500:raise ValueError('Missing fixed train_monitor members')
    return result
