"""Public v5 inputs and frozen-model drafts; no labels enter optimization."""
import json
import hashlib
import random
import time
import numpy as np
import torch
from datasets.reference_layouts import REGISTRY, registry
from .io import ROOT, HERE, sha, checkpoint,save
from .model import GeometryModel, decode
from .features import graph_features

WEIGHTS=ROOT/'experiments/E10_integrated_baseline/runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'
WEIGHT_HASH='47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28'
FREEZE=HERE/'runs/frozen_model.json'
PUBLIC_CACHE=HERE/'runs/public_dev500_v1.pt'


def seed_all(seed=0,threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)


def graph(inputs):
    f=graph_features(inputs); scale=f['scale']; locked=f['locked']; n=int(inputs[0])
    known=torch.cat(((f['specs'][:,:2]+f['specs'][:,2:]/2)/scale,
        (f['specs'][:,2].clamp_min(1e-8)/f['specs'][:,3].clamp_min(1e-8)).log()[:,None]),-1)*(1-f['mask'])
    values={'features':f['features'],'adjacency':f['adjacency'],'mib':f['relations'][0],
        'group':f['relations'][1],'mask':f['mask'],'known':known,
        'area':torch.where(locked,f['specs'][:,2:].prod(1),f['areas'])/scale.square(),
        'specs':torch.where(locked[:,None],f['specs']/scale,torch.zeros_like(f['specs'])),
        'constraints':torch.as_tensor(inputs[5]).long(),'scale':scale,'valid':torch.ones(n,dtype=torch.bool)}
    return {key:value.unsqueeze(0) for key,value in values.items()}


@torch.no_grad()
def predict(model,inputs):
    started=time.perf_counter(); g=graph(inputs)
    positions=(decode(model(g),g)[0]*g['scale'][0]).numpy().astype(float)
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]); locked=c[:,:2].any(1)
    positions[locked,2:]=s[locked,2:];positions[c[:,1]!=0]=s[c[:,1]!=0]
    return positions,time.perf_counter()-started


def panel_metadata(panel='monitor'):
    reg=registry()
    if panel=='monitor':return reg['sets']['dev_monitor500']['records']
    if panel!='confirmation':raise ValueError(panel)
    used={tuple(r['member']) for r in reg['sets']['dev_monitor500']['records']}
    grouped={}
    for record in reg['sets']['dev']['records']:
        if tuple(record['member']) not in used:grouped.setdefault(record['n'],[]).append(record)
    def key(record):return hashlib.sha256(('E20.confirmation500.v1:'+json.dumps(record['member'])).encode()).hexdigest()
    for values in grouped.values():values.sort(key=key)
    chosen=[]
    for depth in range(max(map(len,grouped.values()))):
        for n in sorted(grouped):
            if depth<len(grouped[n]):chosen.append(grouped[n][depth])
            if len(chosen)==500:return chosen
    raise ValueError('Insufficient non-monitor development cases')


def public_cases(panel='monitor'):
    reg=registry(); folder=ROOT/'experiments/E17_e10_geometry_v5/runs/prepared_v5_e10_cache_v1/dev'
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['identity']['registry']!=sha(REGISTRY): raise ValueError('Data registry changed')
    metadata=panel_metadata(panel); needed={tuple(r['member']) for r in metadata}; found={}
    for part in manifest['parts']:
        path=folder/part['name']; st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']): raise ValueError('Source artifact changed')
        value=torch.load(path,map_location='cpu',weights_only=False)
        for row in value['rows']:
            key=tuple(row['member'])
            if key in needed: found[key]=row['case'] # Do not index graph/target/reference supervision.
    return [found[tuple(r['member'])] for r in metadata],metadata,sha(folder/'manifest.json')


def prepare_cache(panel='monitor'):
    path=PUBLIC_CACHE if panel=='monitor' else HERE/'runs/public_confirmation500_v1.pt'
    digest=sha(WEIGHTS)
    if digest!=WEIGHT_HASH:raise ValueError('Retained original E10 checkpoint changed')
    frozen={'checkpoint':WEIGHTS.relative_to(ROOT).as_posix(),'checkpoint_sha256':digest,'registry_sha256':sha(REGISTRY)}
    if FREEZE.exists():
        if json.loads(FREEZE.read_text())!=frozen:raise ValueError('Cannot change frozen model')
    else:save(FREEZE,frozen)
    if path.exists():
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['checkpoint_sha256']!=digest or value['registry_sha256']!=sha(REGISTRY): raise ValueError('Draft cache identity mismatch')
        if value['feature_and_network_sources']!={name:sha(HERE/name) for name in ['cases.py','features.py','model.py']}:
            raise ValueError('Inference source changed; prepare a new draft-cache version')
        return value
    seed_all(); state=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    cases,metadata,source=public_cases(panel); rows=[]
    for index,(case,meta) in enumerate(zip(cases,metadata)):
        draft,seconds=predict(model,case.solver_inputs)
        rows.append({'case':case,'member':meta['member'],'draft':draft,'inference_seconds':seconds})
    value={'checkpoint_sha256':digest,'registry_sha256':sha(REGISTRY),'source_manifest_sha256':source,
        'feature_and_network_sources':{name:sha(HERE/name) for name in ['cases.py','features.py','model.py']},
        'threads':2,'rows':rows,'test_read':False,'panel':panel,'timing':'Includes public feature construction, CPU inference, decode; excludes file and model load'}
    checkpoint(path,value);return value
