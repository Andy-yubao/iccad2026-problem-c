"""Public v5 inputs and frozen-model drafts; no labels enter optimization."""
import json
import random
import time
import numpy as np
import torch
from datasets.reference_layouts import REGISTRY, registry
from .io import ROOT, HERE, sha, checkpoint
from .model import GeometryModel, decode
from .features import graph_features

WEIGHTS=ROOT/'experiments/E17_e10_geometry_v5/runs/20261020_e10_geometry_v5_seed0/checkpoints/best.pt'
WEIGHTS_SHA='3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
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


def public_cases():
    reg=registry(); folder=ROOT/'experiments/E17_e10_geometry_v5/runs/prepared_v5_e10_cache_v1/dev'
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['identity']['registry']!=sha(REGISTRY): raise ValueError('Data registry changed')
    metadata=reg['sets']['dev_monitor500']['records']; needed={tuple(r['member']) for r in metadata}; found={}
    for part in manifest['parts']:
        path=folder/part['name']; st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']): raise ValueError('Source artifact changed')
        value=torch.load(path,map_location='cpu',weights_only=False)
        for row in value['rows']:
            key=tuple(row['member'])
            if key in needed: found[key]=row['case'] # Do not index graph/target/reference supervision.
    return [found[tuple(r['member'])] for r in metadata],metadata,sha(folder/'manifest.json')


def prepare_cache():
    if sha(WEIGHTS)!=WEIGHTS_SHA: raise ValueError('Frozen E17 checkpoint changed')
    if PUBLIC_CACHE.exists():
        value=torch.load(PUBLIC_CACHE,map_location='cpu',weights_only=False)
        if value['checkpoint_sha256']!=WEIGHTS_SHA or value['registry_sha256']!=sha(REGISTRY): raise ValueError('Draft cache identity mismatch')
        return value
    seed_all(); state=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    cases,metadata,source=public_cases(); rows=[]
    for index,(case,meta) in enumerate(zip(cases,metadata)):
        draft,seconds=predict(model,case.solver_inputs)
        rows.append({'case':case,'member':meta['member'],'draft':draft,'inference_seconds':seconds})
    value={'checkpoint_sha256':WEIGHTS_SHA,'registry_sha256':sha(REGISTRY),'source_manifest_sha256':source,
        'feature_and_network_sources':{name:sha(HERE/name) for name in ['cases.py','features.py','model.py']},
        'threads':2,'rows':rows,'test_read':False,'timing':'Includes public feature construction, CPU inference, decode; excludes file and model load'}
    checkpoint(PUBLIC_CACHE,value);return value
