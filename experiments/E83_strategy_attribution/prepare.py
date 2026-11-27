"""Strict frozen weight composition; never fabricate training best/latest."""
import json
import torch
from .io import ROOT,HERE,sha,save,sources,snapshot,checkpoint
from .cache import identity as cache_identity
from .model import GeometryModel

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def parents():
    meta=read(HERE/'provenance.json')['parents']
    for n,v in meta.items():
        if sha(ROOT/v['checkpoint'])!=v['sha256']:raise ValueError('Frozen parent changed: '+n)
    return meta

def load_parent(name):
    v=parents()[name];return torch.load(ROOT/v['checkpoint'],map_location='cpu',weights_only=False,mmap=True)

def build(arm):
    src=load_parent('F81');p=load_parent(arm['placer'])
    config={**src['model_config'],'width':p['model_config']['width'],'learned_residuals':arm['learned_residuals']}
    model=GeometryModel(**config)
    weights={k:v for k,v in src['model'].items() if k.startswith('refiner.')}
    weights.update({'placer.'+k:v for k,v in p['model'].items()})
    model.load_state_dict(weights,strict=True)
    assert all(torch.equal(v,p['model'][k]) for k,v in model.placer.state_dict().items())
    assert all(torch.equal(v,src['model']['refiner.'+k]) for k,v in model.refiner.state_dict().items())
    return model,config

def freeze(arm):
    folder=HERE/'runs'/arm['name'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'composition.pt'
    if not path.exists():
        model,config=build(arm)
        checkpoint(path,{'model':model.state_dict(),'model_config':config,'arm':arm,
            'parents':parents(),'sources':sources(),'training_updates':0,'selection':'Fixed predeclared module intervention; no training best','test_read':False})
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if state['sources']!=sources() or state['parents']!=parents() or state['arm']!=arm:raise ValueError('Composition identity drift')
    identity={'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),
      'sources':sources(),'parents':parents(),'arm':arm,'model_config':state['model_config'],
      'cache_identity':cache_identity(),'training_updates':0,'count':500,'workers':2,'threads_per_worker':2,
      'test_read':False,'selection':'One fixed frozen intervention; no parameter selection'}
    if (folder/'frozen.json').exists() and read(folder/'frozen.json')!=identity:raise ValueError('Cannot replace frozen evaluation')
    save(folder/'frozen.json',identity)
    snapshot(folder/'source_snapshot.zip',identity)
    return folder,state,identity
