"""Freeze a preregistered composition, without inventing a training best."""
import json
import torch
from .io import HERE,ROOT,sha,save,checkpoint,sources,snapshot
from .model import GeometryModel,extract

def frozen_checkpoint(output,baseline=False):
    if baseline:raise ValueError('Existing baselines are reused, never rerun here')
    pre=json.loads((HERE/'runs/preflight.json').read_text(encoding='utf-8'))
    if pre['status']!='passed' or pre['sources']!=sources():raise ValueError('Preflight source drift')
    for row in pre['parents'].values():
        if sha(ROOT/row['checkpoint'])!=row['sha256']:raise ValueError('Parent drift')
    output.mkdir(parents=True,exist_ok=True);path=output/'composition.pt';protocol=output/'frozen_protocol.json'
    if protocol.exists():
        identity=json.loads(protocol.read_text(encoding='utf-8'))
        if identity['sources']!=sources() or identity['checkpoint_sha256']!=sha(path):raise ValueError('Composition drift')
        return torch.load(path,map_location='cpu',weights_only=False),identity
    if path.exists():raise ValueError('Incomplete freeze, inspect composition before continuing')
    source=torch.load(ROOT/pre['parents']['correction']['checkpoint'],map_location='cpu',weights_only=False)
    placer=torch.load(ROOT/pre['parents']['placer']['checkpoint'],map_location='cpu',weights_only=False)
    model=extract(GeometryModel(**pre['model_config']),source['model'],placer['model'])
    identity={'role':'preregistered frozen composition, not trained own-Loss best',
        'sources':sources(),'parents':pre['parents'],'device':'cuda','seed':0,'arm':'topology',
        'parameters':1537022,'correction_parameters':65483,'training_updates':0,
        'preflight_sha256':sha(HERE/'runs/preflight.json'),'test_read':False}
    state={'model':model.state_dict(),'model_config':pre['model_config'],'identity':identity}
    checkpoint(path,state)
    identity={**identity,'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path)}
    save(protocol,identity);snapshot(output/'source_snapshot.zip',identity)
    return state,identity
