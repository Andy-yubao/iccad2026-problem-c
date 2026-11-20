"""One model per new problem; all draft/Q observations are generated fresh."""
import os
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import public_hard_reference
from ..sources.e98_generalization.model import GeometryModel
from ..sources.e98_generalization import batches
from ..worker import loss_graph,terms,score
from ..postprocessor.repair.canonical_native import warmup
from .common import ROOT,OWNER,RUN,read,save,sha,setup,check

MODELS={}
IDENTITY=None

def initialize(identity):
    global MODELS,IDENTITY
    setup();IDENTITY=identity;MODELS={}
    for spec in read(OWNER/'plan.json')['models']:
        path=ROOT/spec['checkpoint'];assert sha(path)==spec['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=GeometryModel(**saved['model_config']).cpu().eval()
        model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False)
        MODELS[spec['id']]=model
    with threadpool_limits(limits=1):warmup()
    save(RUN/f'workers/{os.getpid()}.json',dict(status='ready',pid=os.getpid(),models=2,threads=1))

def evaluate(index):
    check();output=RUN/f'observations/{index:04d}.json'
    if output.exists():raise RuntimeError('Fresh sample already present; no unreviewed repeat')
    row=torch.load(RUN/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
    spec=read(OWNER/'panel.json')['samples'][index]
    assert list(row['member'])==spec['member']
    with torch.inference_mode(),threadpool_limits(limits=1):
        g=loss_graph(row);network=batches.inputs_only(batches.collate([row['graph']],[0],'cpu'))
        assert not {'target','reference'}.intersection(network)
        started=time.perf_counter();state=MODELS[spec['model_id']](network)
        draft=(batches.decode(state,network)[0]*network['scale'][0]).numpy().astype(float)
        inference=time.perf_counter()-started
        values=terms(state,g);assert np.isfinite(list(values.values())).all()
        case=row['case'];inp=case.solver_inputs
        case.reference_positions=public_hard_reference(inp[1],inp[5],inp[6])
        scored=score(draft,case)
    save(output,dict(index=index,member=spec['member'],model_id=spec['model_id'],split=spec['split'],
        n=case.block_count,identity=IDENTITY,loss_terms=values,state=state[0].numpy().tolist(),
        draft_positions=draft.tolist(),inference_seconds=inference,**scored,test_read=False))
    if index%10==0:save(RUN/f'workers/{os.getpid()}.json',dict(status='evaluating',pid=os.getpid(),last_index=index))
    return index
