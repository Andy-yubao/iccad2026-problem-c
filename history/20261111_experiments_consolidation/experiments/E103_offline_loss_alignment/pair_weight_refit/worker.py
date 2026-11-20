"""Only current best29 is newly evaluated; enhanced outputs stay cached."""
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
from .common import ROOT,OWNER,RUN,BANK,KEYS,read,save,sha,setup,check

MODEL=None
IDENTITY=None
PLAN=None

def initialize(identity):
    global MODEL,IDENTITY,PLAN
    setup();IDENTITY=identity;PLAN=read(OWNER/'plan.json')
    path=ROOT/PLAN['current_checkpoint'];assert sha(path)==PLAN['current_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    MODEL=GeometryModel(**state['model_config']).cpu().eval()
    MODEL.load_state_dict(state['model'],strict=True);MODEL.requires_grad_(False)
    with threadpool_limits(limits=1):warmup()
    save(RUN/f'workers/{os.getpid()}.json',dict(pid=os.getpid(),status='ready',models=1,threads=1))

def evaluate_case(index):
    check();path=RUN/f'current/{index:04d}.json'
    if path.exists():
        old=read(path);assert old['identity']==IDENTITY and old['checkpoint_sha256']==PLAN['current_sha256']
        return index
    row=torch.load(BANK/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
    old=read(BANK/f'bank/{index:04d}.json')
    parent=next(d for d in old['drafts'] if d['model_id']=='e98_generalization')
    with torch.inference_mode(),threadpool_limits(limits=1):
        g=loss_graph(row);network=batches.inputs_only(batches.collate([row['graph']],[0],'cpu'))
        assert not {'target','reference'}.intersection(network)
        started=time.perf_counter();state=MODEL(network)
        draft=(batches.decode(state,network)[0]*network['scale'][0]).numpy().astype(float)
        inference=time.perf_counter()-started
        values=terms(state,g);assert np.isfinite(list(values.values())).all()
        case=row['case'];inp=case.solver_inputs
        case.reference_positions=public_hard_reference(inp[1],inp[5],inp[6])
        scored=score(draft,case)
    # Same member and reference denominators are required before pair formation.
    assert old['member']==list(row['member'])
    assert scored['metrics']['hpwl_baseline']==parent['metrics']['hpwl_baseline']
    assert scored['metrics']['bbox_area_baseline']==parent['metrics']['bbox_area_baseline']
    record=dict(index=index,member=old['member'],n=old['n'],model_id='learned_v1_best29',identity=IDENTITY,
        checkpoint_sha256=PLAN['current_sha256'],state=state[0].numpy().tolist(),draft_positions=draft.tolist(),
        loss_terms=values,inference_seconds=inference,**scored,test_read=False)
    save(path,record)
    save(RUN/f'workers/{os.getpid()}.json',dict(pid=os.getpid(),status='evaluating_current',last_case=index))
    return index

def verify_parent_bridge(indices):
    plan=read(OWNER/'plan.json');saved=torch.load(ROOT/plan['enhanced_checkpoint'],map_location='cpu',weights_only=False)
    model=GeometryModel(**saved['model_config']).cpu().eval();model.load_state_dict(saved['model'],strict=True)
    results=[]
    with torch.inference_mode(),threadpool_limits(limits=1):
        warmup()
        for index in indices:
            row=torch.load(BANK/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
            record=read(BANK/f'bank/{index:04d}.json')
            cached=next(d for d in record['drafts'] if d['model_id']=='e98_generalization')
            g=loss_graph(row);network=batches.inputs_only(batches.collate([row['graph']],[0],'cpu'))
            state=model(network);draft=(batches.decode(state,network)[0]*network['scale'][0]).numpy().astype(float)
            np.testing.assert_array_equal(draft,cached['draft_positions'])
            values=terms(state,g)
            np.testing.assert_allclose([values[k] for k in KEYS],[cached['loss_terms'][k] for k in KEYS],rtol=0,atol=0)
            case=row['case'];inp=case.solver_inputs;case.reference_positions=public_hard_reference(inp[1],inp[5],inp[6])
            scored=score(draft,case)
            np.testing.assert_array_equal(scored['positions'],cached['positions'])
            assert scored['Q']==cached['Q']
            results.append(dict(index=index,member=record['member'],same_draft=True,same_ten_terms=True,same_final=True,same_Q=True))
    return results
