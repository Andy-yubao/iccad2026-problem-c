"""Four single-thread CPU processes; each problem produces all five drafts."""
import importlib
import os
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import public_hard_reference
from scoring.evaluation import evaluate_case
from .common import OWNER,RUN,KEYS,read,save,sha,setup,check
from .ports import attach
from .batches import collate,decode
from .simple_loss import components
from .loss_config import A,B
from .postprocessor.solver import optimize
from .postprocessor.repair.canonical_native import warmup

MODELS=[]
IDENTITY=None

def initialize(identity):
    global MODELS,IDENTITY
    setup();IDENTITY=identity;MODELS=[]
    with threadpool_limits(limits=1):
        warmup()
        for rec in read(OWNER/'model_manifest.json')['models']:
            path=OWNER.parents[1]/rec['checkpoint']
            assert sha(path)==rec['checkpoint_sha256']
            state=torch.load(path,map_location='cpu',weights_only=False)
            module=importlib.import_module(f'.sources.{rec["id"]}.model',__package__)
            batch_module=importlib.import_module(f'.sources.{rec["id"]}.batches',__package__)
            model=module.GeometryModel(**rec['model_config']).cpu().eval()
            model.load_state_dict(state['model'],strict=True);model.requires_grad_(False)
            MODELS.append((rec,model,batch_module))
        save(RUN/f'workers/{os.getpid()}.json',dict(status='ready',models=len(MODELS),threads=1,pid=os.getpid()))

def loss_graph(row):
    attach(row)
    g=collate([row['graph']],[0],'cpu')
    for key in ('pin_coordinates','pin_weights','pin_valid','wire_b2b'):
        g[key]=row['graph'][key][None]
    return g

def terms(state,g):
    values=components(state,g,dict(overlap=A,task=B))
    return {k:float(values[k][0]) for k in KEYS}

def score(draft,case):
    started=time.perf_counter()
    final,info=optimize(draft,case.solver_inputs)
    seconds=time.perf_counter()-started
    scored=evaluate_case(case,final,seconds)
    return dict(Q=scored['metrics']['cost'],metrics=scored['metrics'],
        independent_hard_checks=scored['independent_hard_checks'],positions=final.tolist(),
        repair_seconds=seconds,fallback=info['row_retained'],work=info['work'],trace=info['trace'])

def bank_case(index):
    check();started=time.perf_counter()
    row=torch.load(RUN/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
    output=RUN/f'bank/{index:04d}.json'
    if output.exists():
        old=read(output);assert old['identity']==IDENTITY and old['member']==list(row['member'])
        assert len(old['drafts'])==5
        return dict(index=index,cached=True)
    records=[]
    try:
        with torch.inference_mode(),threadpool_limits(limits=1):
            g=loss_graph(row)
            case=row['case'];inputs=case.solver_inputs
            case.reference_positions=public_hard_reference(inputs[1],inputs[5],inputs[6])
            for rec,model,batch_module in MODELS:
                check();start=time.perf_counter()
                network_graph=batch_module.inputs_only(batch_module.collate([row['graph']],[0],'cpu'))
                assert not {'target','reference'}.intersection(network_graph)
                state=model(network_graph)
                assert bool(torch.isfinite(state).all())
                # Each frozen source keeps its original decoder and fixed specs.
                draft=(batch_module.decode(state,network_graph)[0]*network_graph['scale'][0]).numpy().astype(float)
                own=(decode(state,g)[0]*g['scale'][0]).numpy().astype(float)
                np.testing.assert_array_equal(draft,own)
                inference=time.perf_counter()-start
                values=terms(state,g)
                assert all(np.isfinite(list(values.values())))
                scored=score(draft,case)
                records.append(dict(model_id=rec['id'],checkpoint_sha256=rec['checkpoint_sha256'],
                    state=state[0].numpy().tolist(),draft_positions=draft.tolist(),loss_terms=values,
                    inference_seconds=inference,**scored))
                save(RUN/f'workers/{os.getpid()}.json',dict(status='bank',case=index,models_done=len(records),pid=os.getpid()))
        save(output,dict(index=index,member=list(row['member']),n=int(row['case'].block_count),
            identity=IDENTITY,drafts=records,elapsed_seconds=time.perf_counter()-started,test_read=False))
        return dict(index=index,cached=False,seconds=time.perf_counter()-started)
    except BaseException as exc:
        save(RUN/f'errors/bank_{index:04d}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

def direction_case(index):
    check()
    output=RUN/f'direction/{index:04d}.json'
    if output.exists():return read(output)
    row=torch.load(RUN/f'inputs/{index:04d}.pt',map_location='cpu',weights_only=False)
    core=read(RUN/f'bank/{index:04d}.json')
    draft=next(r for r in core['drafts'] if r['model_id']=='e98_generalization')
    proposal=read(RUN/'analysis/proposal.json')
    from .loss_config import COEFFICIENTS
    coefficients=dict(baseline=COEFFICIENTS,learned=proposal['physical_coefficients'])
    with threadpool_limits(limits=1):
        g=loss_graph(row);case=row['case'];inp=case.solver_inputs
        case.reference_positions=public_hard_reference(inp[1],inp[5],inp[6])
        original=torch.tensor([draft['state']],dtype=torch.float32)
        unit=torch.cat((g['area'].sqrt()[...,None].expand(-1,-1,2),torch.ones_like(g['area'][...,None])),-1)
        active=g['mask']*g['valid'][...,None]
        results=[]
        for name,co in coefficients.items():
            check();u=(original/unit).detach().requires_grad_(True)
            values=components(u*unit,g,dict(overlap=A,task=B))
            value=sum(co[k]*values[k].mean() for k in KEYS)
            gradient=torch.autograd.grad(value,u)[0]*active
            assert torch.isfinite(gradient).all()
            rms=(gradient.square().sum()/active.sum().clamp_min(1)).sqrt().clamp_min(1e-12)
            for step in (.002,.01):
                check()
                changed=((u.detach()-step*(gradient/rms).clamp(-3,3))*unit).detach()
                moved=(decode(changed,g)[0]*g['scale'][0]).numpy().astype(float)
                scored=score(moved,case)
                results.append(dict(objective=name,step=step,gradient_rms=float(rms),
                    Q_delta=scored['Q']-draft['Q'],**scored))
        result=dict(index=index,n=core['n'],member=core['member'],baseline_Q=draft['Q'],
            identity=IDENTITY,proposal_sha256=sha(RUN/'analysis/proposal.json'),results=results,test_read=False)
        save(output,result)
        return result
