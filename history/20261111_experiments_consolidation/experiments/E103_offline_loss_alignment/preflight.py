"""Five formal-panel train cases only; outputs are retained in the core bank."""
import ast
import platform
import time
import numpy as np
import torch
import scipy
import clarabel
from threadpoolctl import threadpool_limits
from .common import OWNER,RUN,read,save,sha,sources,setup,KEYS
from .run import materialize
from .worker import initialize,bank_case,loss_graph
from .simple_loss import components
from .loss_config import A,B,COEFFICIENTS

def main():
    started=time.perf_counter();setup()
    for p in OWNER.rglob('*.py'):
        if not {'runs','assets','evidence','__pycache__'}.intersection(p.relative_to(OWNER).parts):
            ast.parse(p.read_text(encoding='utf-8-sig'))
    panel=read(OWNER/'panel.json')
    assert len(set(panel['train_indices']))==1000 and len(set(map(tuple,panel['members'])))==1000
    splits=panel['analysis_splits']
    assert [len(splits[k]) for k in ('fit','tune','check')]==[600,200,200]
    assert sorted(splits['fit']+splits['tune']+splits['check'])==list(range(1000))
    assert set(panel['direction_indices'])<=set(splits['tune']) and len(panel['direction_indices'])==40
    current=sources()
    identity=dict(source_hashes=current,panel_sha256=sha(OWNER/'panel.json'),
        plan_sha256=sha(OWNER/'plan.json'),model_manifest_sha256=sha(OWNER/'model_manifest.json'),
        numerical_environment=dict(torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,
            clarabel=clarabel.__version__,python=platform.python_version(),processor=platform.processor(),
            inference='CPU FP32 deterministic, one process thread, original frozen input/decoder semantics',
            repair='E98 fixed-depth v1, CPU1 per worker, neutral-time Q'),test_read=False)
    ids=panel['implementation_indices'];assert len(ids)==5
    materialize(ids)
    with threadpool_limits(limits=1):
        initialize(identity)
        for i in ids:bank_case(i)
        row=torch.load(RUN/f'inputs/{ids[0]:04d}.pt',map_location='cpu',weights_only=False)
        result=read(RUN/f'bank/{ids[0]:04d}.json')
        state=torch.tensor([result['drafts'][-1]['state']],dtype=torch.float32,requires_grad=True)
        g=loss_graph(row);v=components(state,g,dict(overlap=A,task=B))
        loss=sum(COEFFICIENTS[k]*v[k].mean() for k in KEYS)
        gradient=torch.autograd.grad(loss,state)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    save(OWNER/'preflight.json',dict(status='passed',sources=current,identity=identity,
        unique_train_cases=5,model_sources=5,core_drafts_retained=25,all_decoders_match_common_geometry=True,
        finite_loss_and_state_gradient=True,dev_payloads=0,test_read=False,
        note='Only these five cases were implementation checked; retained in formal1000x5 bank, not regenerated.',
        elapsed_seconds=time.perf_counter()-started))
    print('Passed: five train cases x five frozen models, E98 depth repair, finite common Loss/gradient, outputs retained.',flush=True)

if __name__=='__main__':main()
