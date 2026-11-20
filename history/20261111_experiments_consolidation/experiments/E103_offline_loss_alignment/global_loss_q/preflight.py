"""Five new problems only; fresh outputs retained in the formal dataset."""
import ast
import time
import torch
from ..cache import load_rows
from .common import ROOT,OWNER,RUN,read,save,sha,setup,sources
from .worker import initialize,evaluate

def main():
    setup();started=time.perf_counter();panel=read(OWNER/'panel.json');plan=read(OWNER/'plan.json')
    for path in OWNER.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    samples=panel['samples'];assert len(samples)==2000 and len({tuple(s['member']) for s in samples})==2000
    previous=read(OWNER.parent/'panel.json');assert not set(s['train_index'] for s in samples)&set(previous['train_indices'])
    assert sum(s['model_id']=='enhanced_best71' for s in samples)==1000
    assert sum(s['split']=='fit' for s in samples)==1400
    for model in ('enhanced_best71','current_best29'):
        assert sum(s['model_id']==model and s['split']=='check' for s in samples)==300
    for spec in plan['models']:assert sha(ROOT/spec['checkpoint'])==spec['sha256']
    identity=dict(sources=sources(),plan_sha256=sha(OWNER/'plan.json'),panel_sha256=sha(OWNER/'panel.json'),
        mathematical_mode='pooled individual observations across distinct problems, no pairwise ranking',test_read=False)
    indices=panel['implementation_indices'];assert len(indices)==5
    rows=load_rows('train',[samples[i]['train_index'] for i in indices])
    for i,row in zip(indices,rows):
        path=RUN/f'inputs/{i:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(row,path)
    initialize(identity)
    for i in indices:evaluate(i)
    save(OWNER/'preflight.json',dict(status='passed',sources=sources(),identity=identity,
        unique_new_train_cases=5,observations_retained=5,old_Loss_Q_records_used=0,
        overlap_with_old1000=0,dev_payloads=0,test_read=False,elapsed_seconds=time.perf_counter()-started))
    print('Passed5 fresh independent train problems; no old observations, both models covered, outputs retained.',flush=True)

if __name__=='__main__':main()
