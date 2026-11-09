"""Five train problems bridge cached enhanced results and generate current rows once."""
import ast
import time
from .common import ROOT,OWNER,RUN,BANK,read,save,sha,setup,sources
from .worker import initialize,evaluate_case,verify_parent_bridge

def main():
    started=time.perf_counter();setup()
    for p in OWNER.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    plan=read(OWNER/'plan.json');panel=read(OWNER.parent/'panel.json')
    assert sha(ROOT/plan['enhanced_checkpoint'])==plan['enhanced_sha256']
    assert sha(ROOT/plan['current_checkpoint'])==plan['current_sha256']
    assert plan['regularization']==1. and plan['weight_fits']==1 and plan['models']==2
    current=sources();identity=dict(sources=current,plan_sha256=sha(OWNER/'plan.json'),
        panel_sha256=sha(OWNER.parent/'panel.json'),enhanced_sha256=plan['enhanced_sha256'],current_sha256=plan['current_sha256'])
    ids=panel['implementation_indices'];assert len(ids)==5
    bridge=verify_parent_bridge(ids)
    initialize(identity)
    for i in ids:evaluate_case(i)
    save(OWNER/'preflight.json',dict(status='passed',sources=current,identity=identity,unique_train_cases=5,
        enhanced_bridge=bridge,new_current_cases=5,current_outputs_retained=True,
        test_read=False,dev_payloads=0,elapsed_seconds=time.perf_counter()-started))
    print('Passed5 train cases: cached enhanced drafts/ten terms/finals/Q reproduced exactly; current outputs retained.',flush=True)

if __name__=='__main__':main()
