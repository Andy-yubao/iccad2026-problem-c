"""Exercise training save/resume on repeats of the same five diagnostic rows."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import json,time,torch
from experiments.E52_training_throughput.io import HERE,save
from experiments.E52_training_throughput.data import load_rows
from experiments.E52_training_throughput.cache import verify
from experiments.E52_training_throughput.ops import product2
from experiments.E52_training_throughput.train import parser,run,seed_all

def main():
    started=time.perf_counter();seed_all(0)
    verify('train');verify('dev')
    for data in ([[.13,.71],[.3,.9]], [[.13,.71],[0.,.9]], [[0.,0.],[.13,0.]]):
        a=torch.tensor(data,device='cuda',requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        a.prod(-1).sum().backward();product2(b).sum().backward()
        assert torch.equal(a.grad,b.grad)
    evidence=json.loads((HERE/'runs/20261025_profile5/results.json').read_text(encoding='utf-8'))
    rows=load_rows('train',evidence['indices'],use_cache=False)
    # Diagnostic roles only: no real dev labels or new cases are read here.
    train=[dict(rows[i%5]) for i in range(64)];dev=[dict(r) for r in rows]
    folder=HERE/'runs/20261025_runtime_smoke_only'
    args=parser().parse_args(['--output',str(folder),'--device','cuda','--engine','cuda_graph','--max-hours','.05'])
    run(args,implementation_check=True,stop_after_steps=1,check_rows=(train,dev))
    first=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert first['identity']['implementation_check'] and first['progress']['step']==1
    assert first['progress']['cursor']==32 and not first['finished']
    args.resume=True
    run(args,implementation_check=True,stop_after_steps=2,check_rows=(train,dev))
    second=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert second['progress']['step']==2 and second['progress']['epoch']==1
    assert not second['finished'] and second['elapsed_seconds']>=first['elapsed_seconds']
    value={'status':'passed','seconds':time.perf_counter()-started,'unique_real_cases':5,
           'source_role':'train only; repeated diagnostic fixtures are not a validation result',
           'first_steps':1,'resumed_steps':2,'first_cursor':32,'resumed_epoch':1,
           'cumulative_time_not_reset':True,'implementation_checkpoint_quarantined':True,
           'formal_training_started':False,'test_read':False,'product2_zero_and_nonzero_gradient_checks':True}
    save(HERE/'runs/20261025_profile5/runtime_smoke.json',value)
    print(json.dumps(value,indent=2),flush=True)

if __name__=='__main__':main()
