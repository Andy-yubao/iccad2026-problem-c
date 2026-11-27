"""CPU-only preparation checks: no optimizer steps, CUDA execution, or solver."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
from pathlib import Path
import sys,json,ast,time,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from experiments.E41_rectangular_depth.model import GeometryModel
from experiments.E41_rectangular_depth.arms import model_config
from experiments.E41_rectangular_depth.data import load_rows,collate
from experiments.E41_rectangular_depth.batches import inputs_only
from experiments.E41_rectangular_depth.losses import WEIGHTS
from experiments.E41_rectangular_depth.pipeline import commands,control_identity,OUTPUT,COORDINATOR
from experiments.E41_rectangular_depth.io import sources,save
from experiments.E39_rectangular_attention.model import GeometryModel as FrozenModel

def main():
    start=time.perf_counter();experiment=ROOT/'experiments/E41_rectangular_depth'
    assert not OUTPUT.exists() and not COORDINATOR.exists(),'Preparation must not start formal output'
    for p in experiment.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    config=model_config('rectangle');assert config=={'width':96,'layers':16,'rectangle':True}
    candidate=GeometryModel(**config).eval();count=sum(p.numel() for p in candidate.parameters());assert count==1958083,count
    assert WEIGHTS=={'center_local':1.,'shape':.25,'relative_local':.25,'overlap':.5,'contact':.5,'boundary':.25,'mib':.1,'bbox':.1}
    rows=load_rows('dev_monitor500',indices=[0]);g,_=collate(rows,'cpu');g=inputs_only(g)
    checkpoint=ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0/checkpoints/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    old=FrozenModel(**state['model_config']).eval();old.load_state_dict(state['model'],strict=True)
    copied=GeometryModel(**state['model_config']).eval();copied.load_state_dict(state['model'],strict=True)
    with torch.no_grad():
        a=old(g);b=copied(g);new=candidate(g)
    assert torch.equal(a,b),'Copied architecture differs from frozen E39'
    assert torch.isfinite(new).all() and new.shape==a.shape
    control=control_identity()
    cmd=commands(False)
    assert '--resume' not in cmd['train'] and '--baseline' not in cmd['final']
    assert all('E25' not in token for argv in cmd.values() for token in argv)
    src=sources()
    report={'status':'prepared_not_started','model_config':config,'parameters':count,'MIB_loss_weight':WEIGHTS['mib'],
        'old_depth12_strict_load':True,'old_depth12_output_exact_parity':True,'new_depth16_finite_forward':True,
        'forward_shape':list(new.shape),'checked_dev_members':[list(rows[0]['member'])],
        'source_identity':src,'control_identity':control,'commands':cmd,'seconds':time.perf_counter()-start,
        'training_started':False,'optimizer_steps':0,'solver_calls':0,'test_read':False,'device':'cpu',
        'cuda_training_entry_checked':False,'CUDA_note':'User requested commands only; no CUDA or training steps were executed',
        'formal_run_directories_created':False}
    save(HERE/'results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_identity','control_identity','commands']},ensure_ascii=True,indent=2))
if __name__=='__main__':main()
