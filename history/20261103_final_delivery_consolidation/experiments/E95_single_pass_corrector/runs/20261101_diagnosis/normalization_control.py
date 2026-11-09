"""Frozen intervention plus matched short learning check; same five train cases."""
import sys,time,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.E95_single_pass_corrector.io import HERE,save
from experiments.E95_single_pass_corrector.runtime import seed,CONFIG,charge
from experiments.E95_single_pass_corrector.data import load_rows,collate,inputs_only
from experiments.E95_single_pass_corrector.model import GeometryModel
from experiments.E95_single_pass_corrector.initialization import load
from experiments.E95_single_pass_corrector.loss import per_case

def main():
    seed();ids=[1683,10027,5474,10140,15001];rows=load_rows('train',ids);batch=[rows[i%5] for i in range(32)]
    g=inputs_only(collate(batch,'cuda'));model=GeometryModel(**CONFIG).cuda();load(model)
    initial=copy.deepcopy(model.state_dict());model.eval();draft=model.draft(g)
    last=torch.load(HERE/'runs/20261101_single_call_seed0/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    frozen={}
    for name,state in [('initial',initial),('latest',last['model'])]:
        model.load_state_dict(state);model.eval()
        for normalized in (False,True):
            for layer in model.corrector.layers:layer.normalize_local=normalized
            with torch.no_grad():
                y=model.corrector(g,draft);values=per_case(y,g,draft)
                frozen[f'{name}_{normalized}']=dict(total=float(values['total'].mean()),hidden_rms=float(model.corrector.last_hidden_rms.mean()))
    learning={}
    for normalized in (False,True):
        seed();model.load_state_dict(initial)
        for layer in model.corrector.layers:layer.normalize_local=normalized
        opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
        curve=[]
        for step in range(17):
            model.eval()
            with torch.no_grad():
                v=per_case(model.corrector(g,draft),g,draft)
                curve.append(dict(step=step,total=float(v['total'].mean()),hidden_rms=float(model.corrector.last_hidden_rms.mean())))
            if step==16:break
            model.train();opt.zero_grad(set_to_none=True);v=per_case(model.corrector(g,draft),g,draft);v['total'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.corrector.parameters(),1.,error_if_nonfinite=True);opt.step()
        learning[str(normalized)]=curve
    result=dict(unique_train_cases=5,batch_size=32,steps_per_arm=16,matched_initialization_optimizer_order_seed=True,
        frozen=frozen,learning=learning,test_read=False,scope='implementation/mechanism diagnostic only; no dev or Q tuning')
    save(Path(__file__).parent/'normalization_control.json',result)
    print(json.dumps(dict(frozen=frozen,first_last={k:[v[0],v[-1]] for k,v in learning.items()})),flush=True)

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:charge(time.perf_counter()-started)
