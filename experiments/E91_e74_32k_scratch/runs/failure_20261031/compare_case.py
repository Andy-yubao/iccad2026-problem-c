"""One already-rejected member across saved checkpoints, no training."""
import gc,time
from pathlib import Path
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save
from experiments.E91_e74_32k_scratch.data import load_rows
from experiments.E91_e74_32k_scratch.resident import ResidentRows
from experiments.E91_e74_32k_scratch.model import GeometryModel
from experiments.E91_e74_32k_scratch.loss import per_case
from experiments.E91_e74_32k_scratch.batches import inputs_only

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    folder=Path(__file__).resolve().parent;row=load_rows('train',[1032])[0]
    g=ResidentRows([row]*32,'cuda',max_nodes=120,max_pins=146).batch(list(range(32)))
    files={'best32_before_speedup':HERE/'runs/speedup_resume_20261031/before/best.pt',
           'best41':HERE/'runs/20261031_scratch32k_seed0/checkpoints/best.pt',
           'latest41372':HERE/'runs/20261031_scratch32k_seed0/checkpoints/latest.pt'}
    results={}
    for name,path in files.items():
        c=torch.load(path,map_location='cpu',weights_only=False)
        m=GeometryModel(**c['model_config']).cuda().eval();m.load_state_dict(c['model'])
        peaks=[];handles=[]
        for layer in m.layers:
            handles.append(layer.register_forward_hook(lambda mod,args,value:peaks.append(float(value.detach().abs().max()))))
        with torch.no_grad():state=m(inputs_only(g));v=per_case(state,g,'simple')
        results[name]=dict(step=c['progress']['step'],eval={k:float(x[0]) for k,x in v.items()},
            hidden_abs_max_per_layer=peaks,output_state_abs_max=float(state[0].abs().max()))
        for handle in handles:handle.remove()
        del m,c;gc.collect();torch.cuda.empty_cache()
    inp=row['case'].solver_inputs;n=inp[0]
    info=dict(index=1032,member=list(row['member']),n=n,
        port_connected_fraction=float((g['pin_weights'][0].sum(-1)[:n]>0).float().mean()),
        public_preplaced_count=int((g['constraints'][0,:n,1]!=0).sum()),
        public_b2b_edge_rows=len(inp[2]))
    save(folder/'case1032_comparison.json',dict(status='complete',case=info,checkpoints=results,
        one_member_repeated32_to_preserve_formal_kernel_shapes=True,
        uses_training_Dropout=False,neural_parameter_updates=0,
        formal_checkpoint_untouched=True,test_read=False,seconds=time.perf_counter()-started))
    print(info,flush=True)
    for name,r in results.items():print(name,r['step'],r['eval']['total'],r['eval']['task'],r['output_state_abs_max'],max(r['hidden_abs_max_per_layer']),flush=True)

if __name__=='__main__':main()
