"""Five members from the failed batch; forward interventions only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from .io import HERE,ROOT,save
from .train import seed_all
from .data import load_rows,collate
from .model import GeometryModel
from .batches import inputs_only
from .loss import per_case

def main():
    seed_all(0);ids=[9065,9967,1028,15603,8658];rows=load_rows('train',ids,use_cache=False);g,_=collate(rows,'cuda')
    source=ROOT/'experiments/E72_topology_posttrain/runs/20261029_e42_topology_seed0/checkpoints'
    initial=torch.load(source/'best.pt',map_location='cpu',weights_only=False,mmap=True)
    latest=torch.load(source/'latest.pt',map_location='cpu',weights_only=False,mmap=True)
    results={};drift={}
    for key,x in latest['model'].items():
        if x.is_floating_point():
            y=initial['model'][key];d=float((x-y).norm()/y.norm().clamp_min(1e-8))
            drift[key]=d
    for mode in ('latest','restore_graph_local','zero_graph','initial'):
        model=GeometryModel(**latest['model_config']).cuda().eval();model.load_state_dict(initial['model'] if mode=='initial' else latest['model'])
        if mode=='restore_graph_local':
            replacement={k:(initial['model'][k] if '.local.' in k else v) for k,v in latest['model'].items()}
            model.load_state_dict(replacement)
        handles=[];hidden=[]
        if mode=='zero_graph':handles += [layer.local.register_forward_hook(lambda module,args,out:torch.zeros_like(out)) for layer in model.layers]
        handles += [layer.register_forward_hook(lambda module,args,out:hidden.append(out.detach().abs().amax((1,2)).cpu().tolist())) for layer in model.layers]
        with torch.no_grad():v=per_case(model(inputs_only(g)),g,'topology')
        results[mode]={'cases':[{'train_index':index,**{k:float(v[k][i]) for k in ('total','geometry','topology_separation','contact','overlap','center_local','shape')}} for i,index in enumerate(ids)],'hidden_by_layer':hidden}
        for h in handles:h.remove()
        del model;torch.cuda.empty_cache()
    results.update(parameter_relative_drift=drift,unique_train_cases=5,parameter_updates=0,test_read=False,
        scope='Post hoc five-case interventions. Restoring graph parameters in a changed network is not a trained ablation.')
    save(HERE/'runs/failure_diagnosis/graph_interventions.json',results)
    for mode in ('latest','restore_graph_local','zero_graph','initial'):
        r=results[mode];print(mode,[(x['train_index'],x['total']) for x in r['cases']],max(max(layer) for layer in r['hidden_by_layer']),flush=True)
    print('graph relative drifts',[(k,v) for k,v in drift.items() if '.local.weight' in k],flush=True)
if __name__=='__main__':main()
