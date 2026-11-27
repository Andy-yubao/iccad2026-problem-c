"""Five train cases: MIB-coherent warm output and anchored path behavior."""
import torch,time
from experiments.E78_connectivity_posttrain.io import HERE,save
from experiments.E78_connectivity_posttrain.data import load_rows,collate
from experiments.E78_connectivity_posttrain.batches import inputs_only,decode
from experiments.E78_connectivity_posttrain.model import GeometryModel
from experiments.E78_connectivity_posttrain.pretrained_model import GeometryModel as OldModel
from experiments.E78_connectivity_posttrain.initialization import load
from experiments.E78_connectivity_posttrain.arms import model_config
from experiments.E78_connectivity_posttrain.loss import per_case
from experiments.E78_connectivity_posttrain.train import seed_all

def main():
    seed_all(0);ids=[1683,10027,5474,18140,25001];rows=load_rows('train',ids);g,_=collate(rows,'cuda')
    model=GeometryModel(**model_config('topology')).cuda().eval();source=load(model,model_config('topology'))
    old=OldModel(**source['model_config']).cuda().eval();old.load_state_dict(source['model'])
    with torch.no_grad():a=old(inputs_only(g));b=model(inputs_only(g))
    av=per_case(a,g,'topology');bv=per_case(b,g,'topology')
    grad=torch.autograd.grad(bv['total'].sum(),tuple(p for p in model.parameters() if p.requires_grad),allow_unused=True) if b.requires_grad else None
    result=[]
    for i,index in enumerate(ids):
        valid=g['valid'][i];p=decode(b,g)[i,valid]
        torch.testing.assert_close(p[:,2:].prod(-1),g['area'][i,valid],rtol=3e-6,atol=2e-8)
        result.append({'train_index':index,'before':{k:float(av[k][i]) for k in ('total','common_original_G','true_intersection','mib','anchored_corridor')},
            'coherent_initial':{k:float(bv[k][i]) for k in ('total','common_original_G','true_intersection','mib','anchored_corridor')}})
    save(HERE/'runs/diagnosis/five_case_coherent_proposal.json',{'cases':result,'test_read':False,'formal_training_started':False})
    print(result,flush=True)

if __name__=='__main__':main()
