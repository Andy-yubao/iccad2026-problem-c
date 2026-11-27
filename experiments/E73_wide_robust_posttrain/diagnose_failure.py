"""One rejected training batch; compare failed pre-step weights to E72 epoch0.

The checkpoint RNG is after the rejected replay. Eval is deterministic and
diagnostic; this is not a claim of exact replay of the failed Dropout mask.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import torch
from .io import HERE,ROOT,save,sha
from .train import seed_all
from .data import load_rows,collate
from .model import GeometryModel
from .batches import inputs_only
from .loss import per_case

def main():
    seed_all(0)
    source=ROOT/'experiments/E72_topology_posttrain/runs/20261029_e42_topology_seed0'
    latest=torch.load(source/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
    progress=latest['progress'];ids=progress['order'][progress['cursor']:progress['cursor']+32].tolist()
    rows=load_rows('train',ids,use_cache=False);g,_=collate(rows,'cuda')
    results={}
    for name,path in [('initial',source/'checkpoints/best.pt'),('pre_rejected_step',source/'checkpoints/latest.pt')]:
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
        hidden=[];handles=[layer.register_forward_hook(lambda module,args,out:hidden.append(float(out.detach().abs().max()))) for layer in model.layers]
        v=per_case(model(inputs_only(g)),g,'topology');v['total'].mean().backward()
        norm=torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
        cases=[{'train_index':index,'member':list(row['member']),'n':row['graph']['n'],
            **{k:float(v[k][i].detach()) for k in ('total','geometry','center_local','shape','relative_local','overlap','contact','boundary','mib','bbox','topology_separation')}} for i,(index,row) in enumerate(zip(ids,rows))]
        results[name]={'checkpoint_sha256':sha(path),'mean_total':float(v['total'].mean().detach()),'gradient_norm':float(norm),
            'mean_G':float(v['geometry'].mean().detach()),'mean_topology':float(v['topology_separation'].mean().detach()),
            'layer_hidden_max':hidden,'cases_sorted':sorted(cases,key=lambda c:c['total'],reverse=True)}
        for h in handles:h.remove()
        del model,v;torch.cuda.empty_cache()
    results.update(batch_indices=ids,step=progress['step'],epoch=progress['epoch'],cursor=progress['cursor'],
        unique_train_cases=32,scope='one rejected training batch only; zero parameter updates',test_read=False,
        replay_limit='Saved RNG is after rejected Dropout replay; eval diagnostic is not exact failed-mask reproduction',
        inherited_elapsed_seconds=latest['elapsed_seconds'],projection=latest['graph_step_control'])
    save(HERE/'runs/failure_diagnosis/rejected_batch.json',results)
    for k in ('initial','pre_rejected_step'):
        r=results[k];print(k,r['mean_total'],r['gradient_norm'],'hidden',max(r['layer_hidden_max']),'top3',r['cases_sorted'][:3],flush=True)
if __name__=='__main__':main()
