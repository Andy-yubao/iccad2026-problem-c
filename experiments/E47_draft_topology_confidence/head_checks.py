"""Five-train-case auxiliary interface check; no postprocessing."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import numpy as np
import torch
from .data import load_rows,collate as base_collate
from .batches import inputs_only
from .model import GeometryModel,decode
from .head import RelationHead,collate,labels,loss,predict_latent
from .inference import predict
from .io import HERE,WEIGHTS,WEIGHT_SHA,sha,save
from .backend.frozen.packing import prepare_problem
from .learned_guidance import build
from .fast_geometry import topology
from .global_information import acyclic


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
    checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**checkpoint['model_config']).eval();base.load_state_dict(checkpoint['model'])
    rows=load_rows('train',indices=[0,1,2,3,4]);a,features,_=predict_latent(base,rows[0]['case'].solver_inputs);b,_=predict(base,rows[0]['case'].solver_inputs)
    assert np.array_equal(a,b),'Latent capture changed original draft'
    base.cuda();g=base_collate(rows,'cuda')[0];holder={};handle=base.head.register_forward_pre_hook(lambda module,args:holder.__setitem__('latent',args[0].detach()))
    with torch.no_grad():state=base(inputs_only(g));draft=decode(state,g).cpu();h=holder['latent'].cpu()
    handle.remove();prepared=[]
    for at,row in enumerate(rows):
        n=row['graph']['n'];target=labels(row['graph']['reference']);assert target.any(-1).all()
        prepared.append({'n':n,'latent':h[at,:n],'draft':draft[at,:n],'adjacency':row['graph']['adjacency'],
            'constraints':row['graph']['constraints'],'labels':target})
    batch,y,mask=collate(prepared,'cuda');head=RelationHead().cuda();value=loss(head(batch),y,mask);value.backward()
    assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in head.parameters())
    head.cpu().eval()
    with torch.no_grad():prob=torch.sigmoid(head(features))[0].numpy()
    p=prepare_problem(*rows[0]['case'].solver_inputs)
    for mode in ('learned','learned_uniform','learned_geometric','learned_shuffle'):
        guide=build(a,p,prob,mode,0);out=guide.impose(a,p,topology(a,p.constraints,array=True));assert acyclic(out,p.n)
        assert guide.original.flags.writeable is False and np.array_equal(guide.original,a)
    assert sha(WEIGHTS)==WEIGHT_SHA
    folder=HERE/'runs/head_interface_check/cache';folder.mkdir(exist_ok=True,parents=True);torch.save(prepared,folder/'five_train_drafts.pt')
    save(HERE/'runs/head_checks.json',{'five_train_cases':True,'latent_hook_original_draft_bitwise_equal':True,'finite_loss_and_gradient':True,
        'model_only_guidance_acyclic_or_deferred':True,'frozen_checkpoint_unchanged':True,'reference_never_enters_network':True,'solver_calls':0,'test_read':False})
    print('HEAD CHECKS PASSED',flush=True)


if __name__=='__main__':main()
