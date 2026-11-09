"""CPU-only diagnostic prototype, original five train cases; not a training model."""
import time,json
import torch
from experiments.E95_single_pass_corrector.io import HERE,save
from experiments.E95_single_pass_corrector.data import load_rows,collate,inputs_only
from experiments.E95_single_pass_corrector.model import GeometryModel,decode
from experiments.E95_single_pass_corrector.initialization import load
from experiments.E95_single_pass_corrector.loss import per_case
from experiments.E95_single_pass_corrector.runtime import charge

FOLDER=HERE/'runs/20261101_port_compression_preparation'

def network_view(g,mode):
    if mode=='full':return g
    b,n,p=g['p2b_weights'].shape;result=dict(g)
    if mode=='module_centroid':
        result.update(port_xy=g['port_xy'][:,:0],port_valid=g['port_valid'][:,:0],p2b_weights=g['p2b_weights'][:,:,:0])
        return result
    k=32;xy=torch.zeros(b,k,2);valid=torch.zeros(b,k,dtype=torch.bool);weights=torch.zeros(b,n,k)
    for i in range(b):
        mass=g['p2b_weights'][i].sum(0);selected=g['port_valid'][i]&(mass>0)
        if not bool(selected.any()):continue
        positions=g['port_xy'][i,selected];w=mass[selected]
        lo=positions.amin(0);extent=(positions.amax(0)-lo).clamp_min(1e-12)
        cell=((positions-lo)/extent*torch.tensor([8.,4.])).floor().long()
        cell[:,0].clamp_(0,7);cell[:,1].clamp_(0,3);index=cell[:,1]*8+cell[:,0]
        total=torch.zeros(k).index_add_(0,index,w)
        first=torch.zeros(k,2).index_add_(0,index,positions*w[:,None])
        xy[i]=first/total.clamp_min(1e-12)[:,None];valid[i]=total>0
        weights[i].scatter_add_(1,index[None].expand(n,-1),g['p2b_weights'][i,:,selected])
    result.update(port_xy=xy,port_valid=valid,p2b_weights=weights)
    return result

def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    ids=[1683,10027,5474,10140,15001];rows=load_rows('train',ids);g=inputs_only(collate(rows,'cpu'))
    # Independently verify legacy features already preserve per-module mass/mean.
    total=g['p2b_weights'].sum(-1);mean=(g['p2b_weights'][...,None]*g['port_xy'][:,None]).sum(2)/total.clamp_min(1e-12)[...,None]
    torch.testing.assert_close(g['features'][...,-3],total.log1p(),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(g['features'][...,-2:],mean,rtol=1e-5,atol=1e-6)
    partial=network_view(g,'regional32');full=network_view(g,'module_centroid')
    torch.testing.assert_close(partial['p2b_weights'].sum(-1),total,rtol=1e-6,atol=1e-5)
    original_first=(g['port_xy']*total.new_tensor(1.)*g['p2b_weights'].sum(1)[...,None]).sum(1)
    compressed_first=(partial['port_xy']*partial['p2b_weights'].sum(1)[...,None]).sum(1)
    torch.testing.assert_close(original_first,compressed_first,rtol=1e-5,atol=1e-5)
    # Group assignment must not depend on original port ID/order.
    rev=dict(g);rev['port_xy']=g['port_xy'].flip(1);rev['port_valid']=g['port_valid'].flip(1);rev['p2b_weights']=g['p2b_weights'].flip(2)
    reverse=network_view(rev,'regional32')
    torch.testing.assert_close(reverse['port_xy'],partial['port_xy'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(reverse['p2b_weights'],partial['p2b_weights'],rtol=1e-5,atol=1e-5)
    # Existing encoder supports zero external tokens and one simultaneous head.
    model=GeometryModel(width=192,layers=12,heads=4,dropout=.05,normalize_local=True,attention_backend='manual').eval();load(model)
    reports={}
    with torch.no_grad():
        draft=model.draft(g);base=per_case(draft,g,draft)['total']
        for mode,view in [('full',g),('regional32',partial),('module_centroid',full)]:
            calls=[];hook=model.corrector.head[-1].register_forward_hook(lambda *_:calls.append(1))
            output=model.corrector(view,draft);hook.remove()
            assert len(calls)==1
            torch.testing.assert_close(decode(output,g),decode(draft,g),rtol=0,atol=0)
            torch.testing.assert_close(per_case(output,g,draft)['total'],base,rtol=0,atol=0)
            reports[mode]=dict(external_token_slots=view['port_xy'].shape[1],same_legacy_module_features=torch.equal(view['features'],g['features']),zero_head_decoded_identity_exact=True,full_original_port_Loss_unchanged=True,one_head_call=True)
    result=dict(status='passed',device='CPU only; active GPU training untouched',unique_train_cases=5,indices=ids,
        original_public_port_counts=[len(r['public_ports']['port_xy']) for r in rows],per_module_weight_and_centroid_in_existing_features=True,
        regional32_weight_mass_conserved=True,regional32_global_weighted_first_moment_conserved=True,
        port_reordering_invariant=True,per_module_regional_centroids='not exactly conserved by shared bin centres; original per-module centroid retained in legacy features',
        modes=reports,originals_modified=False,optimizer_steps=0,formal_training_started=False,test_read=False,
        limitation='implementation evidence only; no quality/generalization or speed claim; grid and centroid compression are lossy, exact wirelength still uses original public edges')
    save(FOLDER/'results.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    begin=time.perf_counter()
    try:main()
    finally:charge(time.perf_counter()-begin)
