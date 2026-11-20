"""Synthetic definition checks only; no dataset, optimizer, model or postprocessing."""
import json
from pathlib import Path
import numpy as np
import torch
from metric import target, similarity, layout_similarity, surrogate_loss


def main():
    rng=np.random.default_rng(20261115)
    ref=np.array([(x*3.,y*3.) for y in range(4) for x in range(4)])
    areas=np.ones(len(ref))
    goal=target(ref,areas)
    output={}
    def record(name,p):
        result=similarity(p,goal)
        raw=torch.tensor(p,dtype=torch.float64,requires_grad=True)
        loss=surrogate_loss(raw,goal)
        gradient=torch.autograd.grad(loss['total'],raw)[0]
        assert torch.isfinite(gradient).all() and torch.isfinite(loss['total'])
        output[name]=dict(result,loss=float(loss['total']),gradient_norm=float(gradient.norm()))
        return result
    assert record('reference',ref)['topology_similarity']==1
    assert record('translated',ref+np.array([100.,-50.]))['topology_similarity']==1
    assert record('centers_scaled',ref*.2)['topology_similarity']==1
    assert record('tiny_jitter_at_neighbor_cutoff_ties',ref+rng.normal(0,1e-4,ref.shape))['topology_similarity']==1
    assert record('coincident_collapse',np.zeros_like(ref))['topology_similarity']==0
    assert record('random_identity_permutation',ref[rng.permutation(len(ref))])['topology_similarity']<1
    refrect=np.c_[ref-.5,np.ones((len(ref),2))]
    overlapping=np.c_[ref-5,np.full((len(ref),2),10.)]
    assert layout_similarity(overlapping,refrect,areas)['topology_similarity']==1
    output['larger_rectangles_same_centers']=layout_similarity(overlapping,refrect,areas)
    # Identical x/y orders do not determine neighborhood structure.
    line=np.array([[0.,0.],[1.,0.],[2.,0.],[20.,0.]])
    changed=np.array([[0.,0.],[1.,0.],[19.,0.],[20.,0.]])
    linegoal=target(line,np.ones(4),k=1)
    result=similarity(changed,linegoal)
    assert result['direction_match']==1 and result['topology_similarity']<1
    output['same_orders_changed_neighbors']=result
    # Local-neighbor-only metrics miss moving intact clusters to opposite sides.
    cluster=np.array([[0.,0.],[1.,.2],[.3,1.],[20.,0.],[21.,.2],[20.3,1.]])
    moved=cluster.copy();moved[:3,0]+=20;moved[3:,0]-=20
    result=similarity(moved,target(cluster,np.ones(6),k=2))
    assert result['neighborhood_match']==1 and result['topology_similarity']<1
    output['intact_clusters_exchanged']=result
    # Index permutations must not arbitrarily resolve kth-distance ties.
    permutation=rng.permutation(len(ref))
    noisy=ref+rng.normal(0,.8,ref.shape)
    actual=similarity(noisy,goal)['topology_similarity']
    permuted=similarity(noisy[permutation],target(ref[permutation],areas[permutation]))['topology_similarity']
    assert abs(actual-permuted)<1e-12
    assert output['reference']['loss']==0 and output['translated']['loss']<1e-12 and output['centers_scaled']['loss']<1e-12
    assert output['coincident_collapse']['loss']>0 and output['coincident_collapse']['gradient_norm']>0
    assert output['random_identity_permutation']['loss']>0 and output['random_identity_permutation']['gradient_norm']>0
    wrong=ref[rng.permutation(len(ref))]+rng.normal(0,.03,ref.shape)
    before=surrogate_loss(torch.tensor(wrong,dtype=torch.float64),goal)['total']
    expanded=surrogate_loss(torch.tensor(wrong*100,dtype=torch.float64),goal)['total']
    torch.testing.assert_close(before,expanded,rtol=1e-12,atol=1e-12)
    small_ref=rng.normal(0,3,(5,2))
    small_goal=target(small_ref,np.ones(5),k=2)
    small_prediction=torch.tensor(rng.normal(0,3,(5,2)),dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda p:surrogate_loss(p,small_goal)['total'],(small_prediction,),eps=1e-6,atol=1e-5,rtol=1e-4)
    pre=np.zeros(len(ref),bool);pre[:4]=True
    pregoal=target(ref,areas,preplaced=pre)
    assert not (pregoal.eligible&pre[:,None]&pre[None,:]).any()
    output['verification']=dict(status='passed',translation_and_scale_invariant=True,
        shape_and_overlap_independent_at_fixed_centers=True,reference_score_one_and_loss_zero=True,
        collapse_score_zero_and_nonzero_finite_gradient=True,local_and_global_failure_modes_detected=True,
        equivariant_under_id_permutation_with_distance_ties=True,PP_relations_excluded_only_from_scoring=True,
        ambiguous_neighbor_cutoff_ties_do_not_force_arbitrary_labels=True,
        finite_difference_gradient_check_passed=True,global_expansion_does_not_improve_surrogate=True,
        synthetic_only=True,training=False,optimizer_updates=0,dataset_payloads=0)
    path=Path(__file__).with_name('synthetic_checks.json')
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
