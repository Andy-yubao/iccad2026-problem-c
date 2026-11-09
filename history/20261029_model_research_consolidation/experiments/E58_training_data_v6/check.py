"""Three unique train members and two synthetic MIB cases; no tests or models."""
import argparse
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,mean_area_inputs
from datasets.expanded_reference import ExpandedReferenceLayouts,registry
from .io import ROOT,HERE,RUN,CACHE,read,save,verify_frozen
from .data import prepare


def graph_equal(a,b):
    if set(a)!=set(b):raise AssertionError('Graph schema changed')
    for name in a:
        if isinstance(a[name],torch.Tensor):
            if not torch.equal(a[name],b[name]):raise AssertionError('Tensor mismatch: '+name)
        elif a[name]!=b[name]:raise AssertionError('Graph scalar mismatch: '+name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-ready',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1)
    verify_frozen();reg=registry()
    # Synthetic MIB arithmetic and incompatible locked-shape rejection.
    c=torch.tensor([[0.,0.,1.,0.,0.],[0.,0.,1.,0.,0.]])
    values=(2,torch.tensor([1.,3.]),torch.empty(0,3),torch.empty(0,3),torch.empty(0,2),c,torch.full((2,4),-1.))
    transformed=mean_area_inputs(values)
    assert torch.equal(transformed[1],torch.tensor([2.,2.]))
    assert torch.equal(values[1],torch.tensor([1.,3.])) and transformed[1].sum()==values[1].sum()
    locked_c=c.clone();locked_c[:,0]=1
    incompatible=(*values[:5],locked_c,torch.tensor([[-1.,-1.,1.,1.],[-1.,-1.,2.,.5]]))
    try:mean_area_inputs(incompatible)
    except ValueError as error:assert str(error)=='locked_dimensions_disagree'
    else:raise AssertionError('Incompatible locked MIB shapes were accepted')
    # One same member through original and expanded readers.
    old=prepare(ReferenceLayouts('train')[0]);dataset=ExpandedReferenceLayouts('train')
    fresh=prepare(dataset[0]);assert fresh['member']==old['member'];graph_equal(fresh['graph'],old['graph'])
    denied=[]
    for role in ('local_test','test','tail_test'):
        try:ExpandedReferenceLayouts(role)
        except PermissionError:denied.append(role)
        else:raise AssertionError('Sealed role accepted')
    result={'status':'passed','same_v5_first_member_features':True,'MIB_group_mean_and_sum_verified':True,
        'incompatible_locked_MIB_rejected':True,'test_roles_rejected':denied,'test_payloads_read':False,
        'unique_actual_train_cases':1,'synthetic_MIB_cases':2,'cache_ready_check':args.cache_ready}
    if args.cache_ready:
        from .cache import load_rows,verify
        verify('train');verify('dev')
        ids=[0,16000,32000]
        cached=load_rows('train',ids)
        for index,row in zip(ids,cached):
            direct=prepare(dataset[index]);assert row['member']==direct['member'];graph_equal(row['graph'],direct['graph'])
        assert len(load_rows('train',[16000],train_size=32000))==1
        result['cache_direct_reader_indices_verified']=ids
        result['unique_actual_train_cases']=3
        result['cache_train_count']=64000;result['cache_dev_count']=2000
        # Three actual members plus two synthetic cases comprise this bounded check.
        save(RUN/'verification.json',result)
    else:save(RUN/'preflight.json',result)
    verify_frozen();print(result,flush=True)


if __name__=='__main__':main()
