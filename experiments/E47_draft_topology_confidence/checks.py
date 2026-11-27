"""Small algebra and public constraint checks; no development evaluation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import dataclasses
import numpy as np
from .confidence import Guidance
from .fast_geometry import topology,contact_tree,reduce_edges,step
from .projection import solve
from .backend.frozen.packing import prepare_problem
from .io import HERE,save


def problem(x,preplaced=False):
    c=np.zeros((len(x),5));c[:,3]=1
    if preplaced:c[:,0:2]=1
    return prepare_problem(len(x),np.prod(x[:,2:],1),[],[],[],c,x)


def main():
    x=np.array([[0.,0.,1.,1.],[1.,0.,1.,1.],[2.,0.,1.,1.]])
    p=problem(x);g=Guidance.build(x,p,'oracle_structured',x)
    full=g.impose(x,p,topology(x,p.constraints,array=True));tree=contact_tree(x,p,full,g)
    assert len(g.hard_contacts(tree))==2
    outputs=[]
    for fn in ('projection','shape'):
        y,stats=solve(p,x,reduce_edges(full,p.n),tree,seconds=.2,guidance=g,quadratic=True) if fn=='projection' else step(p,x,reduce_edges(full,p.n),full_edges=full,seconds=.2,guidance=g)
        assert y is not None,stats
        for i,j,a in g.hard_contacts(tree):
            assert abs(y[j,a]-y[i,a]-y[i,a+2])<1e-6
            b=1-a;assert min(y[i,b]+y[i,b+2],y[j,b]+y[j,b+2])-max(y[i,b],y[j,b])>0
        outputs.append(fn)
    assert not g.original.flags.writeable and np.array_equal(g.original,x)
    empty=dataclasses.replace(g,contact_confidence=np.zeros((3,3)),committed=set(),disabled=set(),events=[])
    empty.impose(x,p,full);assert len(empty.hard_contacts(tree))==0
    # Installing 1->2 while preserving 2->0->1 creates a mixed-axis cycle.
    base=np.array([[0,1,0],[2,0,0],[2,1,0]])
    cycle=Guidance.build(x,p,'oracle_structured',x);out=cycle.impose(x,p,base)
    assert any(e['reason']=='axis_cycle' for e in cycle.events)
    assert cycle.release_one() or not cycle.committed
    anchored=x.copy();anchored[1,0]=3;anchored[2,0]=6
    ap=problem(anchored,True);ag=Guidance.build(anchored,ap,'oracle_commit',x);af=ag.impose(anchored,ap,topology(anchored,ap.constraints,array=True))
    assert not len(ag.hard_contacts(contact_tree(anchored,ap,af,ag)))
    assert len(ag.disabled)>0 and np.array_equal(ap.specs,anchored)
    permuted=x[[2,0,1]].copy()
    for kind in ('oracle_full','oracle_hybrid','oracle_hybrid_contacts'):
        gg=Guidance.build(permuted,p,kind,x);ff=gg.impose(permuted,p,topology(permuted,p.constraints,array=True))
        i,j,a=ff.T;assert np.all(x[j,a]-x[i,a]-x[i,a+2]>=-1e-7)
        assert not gg.static_edges.flags.writeable and np.array_equal(gg.original,permuted)
        if kind=='oracle_hybrid_contacts':
            assert len(gg.hard_contacts([]))==2
            y,stats=solve(p,permuted,reduce_edges(ff,p.n),[],seconds=.2,guidance=gg,quadratic=True)
            assert y is not None,stats
            for i,j,a in gg.hard_contacts([]):assert abs(y[j,a]-y[i,a]-y[i,a+2])<1e-6
    save(HERE/'runs/commit_checks.json',{'exact_contact_equations':outputs,'zero_confidence_no_commit':True,'axis_cycle_retracted':True,'immutable_inputs_preserved':True,'original_read_only':True,'test_read':False})
    save(HERE/'runs/global_checks.json',{'hybrid_and_full_witness_relations_valid':True,'global_forest_exact_contacts':True,'unchanged_full_draft_start':True,'test_read':False})
    print('Commit checks passed',flush=True)


if __name__=='__main__':main()
