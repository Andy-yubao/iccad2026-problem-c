"""Bounded synthetic mathematical audit; no real data, model or GPU read."""
import json
from pathlib import Path
import torch
from .overlap import axis_distance, centerline_distance, overlap_per_layout


def value_gradient(delta, extent=(2., 2.), scale=2.):
    d = torch.tensor(delta, dtype=torch.float64, requires_grad=True)
    e = torch.tensor(extent, dtype=torch.float64, requires_grad=True)
    value = centerline_distance(d, e, torch.tensor(scale, dtype=torch.float64))
    gd, ge = torch.autograd.grad(value, (d, e))
    if not all(bool(torch.isfinite(t).all()) for t in (value, gd, ge)):
        raise AssertionError('Nonfinite centerline value/derivatives')
    return {'delta': list(delta), 'extent': list(extent), 'distance': float(value.detach()),
            'delta_gradient': gd.tolist(), 'extent_gradient': ge.tolist()}


def main():
    torch.set_num_threads(1)
    cases = [value_gradient(d, e) for d, e in [
        ((1., 1.), (2., 2.)), ((0., 1.), (2., 2.)),
        ((1.1, 1.), (2., 2.)), ((0., 0.), (2., 2.)),
        ((1e-8, -1e-8), (2., 2.)), ((3., 1.), (2., 2.)),
        ((0., 1.), (20., 2.)), ((.05, .1), (.1, 10.)),
        ((0., 2.), (2., 2.)), ((0., 0.), (0., 0.))]]
    diagonal, strip = cases[:2]
    assert abs(diagonal['distance'] - 2**.5) < 1e-12
    assert strip['distance'] == 1.
    assert cases[5]['distance'] == 0. and cases[8]['distance'] == 0.
    shifts = []
    for base in ((1., 1.), (1.1, 1.)):
        initial = value_gradient(base)['distance']
        for move in ((.01, 0.), (0., .01), (.01, .01)):
            updated = value_gradient(tuple(a+b for a, b in zip(base, move)))['distance']
            shifts.append({'base': list(base), 'outward_move': list(move), 'loss_change': updated-initial})
    assert all(x['loss_change'] < 0 for x in shifts[:3])
    # This expected counterexample must not be disguised as a successful
    # global-monotonicity claim or "fixed" with an unrecorded backward rule.
    assert shifts[4]['loss_change'] > 0
    finite_difference=[]
    for d, e in [((1.1, 1.), (2., 2.)), ((.05, .1), (.1, 10.)), ((0., 1.), (20., 2.))]:
        record=value_gradient(d,e);difference=[]
        for j in (0,1):
            a=list(d);b=list(d);a[j]+=1e-6;b[j]-=1e-6
            fd=(value_gradient(a,e)['distance']-value_gradient(b,e)['distance'])/2e-6
            difference.append(abs(fd-record['delta_gradient'][j]))
        assert max(difference)<1e-6
        finite_difference.append({'delta':list(d),'max_error':max(difference)})
    for d,e in [((1.1,1.),(2.,2.)),((.05,.1),(.1,10.)),((0.,0.),(2.,2.))]:
        original=value_gradient(d,e)['distance']
        assert abs(value_gradient(tuple(-x for x in d),e)['distance']-original)<1e-12
        assert abs(value_gradient(tuple(7*x for x in d),tuple(7*x for x in e),14.)['distance']-7*original)<1e-10
    delta=torch.tensor([[[[0.,0.],[1.,1.],[0.,0.]],
                         [[-1.,-1.],[0.,0.],[0.,0.]],
                         [[0.,0.],[0.,0.],[0.,0.]]]],dtype=torch.float64,requires_grad=True)
    extent=torch.full_like(delta,2.)
    valid=torch.tensor([[True,True,False]])
    pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(3,dtype=torch.bool)[None]
    scale=torch.full((1,3,3),2.,dtype=torch.float64)
    values={kind:overlap_per_layout(delta,extent,scale,pairs,valid,kind) for kind in ('original','centerline_v1')}
    assert abs(float(values['original'].detach())-.25)<1e-12
    assert abs(float(values['centerline_v1'].detach())-.5)<1e-12
    for value in values.values():
        gradient=torch.autograd.grad(value.sum(),delta,retain_graph=True)[0]
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient[:,2]==0).all()) and bool((gradient[:,:,2]==0).all())
    report={'status':'passed_with_documented_direction_counterexample','real_cases_read':0,'test_read':False,
            'formula':'centerline_v1, actual autodiff, relative near-coincidence fallback 1e-4',
            'cases':cases,'finite_moves':shifts,'finite_difference':finite_difference,
            'pair_mask_and_ordered_aggregation':'passed',
            'important_limit':'Radial scalar ordering and global outward monotonicity are incompatible for the user example; this candidate does not guarantee outward gradients on both axes everywhere.',
            'formal_training_started':False}
    target=Path(__file__).resolve().parent/'runs/math_check/results.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'path':str(target),'real_cases_read':0},ensure_ascii=False))


if __name__=='__main__':
    main()
