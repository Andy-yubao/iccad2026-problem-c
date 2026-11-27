"""Exactly 5% radial center spreading with original widths/heights preserved."""
import numpy as np

ARMS=('reference','radial5_literal','radial5_restore','radial5_rebase_restore')


def expand(draft,inputs,arm):
    if arm not in ARMS:raise ValueError(arm)
    original=np.asarray(draft,dtype=float);raw=original.copy();c=np.asarray(inputs[5]);specs=np.asarray(inputs[6]);preplaced=c[:,1]!=0
    lo=original[:,:2].min(0);hi=(original[:,:2]+original[:,2:]).max(0);origin=(lo+hi)/2
    if arm!='reference':
        center=original[:,:2]+original[:,2:]/2
        raw[:,:2]=origin+1.05*(center-origin)-original[:,2:]/2
    actual=raw.copy();translation=np.zeros(2)
    if arm=='radial5_rebase_restore':
        translation=np.maximum(0,-actual[:,:2].min(0));actual[:,:2]+=translation
    if arm in ('radial5_restore','radial5_rebase_restore'):actual[preplaced]=specs[preplaced]
    assert np.array_equal(raw[:,2:],original[:,2:])
    assert np.array_equal(actual[:,2:],original[:,2:])
    if arm!='reference':
        expected=1.05*(original[:,:2]+original[:,2:]/2-origin)
        actual_offset=raw[:,:2]+raw[:,2:]/2-origin
        assert np.all(np.abs(actual_offset-expected)<=1e-10+1e-11*np.abs(expected))
    if arm=='radial5_rebase_restore':assert (actual[:,:2]>=0).all()
    if arm in ('radial5_restore','radial5_rebase_restore'):assert np.array_equal(actual[preplaced],specs[preplaced])
    return actual,raw,{'factor':1. if arm=='reference' else 1.05,'origin':origin.tolist(),'translation':translation.tolist(),
        'restored_preplaced':int(preplaced.sum()) if arm in ('radial5_restore','radial5_rebase_restore') else 0}
