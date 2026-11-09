from .losses import WEIGHTS

def configuration(arm):
    if arm!='large_original160':raise ValueError('Only the authorized large run is supported')
    return {'count_factor':False,'revision':'E61.original_fixed_geometry.v1','weights':dict(WEIGHTS),
        'overlap_kind':'original','overlap_kappa':1.,'selection':'same fixed dev500 total used by backward',
        'schedule':'full fixed weights from the first update; no ramp or alternate proxy'}
