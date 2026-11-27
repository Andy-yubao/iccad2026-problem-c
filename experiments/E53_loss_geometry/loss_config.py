"""Fixed Loss expressions; only these expressions differ between study arms."""
BASE = {'center':.35, 'shape':.1, 'relative':.125, 'overlap':.75,
        'contact':1., 'boundary':.5, 'mib':.5, 'bbox':.15,
        'module':.75, 'compactness':.05, 'clearance':0., 'wire_guard':0.}


def configuration(arm):
    if arm not in ('geometry','tolerant','clearance','wire_guard','anchor_ablation','count_ablation'):
        raise ValueError('Unknown fixed Loss arm: '+arm)
    weights=dict(BASE)
    band=arm in ('tolerant','clearance','wire_guard','count_ablation')
    if arm=='anchor_ablation': weights.update(center=1.,shape=.25,relative=.25)
    if arm=='clearance': weights['clearance']=.5
    if arm=='wire_guard': weights['wire_guard']=.15
    return {'arm':arm,'revision':'E53.fixed_geometry_loss.v1','weights':weights,
            'center_band_local':.1 if band else 0.,'shape_band_logratio':.1 if band else 0.,
            'relative_band_local':.1 if band else 0.,'count_factor':arm!='count_ablation',
            'schedule':'fixed full weights from first batch; identical across splits and epochs',
            'selection':'same total on fixed dev500, never terminal Q'}
