"""Exact complementary split of the active fixed E53 geometry objective."""
PUBLIC_WEIGHTS = {'overlap':.75, 'boundary':.5, 'mib':.5,
                  'module_overlap_counted':.75, 'compactness':.05}
REFERENCE_WEIGHTS = {'center':.35, 'shape':.1, 'relative':.125,
                     'contact':1., 'bbox':.15}

def configuration(arm):
    if arm not in ('no_reference','reference_only'):
        raise ValueError('Unknown arm: '+arm)
    return {'arm':arm, 'revision':'E54.complementary_reference_split.v1',
            'weights':dict(PUBLIC_WEIGHTS if arm=='no_reference' else REFERENCE_WEIGHTS),
            'count_factor':arm=='no_reference',
            'reference_reads':[] if arm=='no_reference' else ['target','reference'],
            'contact_policy':'Reference-derived contact belongs to reference_only; no replacement group loss.',
            'schedule':'fixed full weights, no ramps or weight rescaling',
            'selection':'arithmetic mean of this same total on dev_monitor500'}
