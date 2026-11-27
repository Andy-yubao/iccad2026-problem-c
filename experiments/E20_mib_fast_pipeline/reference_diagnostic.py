"""Offline supervision-consistency diagnostic; never imported by any solver."""
import json
import numpy as np
import torch
from .io import ROOT,HERE,save,sha
from datasets.reference_layouts import REGISTRY,registry


def main():
    torch.set_num_threads(2)
    members={tuple(r['member']) for r in registry()['sets']['dev_monitor500']['records']}
    source=ROOT/'experiments/E17_e10_geometry_v5/runs/prepared_v5_e10_cache_v1/dev'
    manifest=json.loads((source/'manifest.json').read_text());records=[]
    for part in manifest['parts']:
        for row in torch.load(source/part['name'],weights_only=False)['rows']:
            if tuple(row['member']) not in members:continue
            g=row['graph'];c=g['constraints'].numpy();target=g['target'].numpy();free=g['mask'][:,2].numpy()>0
            fitted=target[:,2].copy();groups=0
            for group in np.unique(c[:,2]):
                if group<=0:continue
                ids=np.flatnonzero(c[:,2]==group);rigid=ids[~free[ids]]
                value=float(g['known'][rigid[0],2]) if len(rigid) else float(target[ids,2].mean())
                fitted[ids]=value;groups+=1
            error=(fitted-target[:,2])**2
            records.append({'member':row['member'],'n':int(g['n']),'mib_groups':groups,
                'irreducible_shape_mse':float(error[free].mean()) if free.any() else 0.,
                'label_ratio_spread':float(np.max(np.abs(fitted[free]-target[free,2]))) if free.any() else 0.})
    assert len(records)==500
    q=np.array([r['irreducible_shape_mse'] for r in records])
    save(HERE/'runs/20261020_reference_consistency/diagnostic.json',{
        'count':500,'registry_sha256':sha(REGISTRY),'source_manifest_sha256':sha(source/'manifest.json'),
        'mean_irreducible_shape_mse':float(q.mean()),'cases_with_incompatible_ratio_labels':int((q>1e-12).sum()),
        'max_irreducible_shape_mse':float(q.max()),'records':records,
        'scope':'offline label analysis only, no model inputs or solver changes','test_read':False})
    print({'mean_irreducible_shape_mse':float(q.mean()),'incompatible_cases':int((q>1e-12).sum())})


if __name__=='__main__':main()
