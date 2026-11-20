"""Disposable public compatibility and train-only static producer checks."""
import ast
import pathlib
import time
import numpy as np
import torch
from . import prepare as prep
from .transform import compatible_areas,supervision
from .batches import sample


def main():
    started=time.perf_counter();torch.set_num_threads(1)
    root=prep.HERE
    for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    c=np.zeros((2,5),np.float32);c[:,2]=1
    a,reason=compatible_areas(np.array([2,8],np.float32),c,np.ones((2,2)))
    assert reason is None and np.array_equal(a,[5,5])
    bad=c.copy();bad[:,0]=1
    assert compatible_areas([2,8],bad,[[1,2],[2,4]])[1]=='locked_dimensions_disagree'
    bad=c.copy();bad[0,0]=1
    assert compatible_areas([2,8],bad,[[1,2],[1,1]])[1]=='mean_area_incompatible_with_locked_shape'
    original=np.array([[3,3,1,2],[8,8,4,2]],np.float32)
    target,audit=supervision(original,a,c,np.full((2,4),-1,np.float32))
    np.testing.assert_allclose(target.numpy()[:,:2]+target.numpy()[:,2:]/2,original[:,:2]+original[:,2:]/2,atol=1e-6)
    np.testing.assert_allclose(target.numpy()[0,2:],target.numpy()[1,2:],rtol=0,atol=0)
    shards=sorted(str(p.relative_to(prep.RAW)).replace('\\','/') for p in prep.RAW.glob('worker_*/layouts_*.th'))
    np.random.default_rng(prep.SEED).shuffle(shards)
    chosen=None
    for shard in shards[:10]:
        result=prep.inspect(('train',shard,[(i,i) for i in range(112)]))
        if len(result['accepted'])>=5:
            chosen=result;break
    assert chosen is not None
    selected=chosen['accepted'][:5]
    requests=[(i,r['member'][1],r['n']) for i,r in enumerate(selected)]
    prep.HERE=root/'preparation_checks'
    part=prep.cache_part(('train',0,[(chosen['shard'],requests,chosen['source'])],'disposable-preflight'))
    cache=torch.load(prep.HERE/part['path'],map_location='cpu',weights_only=False)
    for row in cache['rows']:
        g=row['graph'];changed=row['supervision_xywh'].clone();changed[:,:2]+=7
        alternate=sample(row['inputs'],changed)
        for key in ('features','adjacency','mib','group','mask','area','known','specs','constraints','scale'):
            assert torch.equal(g[key],alternate[key]),'Supervision leaked into public features'
        assert bool((g['mask']==1).all())
    prep.HERE=root
    report=dict(status='passed',elapsed_seconds=time.perf_counter()-started,train_cases=5,test_free_labels_read=False,
        synthetic_compatibility_rules=True,shared_MIB_supervision=True,public_features_independent_of_targets=True,
        all_F_P_channels_predicted=True,static_cache_smoke=True,sources={p.name:prep.sha(p) for p in root.glob('*.py')})
    prep.save(root/'preflight.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
