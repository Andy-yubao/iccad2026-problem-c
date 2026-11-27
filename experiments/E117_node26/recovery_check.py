"""Same five train members: cache tensors exactly match original preprocessing."""
import json
import time
import torch
from datasets.random_reference import RandomReference
from datasets import static_tensor_cache
from .io import OWNER,ROOT,save,sources,sha
from . import data
from .train import seed_all

def main():
    started=time.perf_counter();seed_all(0)
    plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    indices=plan['preflight_train_indices']
    ds=RandomReference('train')
    expected=[]
    for i in indices:
        item=ds[i]
        expected.append(data.make_row(item['solver_inputs'],torch.as_tensor(item['supervision']['reference_xywh']).float(),item['member']))
    states=[]
    actual=data.load_rows('train',indices,beat=lambda **kw:states.append(kw))
    # Force only the same five members through the repaired worker transport;
    # existing production-cache entries remain untouched and reusable.
    transported=static_tensor_cache.load_rows(ds,data.__name__,data.cache_identity(plan),data._checksums,
        indices=indices,workers=4,cache_root=ROOT/'tmp/E117_numpy_transport_check')
    for a,b in zip(actual,transported):
        for k in a['graph']:
            x,y=a['graph'][k],b['graph'][k]
            assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y,k
    for a,b in zip(expected,actual):
        assert a['member']==b['member'] and a['graph'].keys()==b['graph'].keys()
        for k in a['graph']:
            x,y=a['graph'][k],b['graph'][k]
            assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y,k
    original=static_tensor_cache._build
    def forbidden(_):raise AssertionError('Cache reuse attempted raw preprocessing')
    static_tensor_cache._build=forbidden
    try:
        reused=data.load_rows('train',indices)
        assert all(torch.equal(a['graph']['features'],b['graph']['features']) for a,b in zip(actual,reused))
    finally:static_tensor_cache._build=original
    old=json.loads((OWNER/'preflight.json').read_text(encoding='utf-8'))
    old_sources=old['sources']
    unchanged=['model.py','original_model.py','loss.py','loss_config.py','simple_loss.py','group_tree.py',
               'features.py','batches.py','ports.py','augmentation.py','cuda_step.py','captured_projection.py',
               'step_control.py','schedule.py','resident.py']
    current=sources()
    for name in unchanged:
        key=(OWNER/name).relative_to(OWNER.parent.parent).as_posix()
        assert old_sources[key]==current[key],name
    result=dict(status='passed',members=[r['member'] for r in actual],unique_train_cases=5,
                dev_payloads=0,test_read=False,all_cached_fields_bitwise_equal=True,
                repeated_load_without_raw_build=True,cache_build_workers=plan['cache_build_workers'],
                previous_model_Loss_engine_gate_reused=True,unchanged_math_files=unchanged,
                repaired_numpy_worker_transport_bitwise_equal=True,
                elapsed_seconds=time.perf_counter()-started,sources=current)
    save(OWNER/'recovery_20261109/repair_check.json',result)
    old['sources']=current
    old['recovery_validation']=dict(cache_check='recovery_20261109/repair_check.json',
        original_GPU_gate_reused=True,model_Loss_numerical_engine_unchanged=True,
        additional_optimizer_updates=0,reason='Only persistent data loading, authorized resume metadata and durable launch changed')
    save(OWNER/'preflight.json',old)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
