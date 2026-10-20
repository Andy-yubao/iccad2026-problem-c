"""Verify spawned cache writers on the same five already checked members."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_name]='1'
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
from .cache import initialize, prepare_part
from .io import HERE, save


def main():
    folder=HERE/'runs/20261020_smoke5_seed0'
    original=torch.load(folder/'tensor_roundtrip.pt',weights_only=False)['rows']
    target=folder/'spawn_cache'; target.mkdir(exist_ok=False)
    results=[]
    for role,indices in (('train',[0,1,2,0]),('dev_monitor500',[0,1])):
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(role,)) as pool:
            futures=[pool.submit(prepare_part,slot,[index],str(target/f'{role}_{slot}.pt')) for slot,index in enumerate(indices)]
            for slot,(index,future) in enumerate(zip(indices,futures)):
                info=future.result(); assert info['torch_threads']==1; results.append(info)
                row=torch.load(target/f'{role}_{slot}.pt',weights_only=False)['rows'][0]
                expected=original[index if role=='train' else index+3]
                assert tuple(row['member'])==tuple(expected['member'])
                for key in expected['graph']:
                    a,b=row['graph'][key],expected['graph'][key]
                    assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
    save(folder/'spawn_cache_results.json',{'status':'passed','unique_real_cases':5,
        'worker_records':results,'threads_per_worker':1,'test_read':False})
    print('Spawned cache and original reference tensors agree on the same five cases',flush=True)


if __name__=='__main__': main()
