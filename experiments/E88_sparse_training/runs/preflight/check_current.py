"""Five existing train members only; data-only context, deterministic selection and lock."""
import sys,json,time,os,ast,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E88_sparse_training import prepare_data as p
from experiments.E88_sparse_training.cache import load_rows,verify
from experiments.E88_sparse_training.io import HERE,save,sources,exclusive
import torch
begin=time.perf_counter();ids=[1683,10027,5474,10140,15001]
assert torch.get_num_threads()==1 and all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
rows=load_rows('train',ids);stats=[p.public_statistics(i,r) for i,r in zip(ids,rows)]
assert len(stats)==5 and all(0<=r['density']<=1 and 0<=r['port_coverage']<=1 for r in stats)
assert all(set(r)=={'index','member','n','density','port_coverage','preplaced_fraction','locked_fraction','boundary_fraction'} for r in stats)
records=[dict(index=0,member=['old',0],n=30,density=.1,port_coverage=.2),
    dict(index=1,member=['old',1],n=30,density=.3,port_coverage=.8),
    dict(index=20000,member=['seen',0],n=30,density=.01,port_coverage=.2),
    dict(index=32000,member=['new',0],n=30,density=.1,port_coverage=.2),
    dict(index=32001,member=['new',1],n=30,density=.4,port_coverage=.2)]
a=p.select_profiles(records,4000)
assert a==p.select_profiles(list(reversed(records)),4000)
assert a['selected_count']==1 and a['new_members'][0]['index']==32000
assert p.preparation_context()['future_training_parent'] is None
with exclusive(HERE/'runs/preflight/lock_test.lock'):
    try:
        with exclusive(HERE/'runs/preflight/lock_test.lock'):raise AssertionError('Duplicate lock accepted')
    except OSError:pass
try:verify('local_test')
except PermissionError:pass
else:raise AssertionError('Sealed cache accepted')
for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
with zipfile.ZipFile(HERE/'runs/preflight/20261031_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in HERE.glob('*.py'):archive.write(path,path.relative_to(ROOT).as_posix())
    archive.write(Path(__file__),Path(__file__).relative_to(ROOT).as_posix())
save(HERE/'runs/preflight/results.json',dict(status='passed',sources=sources(),same_five_train_indices=ids,
    public_statistics_only=True,deterministic_selection=True,prior32000_excluded=True,
    independent_of_E87_completion=True,single_instance_lock_checked=True,
    numeric_threads_per_worker=1,configured_workers=4,sealed_cache_rejected=True,
    full_pool_scanned=False,training_started=False,seconds=time.perf_counter()-begin,test_read=False))
print('Five-case data-only preflight passed; no full scan or training performed')
