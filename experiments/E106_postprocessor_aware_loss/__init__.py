"""Independent postprocessor-aware learned objective. No automatic training."""
import os

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
