"""E41 rectangular-depth control; set library budgets before importing numeric runtimes."""
import os

THREADS = int(os.environ.get('E41_THREADS', '4'))
if THREADS not in (2, 4):
    raise ValueError('E41_THREADS must be 4 for training or 2 per evaluation worker')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = str(THREADS)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
