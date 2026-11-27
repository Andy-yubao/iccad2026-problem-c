"""E43 area-loss experiment; set library budgets before numeric runtimes."""
import os

THREADS = int(os.environ.get('E43_THREADS', '4'))
if THREADS not in (2, 4):
    raise ValueError('E43_THREADS must be 4 for training or 2 per evaluation worker')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = str(THREADS)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
