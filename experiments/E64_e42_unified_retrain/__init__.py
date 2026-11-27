"""E64 module-relative overlap; set library budgets before numeric runtimes."""
import os

THREADS = int(os.environ.get('E64_THREADS', '4'))
if THREADS not in (1, 2, 4):
    raise ValueError('E64_THREADS must be 4 for training, 2 per solver worker, or 1 per renderer')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = str(THREADS)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
