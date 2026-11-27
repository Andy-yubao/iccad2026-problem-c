"""Independent global neural corrector; no repair algorithm in model forward."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
