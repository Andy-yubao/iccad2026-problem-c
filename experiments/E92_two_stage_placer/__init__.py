"""E92 independent two-stage placer; fixed FP32 and library budgets."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
