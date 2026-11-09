"""Separately frozen evaluator; never imported by the training entry point."""
import os
threads=os.environ.get('E92_EVAL_THREADS','4')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]=threads
os.environ['E74_THREADS']=threads
