"""P0 preparation; importing this package never launches a run."""
import os

if os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
    raise RuntimeError('This experiment requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
