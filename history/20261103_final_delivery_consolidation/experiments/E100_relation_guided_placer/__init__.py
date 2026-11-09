"""E100: relation-guided simultaneous neural placer, independently maintained."""
import os

# Must precede the first CUDA context, including module-entry preflight runs.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
