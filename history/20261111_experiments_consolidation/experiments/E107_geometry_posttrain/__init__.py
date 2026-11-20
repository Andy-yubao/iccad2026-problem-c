"""E107 E105 warm geometry posttrain, reference10%, strict explosion abort."""
import os

# Must precede the first CUDA context, including module-entry preflight runs.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
