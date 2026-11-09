"""E98: fixed rich-loss 96x12 baseline before any train-only weight learning."""
import os

# Must precede the first CUDA context, including module-entry preflight runs.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
