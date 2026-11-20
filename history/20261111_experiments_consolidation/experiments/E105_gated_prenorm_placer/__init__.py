"""E105 scheme1 gated pre-norm128x16, original E104 rich loss and augmentation."""
import os

# Must precede the first CUDA context, including module-entry preflight runs.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
