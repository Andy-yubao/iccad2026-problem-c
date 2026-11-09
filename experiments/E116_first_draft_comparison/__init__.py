"""E116 isolated first-draft design comparison; preparation only."""
import os

# Set before importing torch or creating any CUDA context in module entry points.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
