"""Final E169 FloorSet solver."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from .service import Solver

__all__ = ['Solver']
