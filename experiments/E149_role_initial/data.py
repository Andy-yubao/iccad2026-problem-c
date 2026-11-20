"""Reuse original checked tensors, then attach only versioned public-role extras."""
import json
from .base_data import cache_identity,make_row,register_resident,records,REGISTRY,collate
from .base_data import load_rows as base_load
from .role_cache import attach
from .io import HERE
def load_rows(role,indices=None,halted=lambda:False,beat=lambda **kw:None):
    rows=base_load(role,indices,halted,beat)
    if rows is None:return None
    return attach(rows,role,indices,json.loads((HERE/'plan.json').read_text(encoding='utf-8')),halted,beat)
