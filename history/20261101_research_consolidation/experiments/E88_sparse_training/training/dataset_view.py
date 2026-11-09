"""Frozen v5 monitors, with the selected genuine-new sparse train sidecar."""
from datasets.expanded_reference import ExpandedReferenceLayouts,registry,REGISTRY
from .sampler import manifest

class ReferenceLayouts(ExpandedReferenceLayouts):
    def __init__(self,role):
        super().__init__(role,train_size=64000)
        if role=='train':
            full=self.records
            self.records=full[:16000]+[full[r['index']] for r in manifest()['new_members']]
