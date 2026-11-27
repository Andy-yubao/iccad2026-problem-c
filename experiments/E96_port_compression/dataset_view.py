"""Frozen v5 16k members, read through verified nested-v6 prefix cache."""
from datasets.expanded_reference import ExpandedReferenceLayouts,registry,REGISTRY
TRAIN_SIZE=16000
class ReferenceLayouts(ExpandedReferenceLayouts):
    def __init__(self,role):super().__init__(role,train_size=TRAIN_SIZE)
