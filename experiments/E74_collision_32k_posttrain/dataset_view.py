"""Frozen nested32k view; original train/dev monitoring members stay unchanged."""
from datasets.expanded_reference import ExpandedReferenceLayouts,registry,REGISTRY
TRAIN_SIZE=32000
class ReferenceLayouts(ExpandedReferenceLayouts):
    def __init__(self,role):super().__init__(role,train_size=TRAIN_SIZE)
