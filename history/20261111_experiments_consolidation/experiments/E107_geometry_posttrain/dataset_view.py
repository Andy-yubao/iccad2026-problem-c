from datasets.expanded_reference import ExpandedReferenceLayouts, REGISTRY, registry

class ReferenceLayouts(ExpandedReferenceLayouts):
    def __init__(self, role):
        super().__init__(role, train_size=32000)
