"""Independent local copies of the registered E119 and E120 architectures."""
from .e119_model import GeometryModel as E119
from .node12_model import GeometryModel as E120

def GeometryModel(architecture='e119',**configuration):
    if architecture not in ('e119','e120'):raise ValueError(architecture)
    return (E119 if architecture=='e119' else E120)(**configuration)
