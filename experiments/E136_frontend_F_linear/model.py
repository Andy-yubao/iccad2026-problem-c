"""Only the registered E119 architecture; S0 plus the twelve original readouts."""
from .e119_model import GeometryModel as E119

def GeometryModel(architecture='e119', **configuration):
    if architecture != 'e119':
        raise ValueError('E120 research is discontinued: ' + str(architecture))
    return E119(**configuration)
