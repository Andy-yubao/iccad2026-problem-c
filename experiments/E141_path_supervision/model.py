"""Original E119, unchanged; no latent replanning or corruption branch."""
from .e119_model import GeometryModel as E119


def GeometryModel(architecture='e119', **configuration):
    if architecture != 'e119':
        raise ValueError(architecture)
    return E119(**configuration)
