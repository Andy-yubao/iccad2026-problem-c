"""Only one final public-priority objective; no auxiliary or terminal Q."""
from .loss import configuration as objective_configuration
from .control import ARMS


def configuration(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    value = objective_configuration()
    value.update(arm=arm, layer_weights={'final': 1.},
                 selection='unique own final-Loss dev500 best; no Q or postprocessing')
    return value
