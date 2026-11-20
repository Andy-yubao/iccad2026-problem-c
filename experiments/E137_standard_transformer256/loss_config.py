"""One final complete objective for backward, monitoring and model selection."""
from .loss import configuration as objective_configuration

ARMS = ('standard256',)


def configuration(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    value = objective_configuration()
    value.update(version='E137.public-hard-priority.final-only.v1', arm=arm,
                 layer_weights={'final': 1.}, training_authorized=True,
                 selection='unique fixed dev500 final-Loss best; no Q selection')
    return value
