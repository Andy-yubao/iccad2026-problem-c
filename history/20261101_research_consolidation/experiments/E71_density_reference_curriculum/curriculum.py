"""Predeclared mild position-only curriculum; final objective stable by epoch15."""
import math

HIGH = .375
LOW = .25
HOLD_EPOCHS = 5
FINAL_EPOCH = 15
SELECTION_START_EPOCH = 16


def position_weight(epoch):
    if epoch <= HOLD_EPOCHS:
        return HIGH
    if epoch >= FINAL_EPOCH:
        return LOW
    fraction = (float(epoch)-HOLD_EPOCHS)/(FINAL_EPOCH-HOLD_EPOCHS)
    return LOW+(HIGH-LOW)*.5*(1+math.cos(math.pi*fraction))


def configuration():
    return {'term': 'position', 'initial_weight': HIGH, 'final_weight': LOW,
            'hold_epochs': HOLD_EPOCHS, 'cosine_decay_ends_epoch': FINAL_EPOCH,
            'selection_starts_epoch': SELECTION_START_EPOCH,
            'other_weights': 'fixed throughout', 'within_epoch': 'constant',
            'restoration': 'derive current epoch from saved progress; no reset',
            'authorization': '2026-10-28 user requested mild early reference enhancement and a single combined experiment'}
