"""Predeclared capacity controls; parameter count alone is not the conclusion."""
ARMS = {
    'geometry96': dict(width=96, layers=12, rectangle=True, ffn_dropout=.05),
    'geometry160': dict(width=160, layers=12, rectangle=True, ffn_dropout=.05),
    'reference96': dict(width=96, layers=12, rectangle=True, ffn_dropout=.05),
    'reference96_noreg': dict(width=96, layers=12, rectangle=True, ffn_dropout=0.),
    'reference160_noreg': dict(width=160, layers=12, rectangle=True, ffn_dropout=0.),
}


def model_config(arm):
    return dict(ARMS[arm])


def weight_decay(arm):
    return 0. if arm.endswith('_noreg') else 3e-4
