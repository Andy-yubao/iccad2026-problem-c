"""Architecture configurations are frozen only after the E38 capacity comparison."""
import os

# Capacity is an explicit launch identity, never inferred from a test result.
WIDTH = int(os.environ.get('E39_WIDTH', '96'))
DEPTH = int(os.environ.get('E39_DEPTH', '4'))
if WIDTH not in (96, 160) or DEPTH not in (4, 8, 12):
    raise ValueError('E39 trunk must come from the declared capacity family')
ARMS = {
    'geometry_control': {'width': WIDTH, 'layers': DEPTH},
    'rectangle': {'width': WIDTH, 'layers': DEPTH, 'rectangle': True},
    'rectangle_no_gap': {'width': WIDTH, 'layers': DEPTH, 'rectangle': True, 'use_gap': False},
}


def model_config(arm):
    return dict(ARMS[arm])
