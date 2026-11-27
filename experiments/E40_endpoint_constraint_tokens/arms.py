"""Plain, parameter-matched and information-preserving token controls."""
import os

WIDTH = int(os.environ.get('E40_WIDTH', '96'))
DEPTH = int(os.environ.get('E40_DEPTH', '4'))
if WIDTH not in (96, 160) or DEPTH not in (4, 8, 12):
    raise ValueError('Select the trunk explicitly after E38')
MATCHED_WIDTH = round(WIDTH*4/3/4)*4
ARMS = {
    'geometry_control': {'width': WIDTH, 'layers': DEPTH},
    'parameter_control': {'width': MATCHED_WIDTH, 'layers': DEPTH},
    'full_tokens': {'width': WIDTH, 'layers': DEPTH, 'mode': 'full'},
    'pooled_tokens': {'width': WIDTH, 'layers': DEPTH, 'mode': 'pooled'},
}


def model_config(arm):
    return dict(ARMS[arm])
