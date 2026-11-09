"""One predeclared depth experiment; geometry/Loss/data stay unchanged."""
ARMS = {'rectangle': {'width': 96, 'layers': 16, 'rectangle': True}}

def model_config(arm):
    return dict(ARMS[arm])
