"""One fixed value+secant+physical-auxiliary objective, panel-normalized weights."""
import numpy as np
import torch
from .common import OWNER, RUN, read, save


def calibrate(rows):
    """Only rows from fit3000 may set normalization; no Q-driven clipping."""
    assert len(rows) == 3000 and all(r['split'] == 'fit' for r in rows)
    q = torch.cat([r['Q'] for r in rows]).numpy()
    aux = torch.cat([r['auxiliary'] for r in rows]).numpy()
    slopes = np.array([(float(r['Q'][3])-float(r['Q'][4]))/(2*r['pairs'][0]['radius']) for r in rows])
    result = dict(Q_scale=max(float(q.std()), .02),
                  auxiliary_scale=np.maximum(aux.std(0), [.02, .02, .01]).tolist(),
                  slope_scale=max(float(np.sqrt(np.mean(slopes**2))), .05),
                  fit_cases=3000, only_fit_labels=True, no_label_clipping=True)
    save(RUN/'normalization.json', result)
    return result


def role_exp_mean(role):
    panel = read(OWNER/'panel.json')['cases']
    return float(np.mean([np.exp((r['n']-120)/12) for r in panel if r['split'] == role]))


def fitting_loss(prediction, batch, normalization, exp_mean):
    q = prediction['Q'].reshape(-1, 5)
    auxiliary = prediction['auxiliary'].reshape(-1, 5, 3)
    value = ((q-batch['Q']) / normalization['Q_scale']).square().mean(1)
    predicted_slope = (q[:, 3]-q[:, 4])/(2*batch['radius'])
    true_slope = (batch['Q'][:, 3]-batch['Q'][:, 4])/(2*batch['radius'])
    direction = ((predicted_slope-true_slope)/normalization['slope_scale']).square()
    aux_scale = auxiliary.new_tensor(normalization['auxiliary_scale'])
    physical = ((auxiliary-batch['auxiliary'])/aux_scale).square().mean((1, 2))
    weights = .5 + .5*torch.exp((batch['n']-120)/12)/exp_mean
    return dict(total=(weights*(value+direction+.1*physical)).mean(),
                value=(weights*value).mean(), directions=(weights*direction).mean(),
                auxiliary=(weights*physical).mean())
