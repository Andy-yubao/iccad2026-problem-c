"""Register E167 as a new backend run from the verified E165 best67 L0."""

import json
from .io import HERE, ROOT, save, sha, sources
from .loss_config import ARM


def main():
    plan = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
    parent = plan['parent_initial']
    if sha(ROOT / parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('E165 best67 checkpoint digest differs')
    if (HERE / 'runs').exists() or (HERE / 'queue_state.json').exists():
        raise FileExistsError('E167 already has a registered run')
    save(HERE / 'preparation.json', dict(status='prepared', sources=sources()))
    save(HERE / 'authorization.json', dict(
        scope=plan['revision'], training_enabled=True, postprocessing_enabled=False,
        allowed_arms=[ARM], test_read=False, platform='kaggle',
    ))
    save(HERE / 'queue_state.json', dict(status='prepared', completed=[], platform='kaggle'))


if __name__ == '__main__':
    main()
