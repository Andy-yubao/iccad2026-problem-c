"""Register E169 S2 from the verified E168 best45 prefix."""

import json
from .io import HERE, ROOT, save, sha, sources
from .loss_config import ARM


def main():
    plan = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
    parent = plan['parent_initial']
    if sha(ROOT / parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('E168 best45 checkpoint digest differs')
    if (HERE / 'runs').exists() or (HERE / 'queue_state.json').exists():
        raise FileExistsError('E169 already has a registered run')
    save(HERE / 'preparation.json', dict(status='prepared', sources=sources()))
    save(HERE / 'authorization.json', dict(
        scope=plan['revision'], training_enabled=True, postprocessing_enabled=False,
        allowed_arms=[ARM], test_read=False, platform='kaggle',
    ))
    save(HERE / 'queue_state.json', dict(status='prepared', completed=[], platform='kaggle'))


if __name__ == '__main__':
    main()
