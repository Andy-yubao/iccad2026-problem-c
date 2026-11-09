"""Explicitly copy frozen numerical sources and package three completed models."""
from pathlib import Path
import shutil
import torch
from .common import ROOT, OWNER, read, save, sha


def main():
    if (OWNER / 'copy_provenance.json').exists():
        raise RuntimeError('Independent sources already staged; review rather than overwrite')
    previous = ROOT / 'experiments/E103_offline_loss_alignment'
    records = []
    def copy(a, b):
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, b)
        assert sha(a) == sha(b)
        records.append(dict(source=a.relative_to(ROOT).as_posix(), copy=b.relative_to(ROOT).as_posix(),
                            sha256=sha(a), adaptation=None))
    for name in ('features.py', 'batches.py', 'cache.py'):
        copy(previous / name, OWNER / name)
    for source in sorted((previous / 'postprocessor').rglob('*.py')):
        if '__pycache__' not in source.parts:
            copy(source, OWNER / 'postprocessor' / source.relative_to(previous / 'postprocessor'))
    copy(previous / 'postprocessor/budget.json', OWNER / 'postprocessor/budget.json')
    copy(previous / 'sources/e98_generalization/model.py', OWNER / 'sources/model.py')
    (OWNER / 'sources/__init__.py').write_text('"""Frozen original E98 inference model only."""\n', encoding='utf-8')
    specs = [
        ('enhanced71', previous / 'assets/e98_generalization/best.pt', 71,
         'Original completed E98 generalization baseline; primary neighborhood'),
        ('learned29', previous / 'runs/20261104_learned_v1_lr1e3_seed0/checkpoints/best.pt', 29,
         'Completed fitted-Loss continuation, different overlap/task tradeoff'),
        ('reference075_8', previous / 'runs/20261104_reference075_seed0/checkpoints/best.pt', 8,
         'Completed reference075 continuation; another mature local distribution'),
    ]
    models = []
    for name, origin, epoch, reason in specs:
        old = torch.load(origin, map_location='cpu', weights_only=False)
        model_config = old['model_config']
        path = OWNER / f'assets/{name}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        # Inference-only extraction; no optimizer/RNG state, no parameter modification.
        torch.save(dict(model=old['model'], model_config=model_config), path)
        verify = torch.load(path, map_location='cpu', weights_only=False)
        assert all(torch.equal(v, verify['model'][k]) for k, v in old['model'].items())
        models.append(dict(id=name, origin=origin.relative_to(ROOT).as_posix(), origin_sha256=sha(origin),
                           checkpoint=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                           best_epoch=epoch, model_config=model_config, reason=reason,
                           weight_extraction_only=True, optimizer_updates=0))
    save(OWNER / 'copy_provenance.json', dict(files=records, private_runtime_cross_imports=False,
                                           postprocessor='Exact E98.depth_budget.v1 copy'))
    save(OWNER / 'model_manifest.json', dict(models=models, source_family='Completed E98-compatible models',
                                           old_abandoned_search_models_used=False))


if __name__ == '__main__':
    main()
