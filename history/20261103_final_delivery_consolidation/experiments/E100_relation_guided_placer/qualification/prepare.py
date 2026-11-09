"""Register one frozen-best dev500 evaluation; no training or preflight suite."""
import json
from pathlib import Path
import torch
from ..io import ROOT,HERE,sha,save

OWNER=Path(__file__).resolve().parent

def main():
    run=HERE/'runs/20261103_relation_guided_seed0'
    result=json.loads((run/'results.json').read_text(encoding='utf-8'))
    progress=json.loads((run/'progress.json').read_text(encoding='utf-8'))
    saved=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    if result['error'] is not None or not progress['finished']:
        raise RuntimeError('Training did not finish normally')
    if saved['progress']['epoch']!=result['best_epoch'] or saved['progress']['best']!=result['best_fixed_loss']:
        raise ValueError('Frozen checkpoint and training best disagree')
    drift=[name for name,digest in saved['identity']['sources'].items() if sha(ROOT/name)!=digest]
    if drift:raise ValueError('Training sources drifted: '+repr(drift))
    manifest=[]
    parent=ROOT/'experiments/E98_loss_weight_learning/qualification'
    for local in sorted((OWNER/'repair').rglob('*.py')):
        original=parent/local.relative_to(OWNER)
        if sha(original)!=sha(local):raise ValueError('Repair copy changed')
        manifest.append(dict(source=original.relative_to(ROOT).as_posix(),copy=local.relative_to(ROOT).as_posix(),sha256=sha(local)))
    oldplan=json.loads((parent/'plan.json').read_text(encoding='utf-8'))
    panel=json.loads((run/'panels.json').read_text(encoding='utf-8'))
    plan=dict(authorization='2026-11-03 user: 训练好了，请你检验。Frozen best evaluation only, no new training.',
        checkpoint=(run/'checkpoints/best.pt').relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(run/'checkpoints/best.pt'),best_epoch=result['best_epoch'],
        panel='frozen v5 dev_monitor500, original order, no test payloads',
        members=panel['dev_monitor_members'],output=(run/'evaluation_dev500').relative_to(ROOT).as_posix(),
        baseline_output=oldplan['output'],baseline_checkpoint=oldplan['checkpoint'],
        baseline_checkpoint_sha256=oldplan['checkpoint_sha256'],
        full_repair=oldplan['full_repair'],copy_manifest=manifest,
        cpu_threads=1,gpu_batch=1,maximum_seconds=3600,
        diagnostic='same forward initial vs guided coordinates; all/near-reference/initial-overlap/connected pair accuracy versus coordinate-only heuristic; passive diagnosis, no independently retrained ablation',
        nearest_definition='reference rectangle Euclidean edge gap <= .25 mean sqrt module area',
        overlap='both exact covered-area union and training pair-sum overlap; do not mix definitions',
        geometry_gallery='first fixed100: reference / E98 best58 / E100 best61 guided / E100 full repair; shared IDs and scale; no scoring controls or automatic opening',
        engine='copied E98 captured public-input inference adapted to E100 dict output; same copied full repair, no new coefficient or checkpoint selection',
        checks='direct formal evaluation; source/checkpoint/panel binding and runtime finite/legality gates only; no resume tests or preventive suite',
        test_read=False,parameter_updates=0,new_candidates=0)
    save(OWNER/'plan.json',plan)
    save(run/'training_audit.json',dict(status='passed',training=result,
         training_source_drift=drift,checkpoint_epoch=saved['progress']['epoch'],
         checkpoint_step=saved['progress']['step'],checkpoint_sha256=plan['checkpoint_sha256'],
         best_train_monitor=progress['history'][result['best_epoch']]['train_monitor'],
         best_dev_monitor=progress['history'][result['best_epoch']]['dev_monitor'],
         test_read=False,resume_tests=0))
    print(json.dumps({'best_epoch':result['best_epoch'],'output':plan['output'],'source_drift':drift}))

if __name__=='__main__':main()
