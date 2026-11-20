"""Register one original-budget postprocessing run for the fixed16-layer composition."""
from pathlib import Path
import hashlib
import json

OUT=Path(__file__).resolve().parent
RUN=OUT.parents[1]
EXPERIMENT=RUN.parents[1]
ROOT=EXPERIMENT.parents[1]
ARM='e148_s0x2_s1x3_s2x3'
PREFIX='experiments.E148_shared_functional_shape'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if (OUT/'plan.json').exists():raise RuntimeError('Already prepared; inspect existing run')
    source=EXPERIMENT/'evaluation/run.py'
    text=source.read_text(encoding='utf-8')
    text=text.replace('from ..io import',f'from {PREFIX}.io import')
    text=text.replace('from ..model import',f'from {PREFIX}.model import')
    text=text.replace('from ..quality_reporting import',f'from {PREFIX}.quality_reporting import')
    text=text.replace('from ..features import',f'from {PREFIX}.features import')
    text=text.replace('from ..batches import',f'from {PREFIX}.batches import')
    text=text.replace('from .solver import',f'from {PREFIX}.evaluation.solver import')
    text=text.replace('from .repair.canonical_native import',f'from {PREFIX}.evaluation.repair.canonical_native import')
    text=text.replace('from ..loss_config import ARMS',f"ARMS=('{ARM}',)")
    text=text.replace('from ..io import sources',f'from {PREFIX}.io import sources')
    text=text.replace('from ..run_queue import verify',f'from {PREFIX}.run_queue import verify')
    changes={
        "self.model.load_state_dict(state['model'],strict=True)":"self.model.load_state_dict(state['model'],strict=True)\n        self.model.requires_grad_(False)\n        self.model.unit_order=(0,0,1,1,1,2,2,2)",
        "        verify()":"        verify()",
        "    verify()":"    verify()\n    registration=json.loads((OWNER/'registration.json').read_text(encoding='utf-8'))\n    if sha(Path(__file__))!=registration['runner_sha256']:raise RuntimeError('Registered evaluation runner changed')",
        "auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))":"auth=json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))",
        "if not auth.get('training_enabled') or arm not in auth.get('allowed_arms',[]):raise PermissionError('Preparation only')":"if not auth.get('postprocessing_enabled') or arm not in auth.get('allowed_arms',[]):raise PermissionError('Missing postprocessing authorization')",
        "run=HERE/'runs'/f'{arm}_seed0';checkpoint=run/'checkpoints/best.pt'":"run=OWNER.parents[1];checkpoint=run/'checkpoints/best.pt'",
        "out=run/'evaluation_no_mib_dev500'":"out=OWNER/'evaluation_no_mib_dev500'",
        "source_identity=sources()":"source_identity=sources()\n    source_identity[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))",
        "            predictor=None":"            predictor=None\n            saved_states=torch.load(OWNER.parent/'states_14_15_16.pt',map_location='cpu',weights_only=False)\n            if saved_states['checkpoint_sha256']!=frozen['checkpoint_sha256'] or saved_states['members']!=plan['members']:raise ValueError('Saved16-layer draft identity mismatch')",
        "                    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin":"                    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin\n                    if i==0:\n                        if predictor.states.shape[0]!=16:raise ValueError('Expected16 real geometry readouts')\n                        torch.testing.assert_close(predictor.states[-1,0,:int(inputs[0])].cpu(),saved_states['states'][i,2,:int(inputs[0])],rtol=1e-4,atol=2e-5)\n                        save(out/'composition_check.json',dict(readouts=16,unit_order=list(predictor.model.unit_order),matches_registered_batched_draft=True,rtol=1e-4,atol=2e-5))\n                        torch.cuda.synchronize();begin=time.perf_counter();draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin",
        "logpath=HERE/'background/quality.log'":"logpath=OWNER/'background/quality.log'",
        "            save(out/'results.json',result)":"            result.update(unit_order=[0,0,1,1,1,2,2,2],geometry_layers=16,original_checkpoint_unchanged=True,depth_selection='User-selected after fixed dev500 Loss diagnostics; no Q-based checkpoint or depth reselection',optimizer_updates=0)\n            save(out/'results.json',result)",
    }
    for old,new in changes.items():
        if old==new:continue
        if text.count(old)!=1:raise RuntimeError('Ambiguous source mapping: '+old)
        text=text.replace(old,new)
    assert 'from ..' not in text and 'from .solver' not in text
    runner=OUT/'run.py';runner.write_text(text,encoding='utf-8')
    analysis=read(OUT.parent/'results.json')
    assert analysis['status']=='complete' and analysis['count']==500 and analysis['unit_order']==[0,0,1,1,1,2,2,2]
    original=read(EXPERIMENT/'evaluation/frozen_e148_shared_functional_shape.json')
    assert original['checkpoint_sha256']==analysis['checkpoint_sha256']==digest(ROOT/original['checkpoint'])
    frozen=dict(original,arm=ARM,unit_order=[0,0,1,1,1,2,2,2],geometry_layers=16,
        common_final_dev_loss=analysis['stage_losses']['S2_intermediate']['16']['original_total'],
        own_new_final_dev_loss=analysis['final_full_loss'],
        depth_selection='User-authorized16-layer composition after dev500 Loss studies; checkpoint fixed original own-Loss-best46, no Q reselection',
        common_final_train_loss=None,original_source_identity=digest(source),runner_sha256=digest(runner))
    save(OUT/f'frozen_{ARM}.json',frozen)
    plan=read(EXPERIMENT/'evaluation/plan.json')
    plan.update(revision='E148.extra_S1_S2.postprocess.dev500.v1',authorization='User explicitly authorized one background postprocessing run for S0x2,S1x3,S2x3 on original fixed dev500',
        models={ARM:frozen},unit_order=[0,0,1,1,1,2,2,2],geometry_layers=16,
        source_evaluation_sha256=digest(source),runner_sha256=digest(runner),training_authorized=False,
        inference='Recompute full16-layer chain per case for complete latency; saved16-layer states used only for first-case consistency check',
        depth_selection=frozen['depth_selection'],automatic_retry=False)
    save(OUT/'plan.json',plan)
    save(OUT/'authorization.json',dict(postprocessing_enabled=True,training_enabled=False,allowed_arms=[ARM],test_read=False,automatic_retry=False,note=plan['authorization']))
    save(OUT/'registration.json',dict(runner_sha256=digest(runner),original_evaluation_sha256=digest(source),checkpoint_sha256=analysis['checkpoint_sha256'],unit_order=analysis['unit_order']))
    (OUT/'budget.json').write_bytes((EXPERIMENT/'evaluation/budget.json').read_bytes())
    (OUT/'background').mkdir(exist_ok=True)
    print('Prepared one frozen16-layer original-budget dev500 postprocessing run.')

if __name__=='__main__':main()
