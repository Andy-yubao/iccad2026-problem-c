"""Independent E142-controlled scratch arm; only directional geometry changes."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'experiments/E142_dual_geometry'
ARM='e143_directional_travel';SCOPE='E143.directional_travel.v1.seed0'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def mapped(s):
    for old,new in [('experiments.E142_dual_geometry','experiments.E143_directional_travel'),
        ('E142.dual_geometry.v1.seed0',SCOPE),('e142_dual_geometry',ARM),('E142','E143'),("'e142'","'e143'")]:s=s.replace(old,new)
    return s

def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Already prepared')
    gate=json.loads((PARENT/'preflight.json').read_text());assert gate['status']=='passed'
    records={}
    for name,digest in gate['sources'].items():
        src=ROOT/name;assert sha(src)==digest,name
        if not src.is_relative_to(PARENT):continue
        rel=src.relative_to(PARENT)
        if src.suffix!='.py' and rel.as_posix()!='evaluation/budget.json':continue
        if rel.as_posix() in ('prepare.py','preflight.py'):continue
        dst=HERE/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(mapped(src.read_text(encoding='utf-8')),encoding='utf-8')
        records[rel.as_posix()]=dict(parent=name,sha256=digest)
    plan=json.loads((PARENT/'plan.json').read_text())
    result=PARENT/'runs/e142_dual_geometry_seed0/evaluation_no_mib_dev500/results.json'
    checkpoint=PARENT/'runs/e142_dual_geometry_seed0/checkpoints/best.pt'
    plan.update(experiment='E143 bounded directional travel epsilon0.1',revision=SCOPE,status='authorized_preparation',
        authorization='2026-11-16 user: use epsilon0.1; keep0.25s rigid response, remove long response, replace four clearances with first-violation directional travel capped by boundary and corresponding side; other settings unchanged; prepare and background start.',
        arms={ARM:plan['arms']['e142_dual_geometry']},architecture={'e143':plan['architecture']['e142']},
        parameters={'e143':894867},trainable_parameters={'e143':894867},
        dynamic_module_features=45,module_encoder_input=61,
        initialization='scratch seed0; E142 exact random initialization outside first encoder matrix; that matrix drops old probe100 columns without advancing RNG; no learned weights loaded',
        directional_threshold=.1,directional_caps=['current predicted bbox boundary distance','current width for L/R, current height for T/B'],
        rigid_probe_scales=[.25],directional_normalization='distance/sqrt(public area), then original asinh compression',
        directional_definition='Continuous path prefix where every pair relative penetration <= initial value+0.1; exact first forbidden interval, not endpoint search',
        node_feedback='E142 summaries unchanged except new travel4 replaces escape4 and probe1004 removed;45 dynamic +16 static',
        comparison_control=dict(result=result.relative_to(ROOT).as_posix(),result_sha256=sha(result),
            checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint)),
        comparison='Frozen E142 best55 reused; same64 batch, seed0, data, Loss, schedule and4h/200epoch limits; one seed with adaptive stopping',
        queue='Train -> own-Loss-best freeze -> same fixed dev500/unchanged postprocessing/hard checks -> Q log -> E142 comparison',
        preflight='Analytic travel vs independent dense sampling; caps/first-crossing/shallow obstacles; retained feature equivalence; E139 objective and E142 scratch initialization correspondence; guarded batch64/tail32 and checkpoint replay')
    save(HERE/'plan.json',plan)
    ev=json.loads((PARENT/'evaluation/plan.json').read_text());ev.update(revision=SCOPE+'.dev500',authorization=plan['authorization'],models={});save(HERE/'evaluation/plan.json',ev)
    save(HERE/'provenance.json',dict(parent=PARENT.relative_to(ROOT).as_posix(),parent_preflight_sha256=sha(PARENT/'preflight.json'),copied_originals=records,
        changes=['escape4 replaced by bounded travel epsilon.1','probe1004 removed','encoder input65->61 with corresponding E142 scratch columns retained','runtime identities and comparison control']))
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=[ARM],automatic_evaluation=True,automatic_retry=False,test_read=False,note=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None))
    p=HERE/'run_queue.py';s=p.read_text().replace("'E139_existing_batch32'","'E142_existing_batch64'")
    s=s.replace('Same supervision/Loss/panel; changed architecture and batch32->64, historical control, single seed; not architecture-only causal comparison.',
        'E142-controlled directional feedback change; same batch64, panels, seed0, objective and schedule;256 fewer encoder weights; adaptive single-seed result.')
    s=s.replace('dual geometry versus historical E139','directional travel versus E142');p.write_text(s)
    p=HERE/'model.py';s=p.read_text().replace('nn.Linear(49+16,width)','nn.Linear(49+16,width)')
    needle='        self.layers=nn.ModuleList([CoordinatedBlock(base,width) for base in self.layers])'
    assert needle in s
    s=s.replace(needle,needle+'''
        # Keep the E142 scratch RNG stream and every other parameter unchanged.
        # Removed input coordinates are precisely old probe100[16:20].
        original=self.demand_encoder[0]
        with torch.random.fork_rng(devices=[]):reduced=nn.Linear(45+16,width)
        with torch.no_grad():
            reduced.weight.copy_(original.weight[:,list(range(16))+list(range(20,65))])
            reduced.bias.copy_(original.bias)
        self.demand_encoder[0]=reduced
''');p.write_text(s)
    print('Independent E143 sources prepared; formal start gated on preflight')

if __name__=='__main__':main()
