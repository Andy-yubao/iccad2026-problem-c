"""Prepare an isolated, user-requested quality evaluation; preserve trained sources."""
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.E133_flow_matching.control import OWNER,read,sha,sources
from experiments.E133_flow_matching.io import save

def main():
    here=Path(__file__).resolve().parent
    frozen=read(OWNER/'evaluation/frozen_model.json')
    assert sources()==frozen['sources']
    assert sha(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
    assert read(OWNER/'runs/fm_seed0/results.json')['status']=='early_stop'
    evidence=here/'original_evidence';evidence.mkdir(exist_ok=False)
    for name in ('queue_state.json','pipeline_budget.json','runs/fm_seed0/results.json','evaluation/frozen_model.json',
                 'launch_logs/20261112_seed0_prepared/evaluation.stderr.log','evaluation/run.py'):
        target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OWNER/name,target)
    source=(OWNER/'evaluation/run.py').read_text(encoding='utf-8')
    source=source.replace('from ..','from experiments.E133_flow_matching.')
    source=source.replace('from .solver','from experiments.E133_flow_matching.evaluation.solver')
    source=source.replace('from .repair','from experiments.E133_flow_matching.evaluation.repair')
    marker='def public_graph(inputs,pad=120):'
    source=source.replace(marker,"EVALUATION_ROOT=Path(__file__).resolve().parent\nEVALUATION_STARTED=time.perf_counter()\nMAX_EVALUATION_SECONDS=1800\n\n"+marker)
    source=source.replace("out=run/'evaluation_no_mib_dev500'","out=EVALUATION_ROOT/'dev500'")
    source=source.replace('dict(**frozen,members=expected','dict(frozen,members=expected')
    source=source.replace('budget_exhausted()', '(time.perf_counter()-EVALUATION_STARTED>=MAX_EVALUATION_SECONDS)')
    source=source.replace("if stopped():raise InterruptedError('User STOP')", "if stopped() or (EVALUATION_ROOT/'STOP').exists():raise InterruptedError('User STOP')")
    source=source.replace('Total8h budget reached; no continuation','New user-requested evaluation1800s limit; original pipeline budget unchanged')
    source=source.replace("from visualization.layer_comparison import measures", "from visualization.layer_comparison import measures\n            from scoring.draft_geometry import evaluate_geometry,repair_movement")
    source=source.replace("record=dict(index=i,member=expected[i]", "geometry,groups,boundaries=evaluate_geometry(draft,inputs,ref,details=False)\n                    geometry.update(repair_movement(draft,final,inputs[1]))\n                    record=dict(draft_quality=geometry,group_quality=groups,boundary_quality=boundaries,index=i,member=expected[i]")
    source=source.replace("sensitivity(predictor.model,out)","(out/'report.md').write_text('# E133 primary dev500 completed; train5 sensitivity follows\\n',encoding='utf-8')\n            sensitivity(predictor.model,out)")
    (here/'evaluate.py').write_text(source,encoding='utf-8')
    save(here/'registration.json',dict(status='registered_before_evaluation',registered_at=datetime.now(timezone.utc).isoformat(),
        user_instruction='训练好了。请你设法评估这个flow matching模型的质量',
        checkpoint=frozen['checkpoint'],checkpoint_sha256=frozen['checkpoint_sha256'],best_epoch=frozen['best_epoch'],
        original_frozen_sources=frozen['sources'],adapter_sha256=sha(here/'evaluate.py'),
        original_failure='Duplicate test_read keyword while constructing frozen record; zero cases evaluated; original source untouched',
        adapter_changes=['dict(existing_mapping,overrides) fixes duplicate field','isolated output directory','new read-only1800s evaluation budget after original8h elapsed',
            'same raw draft geometry diagnostics as historical E119; outside complete solve timing'],
        original_pipeline_budget_unchanged=True,training_restarted=False,parameters_modified=False,
        scope='same fixed dev500,sole frozen best239,Euler8 one fixed member-seeded sample,original E98.depth_budget.v1;train5 Euler4/8/16 diagnostics only',
        test_read=False,plots=False,new_candidates=False,retry=False))
    print(json.dumps(dict(status='prepared',checkpoint_sha256=frozen['checkpoint_sha256'],evaluation_limit_seconds=1800)))
if __name__=='__main__':main()
