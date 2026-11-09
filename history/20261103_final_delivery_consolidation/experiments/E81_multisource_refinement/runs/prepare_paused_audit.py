"""Record explicit user pause and preregister the read-only audit scope."""
import json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
OUT=HERE/'runs/20261029_placer_corrector_audit';OUT.mkdir(parents=True,exist_ok=True)
now=datetime.now(timezone.utc).isoformat()
plan={'role':'User-requested comprehensive read-only audit, not a training experiment or candidate',
 'authorization':'User explicitly pauses experiments and asks to examine whether poor placer outputs limit correction; require four-stage comparisons',
 'recorded_utc':now,'train_updates':0,'solver_calls':0,'new_terminal_Q_evaluations':0,'test_read':False,
 'sources':['E42 best51','E74 best42','E79 saved results','E80 saved results','E81 best26 and saved results','E82 best12 and saved results'],
 'inference_scope':'Only frozen E42/E74 public-input placers on existing dev2000, to recover missing source coordinates and frame/geometry evidence. Corrected/postprocessed positions and Q reused, no solver or selection.',
 'figures':{'fixed100':'Exactly existing seed20261025 sample from dev_monitor500','targeted_cases':[164,443,329,219,21],
 'stages':['reference before public mean-area transform','frozen placer output','correction output','saved postprocessed output'],
 'models':['E81','E82'],'shared_axes':True,'rating_controls':False},
 'questions':['Does worse raw placer geometry predict higher residual after the same corrector?',
 'Do same-case P42/P74 source differences persist under frozen shared F (E79/E80) and jointly trained F81/F82?',
 'Does a source predicted frame have less area than the sum of mandatory public block areas, creating a necessary infeasibility certificate?',
 'How much improvement remains after correction and after postprocessing; which saved failures reverse raw geometry gains?'],
 'analysis':'Source-quality quintiles using original dev2000 base overlap; separately original G. Paired same-member comparisons, Spearman/partial rank associations adjusting n, pin coverage and graph density; report limitations, no causal or universal irreparability claim.',
 'constraints':['Do not change frozen experiment sources or historical outputs','Do not choose checkpoints byQ','No test or new training','No new corrupted-source sweep or solver evaluation during pause'],
 'max_numerical_seconds':1200,'counterevidence':'A poor source may have more room to improve; case difficulty confounds source and corrected quality. Own-frame restriction and source-specific training exposure must be separated from model capacity.'}
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ('E81_multisource_refinement','E82_global_pressure_refinement'):
    path=ROOT/'experiments'/name/'runs/STOP'
    if not path.exists():path.write_text('User pause for comprehensive read-only audit at '+now+'\n',encoding='utf-8')
p=HERE/'study.json';study=json.loads(p.read_text(encoding='utf-8'))
study.update(status='paused_by_user_for_comprehensive_audit',current_stage='read_only_audit',automation_status='PAUSED')
study['user_pause']={'utc':now,'instruction':'接下来暂停实验。我们需要做一次全面核查。','training_resume_requires_new_explicit_authorization':True,'remaining_candidate_slots_not_consumed':1,'audit':OUT.relative_to(ROOT).as_posix()}
p.write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Research paused; read-only audit registered. Existing charges and checkpoint states unchanged.')
