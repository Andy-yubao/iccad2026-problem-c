"""Preserve pre-training launch faults; no checkpoint/cursor/optimizer was reset."""
from pathlib import Path
import json
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
attempts=[]
for suffix,reason in (('102748','Detached startup returned a PID but created no pipeline state or formal output; both OS locks independently acquired. Exact process exit cause unknown.'),
                      ('103005','Controller executed; child failed before importing training because copied module namespace E81_center_refinement did not exist. Pipeline module names fixed before formal initialization.')):
    path=HERE/f'runs/controller_attempt_20261029_{suffix}.json'
    attempts.append({**read(path),'reason':reason,'formal_updates':0})
output=HERE/'runs/20261029_e42_multisource_seed0'
assert not output.exists(),'Run before the first actual formal initialization'
value={'recorded_utc':datetime.now(timezone.utc).isoformat(),'attempts':attempts,
 'formal_output_absent_before_repair':True,'checkpoint_or_optimizer_reset':False,'training_budget_reset':False,
 'source_status':'Pipeline namespace and preflight CLI resolution check corrected before any formal config/checkpoint/update. Full five-case model/Loss/CUDA/resume check rerun on final sources.',
 'separate_startup_cost':'Startup attempts recorded separately; no training initialization occurred. The new formal run retains original2h/200epoch cap and all historical global costs.',
 'test_read':False}
(HERE/'runs/startup_attempts.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Pre-training startup failures preserved; no formal state reset.')
