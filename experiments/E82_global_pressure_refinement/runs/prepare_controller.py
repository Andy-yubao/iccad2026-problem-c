"""Create experiment-local launch and accounting helpers, not training sources."""
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];OLD=ROOT/'experiments/E81_multisource_refinement'
for name in ('launch_controller.ps1','record_controller.py'):
 t=(OLD/'runs'/name).read_text(encoding='utf-8').replace('E81_multisource_refinement','E82_global_pressure_refinement').replace('20261029_e42_multisource_seed0','20261029_e81_pressure_seed0')
 if name.endswith('.py'):
  t=t.replace("study=read(HERE/'study.json')","study=read(ROOT/'experiments/E81_multisource_refinement/study.json')")
  t=t.replace("save(HERE/'study.json',study)","save(ROOT/'experiments/E81_multisource_refinement/study.json',study)")
  t=t.replace("'phase_candidate':1","'phase_candidate':2").replace('7543025','7545329').replace('1713086','1715390').replace('241547','2304')
  t=t.replace('Fresh all-F AdamW; frozen placers, original four LR rule','Fresh AdamW only2304 pressure weights; all E81 source weights frozen, same four LR rule')
  t=t.replace("study.update(status=state['status'],current_stage=state['stage'],","study.update(status=state['status'],current_stage=state['stage'],current_experiment=HERE.name,")
 (HERE/'runs'/name).write_text(t,encoding='utf-8')
