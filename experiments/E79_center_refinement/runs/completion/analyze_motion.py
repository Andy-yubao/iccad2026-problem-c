"""Saved center displacement/saturation and same-case repair tradeoffs."""
import json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=ROOT/'experiments/E79_center_refinement'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
current=read(HERE/'runs/20261029_e78_center_seed0/terminal/results.json')['records']
source=read(ROOT/'experiments/E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0/terminal/results.json')['records']
rows=[];all_shift=[]
for a,b in zip(current,source):
    assert a['member']==b['member']
    p=np.asarray(a['draft_positions']);s=np.asarray(b['draft_positions']);local=np.sqrt(s[:,2:].prod(-1))
    assert np.array_equal(p[:,2:],s[:,2:])
    shift=np.abs((p[:,:2]+p[:,2:]/2)-(s[:,:2]+s[:,2:]/2))/local[:,None];all_shift.extend(shift.ravel().tolist())
    rows.append({'index':a['index'],'member':a['member'],'n':a['n'],'max_center_shift_local':float(shift.max()),
        'mean_center_shift_local':float(shift.mean()),'near_cap_axis_fraction':float((shift>=.145).mean()),
        'Q_current':a['metrics']['cost'],'Q_source':b['metrics']['cost'],'Q_delta':a['metrics']['cost']-b['metrics']['cost'],
        'intersection_delta':a['draft_diagnostics']['overlap_area_fraction']-b['draft_diagnostics']['overlap_area_fraction'],
        'rescue_current':a['rescue_used'],'rescue_source':b['rescue_used'],
        'grouping_delta':a['metrics']['grouping_violations']-b['metrics']['grouping_violations']})
values=np.asarray(all_shift)
result={'scope':'Saved500 public drafts/results only; no forward, solver or test read',
    'axis_denominator':'All valid node axes, includes fixed/preplaced axes with zero motion; not claimed movable-only',
    'mean_abs_shift_local':float(values.mean()),'max_abs_shift_local':float(values.max()),
    'p95_abs_shift_local':float(np.percentile(values,95)),'near_cap_axis_fraction':float((values>=.145).mean()),
    'cases_any_near_cap':sum(r['near_cap_axis_fraction']>0 for r in rows),
    'all_shapes_bit_exact':True,'worst_Q_deltas':sorted(rows,key=lambda r:r['Q_delta'],reverse=True)[:10],
    'records':rows,'test_read':False}
(HERE/'runs/completion/motion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print({k:v for k,v in result.items() if k not in ('records','worst_Q_deltas')},flush=True)
print('worst',result['worst_Q_deltas'][:3],flush=True)
