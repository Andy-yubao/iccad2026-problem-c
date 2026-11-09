"""Geometry-only audit of saved E45 dev drafts; no solver or dataset calls."""
import json
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def geometry(draft, final):
    draft, final = np.asarray(draft), np.asarray(final)
    ds = (draft[:, :2]+draft[:, 2:]).max(0)-draft[:, :2].min(0)
    fs = (final[:, :2]+final[:, 2:]).max(0)-final[:, :2].min(0)
    dc, fc = draft[:, :2]+draft[:, 2:]/2, final[:, :2]+final[:, 2:]/2
    dc, fc = dc-dc.mean(0), fc-fc.mean(0)
    slope = (dc*fc).sum(0)/np.maximum((dc*dc).sum(0), 1e-12)
    return dict(area_ratio=float(fs.prod()/ds.prod()), width_ratio=float(fs[0]/ds[0]),
                height_ratio=float(fs[1]/ds[1]), relative_distortion=float(np.sqrt(((fc-dc)**2).sum()/max((dc*dc).sum(), 1e-12))),
                affine_x=float(slope[0]), affine_y=float(slope[1]),
                affine_residual=float(np.sqrt(((fc-dc*slope)**2).sum()/max((dc*dc).sum(), 1e-12))))


def main():
    rows, seen = [], set()
    for tag in ('20261023_paired100_v6', '20261023_native_quality_gate5'):
        path = HERE.parent/'E45_draft_fidelity/runs'/tag/'results_local.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        for r in state['records']:
            if r['index'] in seen: continue
            seen.add(r['index']); b = r['variants']['control']; m = b['metrics']
            rows.append(dict(index=r['index'], n=r['n'], Q=m['cost'],
                area_gap=m['area_gap'], hpwl_gap=m['hpwl_gap'], boundary=m['boundary_violations'], grouping=m['grouping_violations'],
                **geometry(r['draft'], b['positions'])))
    folder=HERE/'runs/saved_geometry_audit'; folder.mkdir(exist_ok=True)
    with (folder/'cases.csv').open('w',newline='',encoding='utf-8-sig') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'count':len(rows),'expanded_over_10pct':sum(r['area_ratio']>1.1 for r in rows),
        'area_ratio_mean':float(np.mean([r['area_ratio'] for r in rows])),
        'area_ratio_median':float(np.median([r['area_ratio'] for r in rows])),
        'largest_expansion':sorted(rows,key=lambda r:r['area_ratio'],reverse=True)[:10],
        'gate5':[r for r in rows if r['index'] in (50,368,316,443,164)], 'new_solver_calls':0,'test_read':False}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
