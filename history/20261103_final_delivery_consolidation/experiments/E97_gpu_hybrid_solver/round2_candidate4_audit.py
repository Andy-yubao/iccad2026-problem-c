"""Read saved evidence only; public training group-size inventory, no solve."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math,hashlib,datetime
from pathlib import Path
from .train_io.cache import load_rows
from .train_io.io import save

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    root=HERE/'runs/20261102_round2_candidate4_groupfreedom_seed0/evaluation';arms={}
    for name,folder in [('control',HERE/'runs/20261102_round2_current_control/C3/terminal'),('candidate4',root/'terminal')]:
        rows=[read(p) for p in sorted(folder.glob('*.json'))];assert len(rows)==500
        weights=[math.exp((r['n']-120)/12) for r in rows];mass=sum(weights);parts={k:0. for k in ('wire','area','soft','boundary','grouping','mib')}
        for r,w in zip(rows,weights):
            m=r['metrics'];a=.5*max(0,m['hpwl_gap']);b=.5*max(0,m['area_gap']);s=(1+a+b)*math.expm1(2*m['violations_relative'])
            assert abs(1+a+b+s-m['cost'])<1e-9
            for k,v in [('wire',a),('area',b),('soft',s),('boundary',m['boundary_violations']),('grouping',m['grouping_violations']),('mib',m['mib_violations'])]:parts[k]+=w*v/mass
        arms[name]=parts
    manifest=read(root/'gallery/manifest.json')
    assert all(hashlib.sha256((root/'gallery'/r['image']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['cases'])
    raw=[r for p in sorted((root/'raw').glob('*.json')) for r in read(p)];assert len(raw)==2000
    n=sum(r['n'] for r in raw);bad=sum(r['corrected']['diagnostics']['N10'] for r in raw)
    raw_audit=dict(module_coverage_le_10_percent=1-bad/n,global_worst_module=max(r['corrected']['diagnostics']['max_covered_fraction'] for r in raw),mean_case_worst_module=sum(r['corrected']['diagnostics']['max_covered_fraction'] for r in raw)/2000)
    public={'reused_inventory':'round2_candidate1_saved_audit.json; max group10 unchanged'}
    save(HERE/'runs/round2_candidate4_saved_audit.json',dict(status='complete',arms=arms,raw=raw_audit,gallery_SHA_verified=100,public_group_inventory=public,no_new_inference_solver_Q_optimizer=True,test_read=False))
    print(json.dumps(dict(arms=arms,raw=raw_audit,public_group_inventory=public),indent=2))
if __name__=='__main__':main()
