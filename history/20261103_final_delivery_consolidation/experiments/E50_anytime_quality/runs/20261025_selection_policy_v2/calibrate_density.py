"""Adaptive calibration of a public packing-density regime; no case ID branches."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,numpy as np,math
HERE=Path(__file__).resolve().parent
def main():
    data=[]
    for f in (HERE.parent/'20261024_frozen100_parallel_v1/cases').glob('*.json'):
        r=json.loads(f.read_text());cs=r['solver']['candidates'];ns=max(1,cs[0]['metrics']['max_possible_violations'])
        m=np.array([[c['area'],c['wire']] for c in cs]);s=m.max(0)
        t=np.array([2*math.exp(c['public_quality_cost']-2*sum(c['soft_violations'])/ns)-1 for c in cs])
        co=np.linalg.lstsq(m/s,t,rcond=None)[0]/s;data.append((r,co,ns))
    result=[]
    for threshold in(1.01,1.015,1.02,1.025,1.03):
        q=[];reg=[];density=[]
        for r,co,ns in data:
            cs=r['solver']['candidates'];b=cs[r['solver']['baseline_id']];dense=b['area']*co[0]<=threshold
            am,wm,dw=(1.02,.98,.003) if dense else(1.,1.,0.)
            def score(c):return (1+.5*max(c['area']*co[0]/am-1,0)+.5*max(c['wire']*co[1]/wm-1,0))*math.exp(2*sum(c['soft_violations'])/ns)+dw*c['distance']['D']
            chosen=min(cs,key=score);q.append(chosen['metrics']['cost']);density.append(dense)
            if q[-1]>r['baseline_Q']+1e-8:reg.append({'index':r['index'],'delta':q[-1]-r['baseline_Q'],'dense':bool(dense),'base_density':float(b['area']*co[0])})
        weights=np.exp((np.array([r['n'] for r,*_ in data])-120)/12)
        result.append({'threshold':threshold,'weighted_Q':float(np.average(q,weights=weights)),'dense_count':int(sum(density)),'regressions':reg})
    (HERE/'density_grid.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
