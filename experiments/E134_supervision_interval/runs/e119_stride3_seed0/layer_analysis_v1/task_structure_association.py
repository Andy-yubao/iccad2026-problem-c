"""Secondary descriptive checks for pre-registered reference-free structure."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,rankdata
from .topology import HERE,read

def main():
    rows=[read(HERE/f'cases/{i:04d}.json') for i in range(500)]
    losses=np.load(HERE/'losses.npy');names=read(HERE/'manifest.json')['loss_names']
    q=np.asarray([r['original_final_Q'] for r in rows]);n=np.asarray([r['n'] for r in rows]);order=np.lexsort((np.arange(500),-q))
    tail=np.zeros(500,bool);tail[order[:50]]=True;results=[]
    for layer in (0,3,6,9,12):
        overlap=np.asarray([r['layers'][layer]['geometry']['duplicate_coverage'] for r in rows])
        controls=np.column_stack((np.ones(500),rankdata(n),rankdata(overlap),rankdata(losses[:,layer,names.index('reference_weighted')])))
        for metric in ('group_tree_blocked_fraction','group_foreign_bbox_fraction','boundary_outside_fraction','group_contact_bottleneck'):
            values=np.asarray([r['layers'][layer]['geometry'].get(metric) for r in rows],float);good=np.isfinite(values)
            x=values[good];y=q[good];c=controls[good];xr=rankdata(x);yr=rankdata(y)
            xr-=c@np.linalg.lstsq(c,xr,rcond=None)[0];yr-=c@np.linalg.lstsq(c,yr,rcond=None)[0]
            ranks=rankdata(x);positive=tail[good];pos=positive.sum();neg=(~positive).sum()
            auc=(ranks[positive].sum()-pos*(pos+1)/2)/(pos*neg)
            results.append(dict(layer=layer,metric=metric,n=int(good.sum()),spearman_Q=float(spearmanr(x,y).statistic),
                partial_rank_Q=float(np.corrcoef(xr,yr)[0,1]),tail50_AUC=float(auc),
                mean_tail=float(x[positive].mean()),mean_other=float(x[~positive].mean())))
    (HERE/'task_structure_associations.json').write_text(json.dumps(dict(scope='secondary observational checks; higher structure burden treated as higher risk',rows=results),indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in results if r['layer'] in (3,12)]))

if __name__=='__main__':main()
