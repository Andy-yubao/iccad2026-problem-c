"""At most five counterexamples; raw numerical guard residuals, no training."""
import argparse,json,time
from pathlib import Path
from .io import HERE,HISTORY,np,save,sha
from .probe import decode_teacher
from .certificate import extract
from .joint import joint_step
from .projector import cleanup,separated_snap
from .backend.frozen.packing import prepare_problem
from .backend.frozen.solver import hard_valid
from .backend.frozen.global_refine import topology
from datasets.worksets import LayoutReader,workset_manifest

def residual(x,p):
    if x is None:return None
    i,j=np.triu_indices(p.n,1);o=np.minimum(x[i,:2]+x[i,2:],x[j,:2]+x[j,2:])-np.maximum(x[i,:2],x[j,:2])
    free=~(p.fixed|p.preplaced);r=x[free,2:].prod(1)/p.areas[free]
    return {'strict_valid':hard_valid(x,p),'minimum_xy':float(x[:,:2].min()),
      'area_ratio_min':float(r.min()),'area_ratio_max':float(r.max()),
      'area_bad_count':int(((r<.99)|(r>1.01)).sum()),'overlap_pair_count':int((o>1e-6).all(1).sum()),
      'max_overlap_min_axis':float(o.min(1).max())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    begin=time.perf_counter();args.output.mkdir(parents=True,exist_ok=False)
    teacher_path=HISTORY/'20261018_robust_dev100_seed0/cases.jsonl'
    teachers={r['case_id']:r for r in map(json.loads,teacher_path.read_text(encoding='utf-8').splitlines())}
    reader=LayoutReader(workset_manifest(),'dev_quality');rows=[]
    for index in [1,2,3,4,7]:
        raw=reader.case(index);p=prepare_problem(*raw.solver_inputs)
        teacher=np.asarray(teachers[raw.case_id]['variants']['conic_robust']['final']['positions'])
        cert=extract(teacher,p);seed=decode_teacher(teacher,p,'lower',.02,20261018+index,cert)
        x,stats=joint_step(seed,p,cert['edges'],cert['contacts'],.2,certificate=cert)
        snap,sstats=(None,{}) if x is None else separated_snap(cleanup(x,p),p,cert['edges'],.15,certificate=cert)
        row={'index':index,'case_id':raw.case_id,'native':residual(x,p),'snap':residual(snap,p),
          'stats':stats,'snap_stats':sstats,'teacher':residual(teacher,p)}
        if x is not None:row['native_positions']=x.tolist()
        rows.append(row);print(index,row['native'],row['snap'],flush=True)
    save(args.output/'results.json',{'elapsed_seconds':time.perf_counter()-begin,'rows':rows,
      'source_sha256':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'joint.py',HERE/'projector.py',Path(__file__)]}})

if __name__=='__main__':main()
