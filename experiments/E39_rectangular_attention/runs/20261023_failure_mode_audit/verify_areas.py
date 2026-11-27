"""Verify original public areas vs reference WH for the exact two monitor panels."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
from datasets.reference_layouts import ReferenceLayouts,mean_area_inputs
from datasets.worksets import FrozenMemberReader
torch.set_num_threads(4);torch.set_num_interop_threads(1)
def main():
    start=time.perf_counter();result={}
    for role in ['dev_monitor500','train_monitor500']:
        ds=ReferenceLayouts(role);members=[r['member'] for r in ds.records]
        rd=FrozenMemberReader(members,ds.parent,identity=ds.registry['manifest_sha256'])
        maxdiff=0.;changed=0;total=0;affected=0;maxratio=0.;original_area_invalid=0;dimension_varied_groups=0;groups_count=0
        exactrows=[]
        for i in sorted(range(len(members)),key=lambda i:members[i]):
            raw=rd.case(i);inp=raw.solver_inputs;new=mean_area_inputs(inp)
            original=np.asarray(inp[1],dtype=float);modified=np.asarray(new[1],dtype=float)
            ref=np.asarray(raw.reference_positions);wh=ref[:,2:].prod(1);c=np.asarray(inp[5]);locked=c[:,:2].any(1)
            relative=np.abs(wh/original-1);maxdiff=max(maxdiff,float(relative.max()))
            original_area_invalid+=int((relative[~locked]>.01).sum())
            delta=np.abs(modified/original-1);ch=(delta>.01)&~locked
            changed+=int(ch.sum());total+=len(original);affected+=int(ch.any());maxratio=max(maxratio,float(delta.max()))
            for g in np.unique(c[:,2]):
                if g<=0:continue
                ids=np.flatnonzero(c[:,2]==g)
                if len(ids)<2:continue
                groups_count+=1
                dimension_varied_groups+=int(np.ptp(ref[ids,2])>1e-4 or np.ptp(ref[ids,3])>1e-4)
            exactrows.append({'index':i,'member':members[i],'changed_area_fraction':float(ch.mean()),'mean_abs_area_change':float(delta[~locked].mean()) if (~locked).any() else 0.})
        result[role]={'count':len(members),'max_reference_area_vs_original_input_relative_error':maxdiff,
            'reference_area_violations_original_task':original_area_invalid,'changed_blocks_gt1pct':changed,'blocks':total,'changed_cases':affected,
            'max_relative_area_change':maxratio,'original_mib_dimensions_varied_groups':dimension_varied_groups,'mib_groups':groups_count,'rows':exactrows}
        print(role+' original area verification complete',flush=True)
    result.update(seconds=time.perf_counter()-start,training=False,solver_calls=0,test_read=False,threads=4,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (HERE/'original_area_verification.json').open('w',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
