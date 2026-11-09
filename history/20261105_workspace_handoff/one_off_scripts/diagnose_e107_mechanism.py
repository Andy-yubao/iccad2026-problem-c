import sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()))
import json,torch
from experiments.E107_geometry_posttrain.io import save
from experiments.E107_geometry_posttrain.train import RUN,seed_all,restore_rng
from experiments.E107_geometry_posttrain.model import GeometryModel
from experiments.E107_geometry_posttrain.runtime import parent_model
from experiments.E107_geometry_posttrain.augmentation import axis_swap
from experiments.E107_geometry_posttrain.batches import inputs_only,decode
from experiments.E107_geometry_posttrain.loss import per_case
from experiments.E107_geometry_posttrain.loss_config import COEFFICIENTS,SOFT_TOLERANCES
from experiments.E107_geometry_posttrain.simple_loss import components
seed_all(0)
s=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False);f=torch.load(RUN/'failure.pt',map_location='cpu',weights_only=False)
m=GeometryModel(**s['model_config']).cuda();m.load_state_dict(s['model']);m.eval();parent=parent_model()
b=axis_swap({k:v.cuda() for k,v in f['batch'].items()});pub=inputs_only(b);i=0
with torch.no_grad():
    anchor=parent(pub);clean=m(pub)
    base=per_case(anchor,b,anchor,anchor);current=per_case(clean,b,anchor,clean)
    restore_rng(f['rng']);parent.train();pstoch=parent(pub);parent.eval()
    restore_rng(f['rng']);m.train();cstoch=m(pub);m.eval()
    pdrop=per_case(pstoch,b,anchor,anchor);cdrop=per_case(cstoch,b,anchor,clean)
    p=decode(clean,b);q=decode(anchor,b);n=int(b['valid'][i].sum());area=b['area'][i,:n]
    displacement=torch.linalg.vector_norm((p[i,:n,:2]+p[i,:n,2:]/2)-(q[i,:n,:2]+q[i,:n,2:]/2),dim=-1)/area.sqrt()
    aspect=(torch.log(p[i,:n,2]/p[i,:n,3])-torch.log(q[i,:n,2]/q[i,:n,3])).abs()
    roles={'parent_clean':base,'student_clean':current,'parent_paired_dropout':pdrop,'student_paired_dropout':cdrop}
    result=dict(member=f['members'][i],n=n,accepted_updates=s['progress']['step'],parameter_updates=0,test_read=False,
        replay_scope='Read-only failure-batch diagnosis. Deterministic clean outputs are exact for saved weights; paired Dropout uses one common saved post-failure RNG draw, not the original rejected draw.',
        recorded_rejected_loss=float(f['guard']['per_case_total'][i]),
        cases={name:dict(raw={k:float(v[i]) for k,v in vals.items()},weighted={k:float(vals[k][i])*w for k,w in COEFFICIENTS.items()}) for name,vals in roles.items()},
        clean_drift=dict(center_median=float(displacement.median()),center_mean=float(displacement.mean()),center_p95=float(torch.quantile(displacement,.95)),center_max=float(displacement.max()),log_aspect_mean=float(aspect.mean()),log_aspect_max=float(aspect.max())),
        interpolation=[])
    # One intermediate output point on the same failed case, not a new candidate or update.
    mid=anchor+.5*(clean-anchor);middle=per_case(mid,b,anchor,mid)
    result['interpolation']=[dict(alpha=a,total=float(v['total'][i]),area_excess=float(v['area_excess'][i]),tail=float(v['tail'][i]),whitespace=float(v['whitespace'][i]),N10=float(v['N10'][i]),soft_barrier=float(v['soft_barrier'][i]),trust=float(v['trust'][i])) for a,v in [(0.,base),(.5,middle),(1.,current)]]
    for name,vals in result['cases'].items():
        raw=vals['raw'];praw=result['cases']['parent_clean']['raw']
        vals['soft_ratios']={k:dict(parent=praw[k],value=raw[k],bound=max(praw[k],t),weighted_penalty=2*max(raw[k]/max(praw[k],t)-1,0)**2) for k,t in zip(('boundary','grouping','mib'),SOFT_TOLERANCES)}
save(RUN/'failure_mechanism_diagnosis.json',result)
print(json.dumps(dict(member=result['member'],clean_drift=result['clean_drift'],interpolation=result['interpolation'],cases={name:dict(total=x['raw']['total'],weighted=x['weighted'],soft_ratios=x['soft_ratios']) for name,x in result['cases'].items()}),ensure_ascii=False,indent=2),flush=True)
