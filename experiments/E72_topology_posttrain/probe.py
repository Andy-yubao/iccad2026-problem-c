"""Five fixed E62 training cases; no parameter training or test access."""
import torch
from .io import HERE,save
from .train import seed_all
from .model import GeometryModel,decode
from .initialization import load
from .arms import model_config
from .data import load_rows,collate
from .batches import inputs_only
from .loss import per_case
from .topology import details

def main():
    seed_all(0); ids=[1683,10027,5474,9793,1814]
    rows=load_rows('train',ids);g,_=collate(rows,'cuda')
    model=GeometryModel(**model_config('topology')).cuda().eval();load(model,model_config('topology'))
    with torch.no_grad():s=model(inputs_only(g))
    s=s.detach().requires_grad_();v=per_case(s,g,'topology');t,d=details(s,g)
    go=torch.autograd.grad(v['overlap'].sum(),s,retain_graph=True)[0]
    gt=torch.autograd.grad(t.sum(),s,retain_graph=True)[0]
    gg=torch.autograd.grad(v['geometry'].sum(),s,retain_graph=True)[0]
    total=torch.autograd.grad(v['total'].sum(),s)[0]
    report=[]
    for i,row in enumerate(rows):
        pair=d['eligible'][i];active=pair&(d['deficit'][i]>1e-6)
        slack=d['extent'][i]-d['pd'][i].abs()
        overlap=(slack.amin(-1)>1e-6)&pair
        old_axis=slack.argmin(-1);new_axis=d['axis'][i].squeeze(-1)
        wrong=(d['signed'][i]<0)&pair
        opposing=(old_axis!=new_axis)|((d['signed'][i]<0)&(old_axis==new_axis))
        def cos(a,b):return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))
        base=decode(s,g).detach();local=g['area'].sqrt()
        updates={}
        for label,gradient in (('G',gg),('new_total',total)):
            # Output-space direction diagnostic only; does not claim Adam equivalence.
            change=-gradient/gradient.flatten(1).norm(dim=1)[:,None,None].clamp_min(1e-12)*.01
            q=decode(s.detach()+change,g).detach()
            mask=g['valid'][i];p=q[i,mask]
            bbox=float((p[:,:2]+p[:,2:]).amax(0).sub(p[:,:2].amin(0)).prod())
            intersection=(torch.minimum(p[:,None,:2]+p[:,None,2:],p[None,:,:2]+p[None,:,2:])-torch.maximum(p[:,None,:2],p[None,:,:2])).clamp_min(0).prod(-1)
            intersection.fill_diagonal_(0)
            updates[label]={'bbox_normalized':bbox,'intersection_over_total_area':float(intersection.sum()/2/g['area'][i].sum()),'topology':float(details(s.detach()+change,g)[0][i])}
        report.append({'train_index':ids[i],'member':list(row['member']),'n':int(g['valid'][i].sum()),
            'G':float(v['geometry'][i].detach()),'topology':float(t[i].detach()),
            'eligible_unordered_pairs':int(pair.sum()/2),'active_unordered_pairs':int(active.sum()/2),
            'overlap_near_pairs':int(overlap.sum()/2),'min_axis_conflict_on_overlapping_pairs':int((overlap&opposing).sum()/2),
            'wrong_order_near_pairs':int(wrong.sum()/2),'missed_nonoverlap_wrong_order_pairs':int((wrong&~overlap).sum()/2),
            'overlap_topology_gradient_cosine':cos(go[i],gt[i]),'G_topology_gradient_cosine':cos(gg[i],gt[i]),
            'output_space_directions':updates})
    save(HERE/'runs/diagnosis/five_case.json',{'cases':report,'unique_train_cases':5,'parameter_updates':0,'test_read':False,
        'limits':'Mechanism diagnostic on fixed training examples; output steps are not Adam or trained-model evidence.'})
    print(report,flush=True)
if __name__=='__main__':main()
