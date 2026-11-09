"""Five real examples; exact native reconstruction before frozen source swap."""
import json,time
import torch
from .io import HERE,ROOT,sha,save,sources
from .data import load_rows,collate
from .batches import inputs_only
from .model import GeometryModel,decode,extract
from .native_model import GeometryModel as Native
from .frame import predicted_frame
from .loss import per_case
from .train import seed_all

def parents():
    meta=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))['parents']
    for row in meta.values():
        if sha(ROOT/row['checkpoint'])!=row['sha256']:raise ValueError('Parent changed')
    return meta

def main():
    begin=time.perf_counter();seed_all(0);meta=parents()
    source=torch.load(ROOT/meta['correction']['checkpoint'],map_location='cpu',weights_only=False)
    placer=torch.load(ROOT/meta['placer']['checkpoint'],map_location='cpu',weights_only=False)
    for name,digest in source['identity']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source provenance drift: '+name)
    ids=[1683,10027,5474,18140,25001]
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for raw,packed in zip(rows,cached):
        assert raw['member']==packed['member']
        for k,v in raw['graph'].items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,packed['graph'][k]),k
            else:assert v==packed['graph'][k]
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    native=Native(**source['model_config']).eval();native.load_state_dict(source['model'],strict=True)
    reconstructed=extract(GeometryModel(**source['model_config']),source['model'])
    config={**source['model_config'],**placer['model_config']}
    transferred=extract(GeometryModel(**config),source['model'],placer['model'])
    assert sum(p.numel() for p in transferred.parameters())==1537022
    assert sum(p.numel() for name,p in transferred.named_parameters() if not name.startswith('placer.'))==65483
    before={k:v.clone() for k,v in transferred.state_dict().items()}
    records=[]
    with torch.no_grad():
        for index,row in zip(ids,rows):
            g,_=collate([row],'cpu');pub=inputs_only(g)
            expected=native(pub);actual=reconstructed(pub)
            torch.testing.assert_close(actual,expected,rtol=0,atol=2e-7)
            state=transferred(pub);p=decode(state,g);frame=predicted_frame(transferred.placer(pub),pub)
            lo,hi=frame
            assert bool((p[...,:2]>=lo[:,None]-2e-6).all())
            assert bool((p[...,:2]+p[...,2:]<=hi[:,None]+2e-6).all())
            torch.testing.assert_close(p[...,2:].prod(-1),g['area'],rtol=3e-6,atol=2e-8)
            placed=~g['mask'][...,0].bool();locked=~g['mask'][...,2].bool()
            assert torch.equal(p[placed],g['specs'][placed])
            assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
            values=per_case(state,g,'topology');assert all(torch.isfinite(v).all() for v in values.values())
            try:transferred(g);raise AssertionError('Labels accepted')
            except ValueError:pass
            records.append({'train_index':index,'member':list(row['member']),
                'native_max_state_error':float((expected-actual).abs().max()),
                **{k:float(values[k][0]) for k in ('total','common_original_G','true_intersection','source_true_intersection','box_expansion')}})
        g,_=collate(rows,'cuda');pub=inputs_only(g)
        native=native.cuda();reconstructed=reconstructed.cuda();transferred=transferred.cuda()
        torch.testing.assert_close(reconstructed(pub),native(pub),rtol=0,atol=2e-7)
        gpu=transferred(pub).cpu();cpu=transferred.cpu()({k:v.cpu() for k,v in pub.items()})
        torch.testing.assert_close(cpu,gpu,rtol=2e-4,atol=5e-6)
        transferred.train()
        assert all(not m.training for m in transferred.modules())
        assert all(not p.requires_grad for p in transferred.parameters())
        assert all(torch.equal(v,before[k]) for k,v in transferred.state_dict().items())
        assert all(torch.equal(v,placer['model'][k]) for k,v in transferred.placer.state_dict().items())
    # Confirm complete F, not just the last head, is unchanged.
    for new,old in (('first','source.source.corrector.'),('second','source.corrector.'),('third','corrector.')):
        for k,v in getattr(transferred,new).state_dict().items():assert torch.equal(v,source['model'][old+k])
    save(HERE/'runs/preflight.json',{'status':'passed','parents':meta,'sources':sources(),
        'model_config':config,'parameters':1537022,'correction_parameters':65483,'training_updates':0,
        'records':records,'seconds':time.perf_counter()-begin,'test_read':False})
    print('Frozen module-transfer preflight PASSED',flush=True)

if __name__=='__main__':main()
