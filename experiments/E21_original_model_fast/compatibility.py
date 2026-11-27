"""Five public-input checks of the original retained model's exact calculation."""
import torch
import numpy as np
from .io import ROOT,HERE,sha,save,sources,snapshot
from .cases import WEIGHTS,WEIGHT_HASH,graph,seed_all,predict
from .model import GeometryModel
from .legacy_model import GeometryModel as LegacyModel
from .batches import decode as old_decode


@torch.no_grad()
def main():
    seed_all();assert sha(WEIGHTS)==WEIGHT_HASH
    state=torch.load(WEIGHTS,weights_only=False,map_location='cpu')
    model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    legacy=LegacyModel(**state['model_config']).eval();legacy.load_state_dict(state['model'])
    rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    results=[]
    for index in [21,127,275,447,456]:
        inputs=rows[index]['case'].solver_inputs;g=graph(inputs)
        a=model(g);b=legacy(g)
        assert torch.equal(a,b)
        x,_=predict(model,inputs);y=(old_decode(b,g)[0]*g['scale'][0]).numpy().astype(float)
        c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
        y[locked,2:]=s[locked,2:];y[c[:,1]!=0]=s[c[:,1]!=0]
        assert np.array_equal(x,y)
        results.append({'index':index,'maximum_state_error':float((a-b).abs().max()),'maximum_physical_error':float(np.max(np.abs(x-y)))})
    folder=HERE/'runs/20261020_compatibility5';identity={'sources':sources(),'checkpoint_sha256':WEIGHT_HASH,'test_read':False}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    save(folder/'results.json',{'records':results,'test_read':False,'passed':True});print('Five public cases: exact states and decoded layouts match original E10 implementation')


if __name__=='__main__':main()
