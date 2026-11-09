"""Five saved cases, immutable identities and branch weights; no solver/Q call."""
import ast,json,time
import numpy as np
from ..io import ROOT,HERE,save,sha
from .common import HOME,ARM,freeze,registration,sources,seed_all
from .run import diagnostic,validate_terminal
from .worker import STEADY

def main():
    begin=time.perf_counter();reg=registration()
    for p in HOME.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    source=ROOT/reg['previous_results']['E90'];old=json.loads(source.read_text(encoding='utf-8'))
    config=json.loads((source.parent/'config.json').read_text(encoding='utf-8'));assert config['steady']==STEADY
    from ..dataset_view import ReferenceLayouts
    validate_terminal(old['records'],ReferenceLayouts('dev_monitor500').records)
    duplicate=[old['records'][0],old['records'][0]]
    try:validate_terminal(duplicate,ReferenceLayouts('dev_monitor500').records)
    except ValueError:pass
    else:raise AssertionError('Duplicate cursor accepted')
    from ..data import load_rows
    rows=load_rows('dev_monitor500',range(5))
    expected=json.loads((source.parents[1]/'inspection500.json').read_text(encoding='utf-8'))['geometry500']['E90']['records']
    for i,row in enumerate(rows):
        padded=np.concatenate([np.asarray(old['records'][i]['draft_positions']),np.zeros((2,4))])
        result=diagnostic((i,row,padded,{'total':0.}))
        assert result['N10']==expected[i]['N10']
        for key in ('module_pass10_fraction','max_covered_fraction','bbox_over_A'):assert abs(result[key]-expected[i][key])<1e-12
    from ..loss_config import COEFFICIENTS,configuration
    assert configuration('G')['reference']==configuration('C')['reference']==1.
    assert COEFFICIENTS['G']['task']==COEFFICIENTS['C']['task']
    assert COEFFICIENTS['G']['overlap']==2*COEFFICIENTS['C']['overlap']
    state,identity=freeze()
    # Five shared dev cases only; check deployed CPU inference against training decode.
    import torch
    from ..model import GeometryModel
    from ..data import collate
    from ..batches import inputs_only,decode
    from .runtime.inference import predict
    seed_all(0,2);model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    batch,_=collate(rows,'cpu')
    with torch.no_grad():
        output=model(inputs_only(batch));draft=(decode(output,batch)*batch['scale'][:,None,None]).numpy()
        from ..loss import per_case
        c=per_case(output,batch,'C');g=per_case(output,batch,'G')
        assert torch.allclose(g['total']-c['total'],COEFFICIENTS['C']['overlap']*c['overlap_total'],atol=1e-6,rtol=1e-5)
    errors=[]
    for i,row in enumerate(rows):
        predicted,_=predict(model,row['case'].solver_inputs)
        error=float(np.max(np.abs(predicted-draft[i,:row['graph']['n']]))/float(row['graph']['scale']))
        errors.append(error)
    if max(errors)>1e-4:raise ValueError('Deployed decode differs: '+str(errors))
    result={'status':'passed','arm':ARM,'checkpoint_sha256':identity['checkpoint_sha256'],
            'selected_epoch':identity['best_epoch'],'sources':sources(),'saved_geometry_cases':5,'model_inference_cases':5,
            'normalized_cpu_decode_max_errors':errors,'postprocessing_calls':0,'test_read':False,
            'checks':['syntax/import','frozen source/runtime/evidence hashes','own-arm Loss minimum and exact best epoch',
                      'exact E90 flash settings and members','duplicate resume cursor rejected','padding-safe exact coverage',
                      'C/G coefficient and actual Loss difference','training vs deployed decode'],
            'elapsed_seconds':time.perf_counter()-begin}
    save(HERE/f'runs/evaluation/preflight_{ARM.lower()}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
