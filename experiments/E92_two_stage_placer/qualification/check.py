"""Preparation checks only: no NN inference, optimizer step or Q evaluation."""
import ast,copy,json,math
from .common import HERE,ROOT,HOME,registration,ready,freeze,sources
from ..io import save,sources as training_sources
from .run import gate,validate_terminal,diagnostic
from .worker import STEADY

def main():
    checks={}
    for path in HOME.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    reg=registration()
    source=next(n for n in reg['baseline']['source_files'] if n.replace('\\','/').endswith('terminal/results.json'))
    terminal=ROOT/source;config=json.loads((terminal.parent/'config.json').read_text(encoding='utf-8'))
    assert STEADY==config['steady']
    from ..dataset_view import ReferenceLayouts
    metadata=ReferenceLayouts('dev_monitor500').records
    records=json.loads(terminal.read_text(encoding='utf-8'))['records']
    validate_terminal(records,metadata)
    checks['frozen_runtime_and_members']='all copied sources verified; exact E90 flash options and same500 metadata'
    # Five saved E90 drafts only: no NN call and no fresh Q. Include artificial
    # zero padding to reproduce the historical reporting bug's relevant boundary.
    from ..data import load_rows
    import numpy as np
    rows=load_rows('dev_monitor500',range(5))
    old_geometry=json.loads((terminal.parents[1]/'inspection500.json').read_text(encoding='utf-8'))['geometry500']['E90']['records']
    for index,row in enumerate(rows):
        assert list(row['member'])==records[index]['member']
        padded=np.concatenate([np.asarray(records[index]['draft_positions']),np.zeros((2,4))],axis=0)
        actual=diagnostic((index,row,padded,{'total':0.}))
        expected=old_geometry[index]
        assert actual['N10']==expected['N10']
        for key in ('module_pass10_fraction','max_covered_fraction','bbox_over_A'):
            assert abs(actual[key]-expected[key])<1e-12,(index,key,actual[key],expected[key])
    checks['saved_geometry_padding']='five saved E90 drafts reproduce exact coverage/N10/bbox; zero padding is excluded'
    limits=reg['qualification']['gates']
    values={'dev500_F_loss':limits['dev500_F_loss_lte'],'weighted_Q':limits['weighted_Q_lte'],
        'hard_legal':500,'max_Q':limits['max_Q_lte'],'module_pass10_pooled':limits['module_pass10_pooled_gte'],
        'worst_module_coverage_p95':limits['worst_module_coverage_p95_lte'],
        'numerical_stability_pass':True,'F_phase_checkpoint_exists':True}
    assert all(gate(values,limits).values())
    bad=copy.deepcopy(values);bad['weighted_Q']=float('nan');assert not gate(bad,limits)['weighted_Q']
    bad=copy.deepcopy(values);bad['F_phase_checkpoint_exists']=False;assert not gate(bad,limits)['F_checkpoint']
    checks['gate_boundaries']='boundary accepted, NaN quality rejected, missing F rejected'
    try:validate_terminal([records[0],records[0]],metadata)
    except ValueError:pass
    else:raise AssertionError('Duplicate terminal cursor accepted')
    checks['resume_cursor']='duplicate completed members rejected'
    ok,reason=ready()
    if not ok:
        try:freeze()
        except RuntimeError as error:assert reason in str(error)
        else:raise AssertionError('Premature model freeze accepted')
    checks['readiness_guard']={'ready':ok,'reason':reason,'new_model_inference_calls':0,'new_Q_calls':0}
    # Adding this directory must not invalidate the active training registration.
    preflight=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    assert training_sources()==preflight['sources']
    checks['training_identity']='unchanged; qualification separately versioned'
    report={'status':'passed','checks':checks,'sources':sources(),'saved_geometry_cases':5,'model_inference_cases':0,
        'postprocessing_calls':0,'test_read':False,'boundary':'static/saved-result checks only; real frozen-model evaluation remains pending'}
    save(HERE/'runs/qualification_preparation/checks.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
