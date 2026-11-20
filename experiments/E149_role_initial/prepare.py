"""Independent initial-only role experiment; preserve all historical sources/runs."""
from pathlib import Path
import json
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'experiments/E148_shared_functional_shape'
ARM='e149_role_initial'
SCOPE='E149.role_initial.v1.seed0'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if (HERE/'plan.json').exists():raise RuntimeError('Already prepared')
    files=('features.py','batches.py','ports.py','original_model.py','simple_loss.py','group_tree.py','step_control.py','captured_projection.py','schedule.py','train.py','cuda_step.py','augmentation.py','io.py')
    provenance={}
    for name in files:
        raw=(PARENT/name).read_bytes()
        if name in ('features.py','batches.py','ports.py'):(HERE/name).write_bytes(raw)
        else:
            s=raw.decode('utf-8').replace('\r\n','\n').replace('experiments.E148_shared_functional_shape','experiments.E149_role_initial').replace('E148.shared_functional_shape.v1.seed0',SCOPE).replace('E148','E149')
            if name in ('train.py','cuda_step.py'):s=s.replace('from .batches import inputs_only','from .public import inputs_only')
            if name=='train.py':
                s=s.replace('layers=12','layers=2')
                s=s.replace('freeze own-Loss best then fixed dev500 through authorized queue','Loss-only terminal record; no postprocessing or Q authorized')
            if name=='augmentation.py':
                s=s.replace('    return out\n',"    select('port_features',torch.stack((batch['port_features'][...,1],batch['port_features'][...,0],batch['port_features'][...,2],batch['port_features'][...,3]),-1))\n    return out\n")
            if name=='io.py':
                start=s.index('    files = ',s.index('def sources():'));end=s.index('    files += ',start)
                s=s[:start]+"    files = [p for p in HERE.rglob('*.py') if not set(p.parts)&{'runs','preparation_diagnostics','role_cache_v1'}] + [HERE/name for name in ('plan.json','provenance.json','README.md','DESIGN.md')]\n"+s[end:]
            (HERE/name).write_text(s,encoding='utf-8')
        provenance[name]={'parent':str((PARENT/name).relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()}
    (HERE/'base_data.py').write_bytes((PARENT/'data.py').read_bytes())
    (HERE/'base_resident.py').write_bytes((PARENT/'resident.py').read_bytes())
    source=(PARENT/'e119_model.py').read_text(encoding='utf-8')
    source=source[:source.index('class GeometryModel(OriginalModel):')]
    (HERE/'initial_block.py').write_text(source,encoding='utf-8')
    loss=(PARENT/'loss.py').read_text(encoding='utf-8');loss=loss[:loss.index('def supervised_per_case(')]
    loss+='''def per_case(states,g,phase,objective=DEFAULT):
    from .loss_config import ARMS
    if phase not in ARMS or states.shape[0]!=1:raise ValueError('One initial readout only')
    out=base_per_case(states[0],g,objective)
    out['full_total']=out['total'];out['full_overlap_weighted']=out['overlap_weighted']
    out['overlap_weighted']=torch.zeros_like(out['overlap_weighted'])
    out['total']=sum(out[k] for k in TOTAL_COMPONENTS if k!='overlap_weighted')
    out['final_total']=out['total'];out['original_final_total']=out['original_total']
    return out
'''
    (HERE/'loss.py').write_text(loss,encoding='utf-8')
    plan=json.loads((PARENT/'plan.json').read_text(encoding='utf-8'))
    for key in ('comparison_control','hidden_propagation','problem_conditioning','node_feedback','pair_feedback','shape_responses','unit_order','unit_hidden_layers','unit_parameter_sharing','dynamic_module_features','module_encoder_input','local_demand_width','relation_width','arms'):
        plan.pop(key,None)
    plan.update(experiment='E149 public-role enhanced initial generator only',revision=SCOPE,
        authorization='2026-11-17 user approved role-encoding proposal but restricted training to initial generator, direct initial-output supervision without overlap; prepare then launch background; Loss only',
        architecture={'e149':'Width64 two static graph/port attention blocks, public structural role16, full connected-port tokens, exact shortest-hop pair bias, one initial layout head'},
        parameters={'e149':None},trainable_parameters={'e149':None},
        initialization='Scratch seed0; only initial generator and its new public role/port encoders exist; no historical weights or refinement units',
        loss='One initial readout with weight1. Original full base coefficients excluding entire direct overlap group; collision terms diagnostic only',
        supervision={'initial':1.0},training_mode='Single joint optimizer for initial generator and its role encoders only; no S0/S1/S2 refinement or terminal backpropagation',
        role_features=16,port_features=4,pair_features=6,
        terminal_evaluation='Loss only: freeze own fixed-dev500-Loss best; no postprocessing/Q/test',
        queue='preflight -> scratch initial-only training -> frozen own-Loss best -> Loss-only summary',
        comparison='E148 joint initial generator was not directly supervised on this objective; descriptive reference only, not a matched initial-only control',
        preflight='Independent synthetic shortest-path/anchor checks, public isolation, module/port permutation, axis augmentation, initial-only objective/gradient identity, guarded CUDA64/tail32 and serialized resume parity',
        overlap_omission='All initial direct overlap group excluded, including penetration, mean load and max excess; raw diagnostics retained',
        automatic_evaluation=False,postprocessing_calls=0,Q_calls=0,
        reporting='Fixed train500/dev500 initial no-overlap Loss and all base components; full/collision diagnostics not used for best/LR/earlystop',
        cache_policy='Reuse exact original shared36k/dev500 cache. Build/reuse only new public-role derived sidecar chunks, with source/member/producer identity',
        budget='One initial-only candidate total4h/200epochs; original batch64/tail32 and numerical protection retained')
    save(HERE/'plan.json',plan)
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=[ARM],automatic_evaluation=False,automatic_retry=False,test_read=False,note=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None))
    save(HERE/'provenance.json',dict(parent=str(PARENT.relative_to(ROOT)),copied=provenance,changes=['Only initial generation trained/read out','Pure-public structural/port/hop encoding','Single no-overlap initial objective, no refinement or Q']))
    print('E149 independent engine prepared; training disabled until checks pass.')

if __name__=='__main__':main()
