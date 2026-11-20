"""Create an independent E148 experiment; never copy prior run artifacts."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'experiments/E147_supervision_hidden_reset'
MODULE='experiments.E148_shared_functional_shape'
SCOPE='E148.shared_functional_shape.v1.seed0'

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')

def save(path,value):write(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    if (HERE/'plan.json').exists():raise RuntimeError('Already prepared')
    excluded={'prepare.py','preflight.py','model.py','loss_config.py','shape_feedback.py'}
    copied={}
    for p in sorted(PARENT.rglob('*.py')):
        rel=p.relative_to(PARENT)
        if len(rel.parts)==1 and p.name not in excluded or rel.parts[0]=='evaluation':
            if 'runs' in rel.parts or '__pycache__' in rel.parts:continue
            raw=p.read_text(encoding='utf-8')
            text=raw.replace('experiments.E147_supervision_hidden_reset',MODULE).replace('E147.supervision_hidden_reset.v1.seed0',SCOPE).replace('E147','E148')
            write(HERE/rel,text)
            copied[rel.as_posix()]={'parent':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    for name in ('budget.json','plan.json'):
        obj=json.loads((PARENT/'evaluation'/name).read_text(encoding='utf-8'))
        if name=='plan.json':
            obj.update(revision=SCOPE+'.dev500',authorization='2026-11-17 user authorized joint E148 training and original dev500 evaluation',completed_arm_order=[],models={})
        save(HERE/'evaluation'/name,obj)
    # Preserve the public feature/cache producer bytes exactly.
    for name in ('features.py','batches.py','ports.py'):
        (HERE/name).write_bytes((PARENT/name).read_bytes())
    plan=json.loads((PARENT/'plan.json').read_text(encoding='utf-8'))
    for key in ('hidden_reset_after_layers','hidden_retained_after_layers','preflight_parent_sha256'):
        plan.pop(key,None)
    arm='e148_shared_functional_shape'
    plan.update(experiment='E148 joint shared two-layer functional networks with shape responses',revision=SCOPE,
        authorization='2026-11-17 user: first sublayer alone reads problem; no blank feedback; accept shape responses; prepare and launch background. Correction: train all networks together, supervise only layers4/8/12 with .1/.1/.8.',
        architecture={'e148':'Three reusable two-layer units called 0,0,1,1,2,2; hidden only inside units; public graph conditioning at first sublayer only; residual xy/log-aspect heads'},
        parameters={'e148':None},trainable_parameters={'e148':None},
        arms={arm:{'layer_weights':{'4':.1,'8':.1,'12':.8},'no_overlap_layers':[4],'full_loss_layers':[8,12]}},
        initialization='All modules scratch seed0; no trained weights loaded. Residual final linear weights scaled .01, bias zero. Original two graph blocks initialize draft; separate public-only two-block encoder conditions unit entry.',
        loss='Original E139 base terms/coefficients, supervision only4/8/12 weights .1/.1/.8; only4 omits direct overlap group',
        budget='Whole jointly trained candidate:4h/200epochs, not per functional network',
        dynamic_module_features=65,module_encoder_input={'first':81,'second':65},
        hidden_propagation='Every two-layer unit has fresh entry from pure public graph encoding and input layout; internal hidden retained; only layout and its gradient cross unit calls; no invocation/depth ID',
        problem_conditioning='Separate shared pure-public graph encoder (two attention blocks) computed once per forward, cached read-only; only first sublayer of each call reads it and raw features. Second sublayer reads previous hidden, current layout/dynamic feedback, pair/local graph relations, no repeated static encoding.',
        node_feedback='Unmodified E14249 +16 detached shape responses; no blank area features or loss; original uncapped escape4 and translation8 retained',
        shape_responses={'delta_log_aspect':[.2,-.2,.8,-.8],'metrics':['log1p_overlap_load','nearest_same_group_contact_cost','required_boundary_distance','bbox_area_over_public_total_area'],'current_center_fixed_before_original_nonnegative_decode_rule':True},
        training_mode='One joint optimizer, all three functional networks and both public/init encoders trained together; no stage freezing or separate training',
        unit_order=[0,0,1,1,2,2],unit_hidden_layers=2,unit_parameter_sharing='Entire unit shared across its two calls; its two internal layers differ',
        preflight='Synthetic shape oracle; exact original49 and pair18; public-only isolation; entry-only static condition; repeat-unit identity; interventions holding layouts fixed; exact new loss and gradients; guarded CUDA64/tail32; resume parity',
        comparison='E142/E147 historical same36k/batch64 panels; architecture, feedback and supervision change together, no single-factor causality',
        queue='Joint train -> frozen own-Loss-best -> original dev500/postprocessing/hard checks -> immediate Q logs -> descriptive E142 comparison')
    plan['architecture_parent']='Independent E147 numerical engine / E142 pair and node feedback'
    save(HERE/'plan.json',plan)
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=[arm],automatic_evaluation=True,automatic_retry=False,test_read=False,note=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',active=None,completed=[]))
    save(HERE/'provenance.json',dict(parent=PARENT.relative_to(ROOT).as_posix(),copied=copied,changes=['Three shared two-layer residual units; separate pure-public condition encoder; only entry reads static condition','E14249 plus shape16; no blank statistics','Joint supervision4/8/12 .1/.1/.8, overlap omitted only4']))
    s=(HERE/'loss.py').read_text(encoding='utf-8')
    write(HERE/'loss.py',s)
    s=(HERE/'run_queue.py').read_text(encoding='utf-8')
    s=s.replace("E142 control with same parameters, seed0,36k,batch64, objective and schedule; only supervised-layer hidden resets differ; adaptive single-seed result.",plan['comparison'])
    s=s.replace('supervised hidden reset versus E142','joint shared functional units and shape feedback versus E142')
    write(HERE/'run_queue.py',s)
    print('Independent E148 engine prepared; training still disabled.')

if __name__=='__main__':main()
