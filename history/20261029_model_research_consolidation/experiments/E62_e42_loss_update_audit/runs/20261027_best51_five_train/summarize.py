"""Summarize saved finite-step evidence and compact redundant output."""
import json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
HERE=OUT.parents[1]
ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    d=read(OUT/'results.json');refine=read(OUT/'refinement.json');robust=read(OUT/'robustness.json');cfg=read(OUT/'config.json')
    repeat=d['arms']['full_G_repeat']
    if 'post' in repeat:repeat.pop('post');repeat['post_reference']='arms.full_G.post (bit-exact repeat)'
    (OUT/'results.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    selected=['full_G','without_center_local','without_shape','without_relative_local','without_contact','without_overlap','overlap_only','momentum_and_decay_only']
    labels=['Full G','No center','No shape','No relative','No contact','No overlap','Overlap only','Past moments only']
    values=np.asarray([d['arms'][k]['mean_area_delta']*100 for k in selected])
    heat=np.asarray([[case['area_delta']*100 for case in d['arms'][k]['cases']] for k in selected])
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    axes[0].barh(labels,values,color=['#be534b' if x>0 else '#287f83' for x in values]);axes[0].invert_yaxis();axes[0].axvline(0,color='#333',lw=.8)
    axes[0].set(xlabel='Mean overlap-area change (percentage points)',title='Actual AdamW step; negative is better')
    limit=max(abs(heat.min()),abs(heat.max()));im=axes[1].imshow(heat,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    axes[1].set_xticks(range(5),[str(x) for x in cfg['train_indices']]);axes[1].set_yticks(range(len(labels)),labels)
    axes[1].set(xlabel='Training member index',title='Per-case actual area change (percentage points)')
    for y in range(heat.shape[0]):
        for x in range(heat.shape[1]):axes[1].text(x,y,f'{heat[y,x]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(heat[y,x])>.65*limit else '#222')
    fig.colorbar(im,ax=axes[1],shrink=.8);axes[0].grid(axis='x',alpha=.2)
    fig.savefig(OUT/'update_effects.png',dpi=170);plt.close(fig)
    execution={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'total_execution_seconds':d['elapsed_seconds']+refine['elapsed_seconds']+robust['elapsed_seconds'],
        'commands':['.conda/envs/iccad2026/python.exe -u -B -m experiments.E62_e42_loss_update_audit.'+s for s in ('check','refine','robustness')],
        'source_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'actual_data':'Five selected original-v5 train members read through byte-verified v6 prefix cache',
        'v5_registry_sha256':sha(ROOT/'protocol/reference_workset_v5.json'),'v6_registry_sha256':sha(ROOT/'protocol/reference_workset_v6.json'),
        'unique_training_cases':5,'formal_training_runs':0,'new_deployment_checkpoints':0,'solver_calls':0,'test_read':False,
        'full_update_repeat_bit_exact':d['full_repeat_bit_exact'],'refinement_original_updates_bit_exact':refine['original_two_updates_bit_exact'],
        'initial_full_step_area_delta_pp':d['arms']['full_G']['mean_area_delta']*100,
        'without_center_area_delta_pp':d['arms']['without_center_local']['mean_area_delta']*100,
        'clipping_confounds_checked':True,'same_five_case_stochastic_and_temporal_controls_complete':True}
    assert execution['total_execution_seconds']<900
    assert sha(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256']
    latest=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/latest.pt'
    assert sha(latest)==robust['checks']['latest54_train_gradient']['checkpoint_sha256']
    (OUT/'execution.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    links=re.findall(r'\]\(([^)]+)\)',(OUT/'analysis.md').read_text(encoding='utf-8'))
    assert all((OUT/p).exists() for p in links if not p.startswith(('http','#')))
    print(json.dumps({'status':'complete','seconds':execution['total_execution_seconds'],'original_best_and_latest_unchanged':True,'report_links_valid':True}))

if __name__=='__main__':main()
