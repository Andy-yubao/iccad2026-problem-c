"""Render compact scientific charts from the saved audit tables only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
    data=json.loads((HERE/'additional_summary.json').read_text(encoding='utf-8'))
    keys=['group_fraction','raw_overlap','raw_contact_loss','preplaced_fraction','locked_fraction','mib_count_blocks','corner_fraction']
    outcomes=['center_move','topology_change_fraction','failed_shape_calls','postprocess_seconds','final_Q']
    lookup={(r['x'],r['y']):r['fixed_n_partial_rho'] for r in data['fixed_n_associations']}
    matrix=np.array([[lookup[k,y] for y in outcomes] for k in keys])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
    im=ax.imshow(matrix,vmin=-.65,vmax=.65,cmap='RdBu_r')
    ax.set_xticks(range(len(outcomes)),['Repair move','Axis changes','Nonlegal candidates','Repair time','Final Q'],rotation=20,ha='right')
    ax.set_yticks(range(len(keys)),['Grouped fraction','Draft overlap','Reference-contact loss','Preplaced fraction','Locked fraction','MIB block count','Corner fraction'])
    for i in range(len(keys)):
        for j in range(len(outcomes)):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.4 else 'black')
    ax.set_title('Association within identical module counts\nPartial rank correlation; exploratory, not causal',fontsize=12,pad=12)
    fig.colorbar(im,ax=ax,label='Partial rank correlation');fig.savefig(HERE/'conditional_associations.png',dpi=170);plt.close(fig)
    s=json.loads((HERE/'summary.json').read_text(encoding='utf-8'));target=s['targets']['dev_monitor500']
    fig,ax=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    ax[0].bar(['Reference ratio\n(decoded at new area)','E39 predicted ratio'],[target['oracle_loss_mib'],s['overall']['raw_mib_pair_loss_mean']],color=['#e38159','#398aad'])
    ax[0].set(ylabel='Mean MIB pair log-dimension loss',title='Model partially learns equal shape')
    ax[1].bar(['Training monitor','Dev monitor'],[s['targets']['train_monitor500']['mib_ratio_conflict_cases']/500,s['targets']['dev_monitor500']['mib_ratio_conflict_cases']/500],color=['#e38159','#398aad'])
    ax[1].set(ylim=(0,1.12),ylabel='Fraction of cases',title='Original ratios conflict with equal shape')
    for i,v in enumerate([.982,.97]):ax[1].text(i,v+.02,f'{v:.1%}',ha='center')
    fig.suptitle('Frozen v5 target consistency audit | 500 train + 500 dev monitoring cases',fontsize=12)
    fig.savefig(HERE/'target_conflict.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
