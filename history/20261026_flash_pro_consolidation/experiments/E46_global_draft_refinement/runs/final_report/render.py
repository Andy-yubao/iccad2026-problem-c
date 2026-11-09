"""Render saved outcomes only, through the shared layout renderer."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json
import numpy as np
from datasets.reference_layouts import ReferenceLayouts
from experiments.E46_global_draft_refinement.io import HERE,save
from visualization.layouts import plot_case,geometry_details


def main():
    folder=Path(__file__).resolve().parent;images=folder/'images';images.mkdir(exist_ok=True)
    detail=json.loads((folder/'diagnostics.json').read_text(encoding='utf-8'))
    states=[json.loads((HERE/'runs'/tag/'results_local.json').read_text(encoding='utf-8')) for tag in ('20261024_coupled_screen100','20261024_frozen_confirm400')]
    records={r['index']:r for s in states for r in s['records']};gate=json.loads((HERE/'runs/20261024_shape_memory_gate5/results_local.json').read_text(encoding='utf-8'))
    selected=[('best_large',detail['best_improvements'][0]['index'],'large_area_fast'),('worst_large',detail['largest_regressions'][0]['index'],'large_area_fast'),('draft_shape_example',443,'shape_trust')]
    data=ReferenceLayouts('dev_monitor500');manifest=[]
    for label,index,policy in selected:
        r=records[index] if policy=='large_area_fast' else next(r for r in gate['records'] if r['index']==index)
        case=data[index]['case'];constraints=np.asarray(case.solver_inputs[5]).astype(int).tolist()
        b,c=r['variants']['control'],r['variants'][policy]
        variants={'raw':{'positions':r['draft'],'metrics':{}},'control':{'positions':b['positions'],'metrics':b['metrics']},'candidate':{'positions':c['positions'],'metrics':c['metrics']}}
        for v in variants.values():v['soft_details']=geometry_details(v['positions'],constraints)
        row={'panel':'E46 saved dev layouts','case_id':f'monitor index {index}','constraints':constraints,'variants':variants,
            'caption':('Rejected shape-trust mechanism, illustrative case only' if policy=='shape_trust' else 'Frozen large-area + public-stop candidate; saved outputs, no new solving')}
        out=images/f'{label}_index{index}.png'
        layout=plot_case(row,out,stages=(('raw','Original model draft'),('control',f"Original steady (Q={b['metrics']['cost']:.4f})"),('candidate',f"{policy} (Q={c['metrics']['cost']:.4f})")))
        ref=np.array(case.reference_positions);raw=np.array(r['draft']);unit=np.sqrt(np.asarray(case.solver_inputs[1]));reference_center=ref[:,:2]+ref[:,2:]/2
        row_diag={'index':index,'policy':policy,'plot':out.relative_to(folder).as_posix(),'layout':layout,'saved_reference_error_diagnostic':{}}
        for stage,v in variants.items():
            x=np.array(v['positions']);center=x[:,:2]+x[:,2:]/2
            row_diag['saved_reference_error_diagnostic'][stage]={'mean_center_error_in_block_units':float(np.mean(np.linalg.norm(center-reference_center,axis=1)/unit)),
                'mean_abs_log_ratio_error':float(np.mean(abs(np.log(x[:,2]/x[:,3])-np.log(ref[:,2]/ref[:,3]))))}
        ratio=ref[:,2:]/raw[:,2:]
        row_diag['reference_original_WH_within_model_075_1333_fraction']=float(np.mean(((ratio>=.75)&(ratio<=4/3)).all(1)))
        row_diag['reference_scope']='Offline original-reference diagnostic only; no claim that original labels are hard legal for transformed MIB inputs.'
        manifest.append(row_diag);print(label,index,str(out),flush=True)
    save(folder/'layout_manifest.json',{'plots':manifest,'new_solver_calls':0,'new_model_calls':0,'test_read':False})


if __name__=='__main__':main()
