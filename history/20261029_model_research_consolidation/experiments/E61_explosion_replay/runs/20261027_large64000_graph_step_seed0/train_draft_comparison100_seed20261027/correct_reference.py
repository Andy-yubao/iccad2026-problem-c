"""Correct display-only FP32 normalization round-trip; reuse all model outputs."""
import json
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
from generate import HERE,read,save,render,init_render,build_viewer
from datasets.expanded_reference import ExpandedReferenceLayouts
from visualization.layouts import overlap_blocks

def main():
    start=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original=ExpandedReferenceLayouts('train');drafts=read(HERE/'drafts.json');records=drafts['records'];differences=[]
    for ordinal,r in enumerate(records):
        item=original[r['train_index']];assert list(item['member'])==r['member']
        before=np.asarray(r['reference']);after=item['positions'];span=max(np.ptp(np.r_[after[:,:2],after[:,:2]+after[:,2:]],axis=0))
        differences.append({'train_index':r['train_index'],'max_roundtrip_coordinate_delta':float(abs(before-after).max()),
            'reference_red_blocks_before':len(overlap_blocks(before.tolist(),span*1e-9)),
            'reference_red_blocks_original':len(overlap_blocks(after.tolist(),span*1e-9))})
        r['reference']=after.tolist()
        if (ordinal+1)%20==0:print('Original reference read:',ordinal+1,flush=True)
    save(HERE/'drafts.json',drafts)
    ids=[r['train_index'] for r in records]
    images=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_render) as pool:
        for r in pool.map(render,ids,[True]*100):
            images.append(r)
            if len(images)%20==0:print('Corrected reference rendered:',len(images),flush=True)
    lookup={r['train_index']:r for r in images};manifest=read(HERE/'manifest.json')
    for r in manifest['cases']:r.update(lookup[r['train_index']])
    manifest['reference_coordinate_source']='Direct original unnormalized training labels; no cache scale round-trip'
    save(HERE/'manifest.json',manifest);build_viewer(manifest)
    elapsed=time.perf_counter()-start
    correction={'status':'complete','kind':'display-only reference coordinate correction','model_predictions_changed':False,
        'new_model_inference':False,'postprocessing':False,'test_read':False,'cases':differences,'elapsed_seconds':elapsed}
    save(HERE/'reference_correction.json',correction)
    summary=read(HERE/'summary.json');summary['elapsed_seconds']+=elapsed;summary['direct_original_reference']=True;save(HERE/'summary.json',summary)
    progress=read(HERE/'progress.json');progress['elapsed_seconds']+=elapsed;save(HERE/'progress.json',progress)
    print(json.dumps({'corrected':100,'previous_false_reference_red_cases':sum(x['reference_red_blocks_before']>0 and x['reference_red_blocks_original']==0 for x in differences),'max_reference_delta':max(x['max_roundtrip_coordinate_delta'] for x in differences),'elapsed_seconds':elapsed}),flush=True)

if __name__=='__main__':main()
