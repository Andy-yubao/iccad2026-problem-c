"""Secondary diagnostic: separate raw head order from nonnegative decoding."""
import json
from pathlib import Path
import numpy as np
from .topology import HERE,center_setup,center_metrics

def main():
    (HERE/'raw_center_followup_plan.json').write_text(json.dumps(dict(
        status='secondary_followup',reason='Observed nonnegative position clamp; distinguish existing head center information from decoded geometry',
        metrics=['local_axis_accuracy','knn6_recall'],fixed_reference_pairs=True,new_forward_calls=0,new_training=False),indent=2)+'\n',encoding='utf-8')
    raw=np.load(HERE/'states.npy',mmap_mode='r');areas=np.load(HERE/'areas.npy');flags=np.load(HERE/'flags.npy');ref=np.load(HERE/'reference.npy')
    all_values=np.zeros((500,13,2));nonpre=np.zeros_like(all_values)
    for i in range(500):
        n=int((areas[i]>0).sum());scale=np.sqrt(areas[i].sum());reference=ref[i,:n,:2]+ref[i,:n,2:]/2
        setup=center_setup(reference,areas[i,:n],flags[i,:n,1]!=0)
        free=np.flatnonzero(flags[i,:n,1]==0);free_setup=center_setup(reference[free],areas[i,free])
        for k in range(13):
            centers=np.asarray(raw[i,k,:n,:2],float)*scale
            values,_,_=center_metrics(centers,setup);all_values[i,k]=[values['local_axis_accuracy'],values['knn6_recall']]
            values,_,_=center_metrics(centers[free],free_setup);nonpre[i,k]=[values['local_axis_accuracy'],values['knn6_recall']]
    np.savez_compressed(HERE/'raw_center_topology.npz',all_nodes=all_values,nonpre=nonpre)
    print(json.dumps(dict(all_nodes=all_values.mean(0).tolist(),nonpre=nonpre.mean(0).tolist())))

if __name__=='__main__':main()
