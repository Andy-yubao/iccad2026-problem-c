"""Complete saved diagnosis after fixing only the synthetic FP32 example."""
from datetime import datetime,timezone
import numpy as np
from .io import HERE,read,save,sha
from .collect import RUNS,key
from .diagnose import OUT,OLD,summarize,saved_center_controls,pin_encoding_collision
from .diagnostic_metrics import coverage_label


def main():
    if (OUT/'results.json').exists():raise ValueError('Do not overwrite completed diagnosis')
    records=read(OUT/'case_diagnostics.json');features=read(OLD/'features.json')['records']
    old=read(OLD/'forwards.json');stored={key(r['member']):r for r in old['models']['E65_geometry']}
    error=max(abs(r['G']-stored[key(r['member'])]['loss_total']) for r in records)
    assert error<2e-5
    config=read(OUT/'config.json')
    assert sha(RUNS['E65_geometry']/'checkpoints/best.pt')==config['E65_checkpoint_sha256']
    base_digest=sha(RUNS['E42']/'checkpoints/best.pt')
    assert base_digest==old['identities']['E42']['checkpoint_sha256']
    result={'status':'complete','case_count':len(records),'summary':summarize(records),
            'saved_E63_controls':saved_center_controls(features),'pin_encoding_collision':pin_encoding_collision(),
            'E42_checkpoint_sha256':base_digest,'E65_checkpoint_sha256':config['E65_checkpoint_sha256'],
            'parameter_probe_count':len(read(OUT/'parameter_gradients.json')),
            'input_sensitivity_cases_per_model':len(read(OUT/'input_sensitivity.json')['E65']),
            'training_started':False,'optimizer_created':False,'postprocessing_started':False,'test_read':False,
            'stored_E65_G_max_abs_difference':error,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'continuation':'All500 forwards,5 train gradient probes and5x2 sensitivity cases already saved. Initial run failed only at synthetic exact-equality assertion due to a one-ULP centroid rounding difference. Changed synthetic normalization to exact scale2 and finished from saved evidence; no model inference repeated.',
            'seconds_initial_config_to_last_forward_file':(OUT/'input_sensitivity.json').stat().st_mtime-datetime.fromisoformat(config['utc']).timestamp()}
    density_median={n:float(np.median([f['connection_density'] for f in features if f['n']==n])) for n in {f['n'] for f in features}}
    result['pin_density_cross_strata']={}
    for p in ('low','middle','high'):
        for sparse in (True,False):
            subset=[r for r in records if coverage_label(r['pin_coverage'])==p and (r['density']<=density_median[r['n']])==sparse]
            result['pin_density_cross_strata'][p+('_sparse' if sparse else '_dense')]=summarize(subset)['all']
    save(OUT/'results.json',result)
    print('Saved evidence completed without new inference.')
    print('E63 controls',result['saved_E63_controls']['strata'])
    print('cross strata',{k:{t:v[t] for t in ('count','G','overlap')} for k,v in result['pin_density_cross_strata'].items()})


if __name__=='__main__':main()
