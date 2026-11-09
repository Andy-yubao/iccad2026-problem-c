"""Verify fixed public count normalization; no model or numeric optimization."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import math,json,hashlib,collections
from .train_io.cache import load_rows,identity
from .train_io.io import HERE,save,sha
def main():
    output=HERE.parent/'runs/20261102_round2_public_count_inventory';output.mkdir(exist_ok=False)
    rows=load_rows('train');weights=[math.exp((r['graph']['n']-120)/12) for r in rows];mean=sum(weights)/len(weights);assert len(rows)==16000 and abs(mean-.12578465955104443)<1e-12
    values=dict(status='passed',count=len(rows),mean=mean,normalizer=.12578465955104443,N_counts=dict(collections.Counter(r['graph']['n'] for r in rows)),members_and_N_sha256=hashlib.sha256(json.dumps([(r['member'],r['graph']['n']) for r in rows]).encode()).hexdigest(),cache_identity=identity(),no_optimizer_solver_dev_Q_or_model=True,test_read=False)
    save(output/'results.json',values);print(json.dumps({k:v for k,v in values.items() if k not in ('N_counts','cache_identity')},indent=2))
if __name__=='__main__':main()
