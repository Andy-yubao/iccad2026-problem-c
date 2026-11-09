"""Explicit engineering migration; preserve cursor, RNG, Adam and budget."""
import shutil,time
import torch
from .common import HERE,RUN,read,save,sha,snapshot,sources,utc

def main():
    assert (RUN/'STOP').exists();check=read(RUN/'engineering/check_v2.json');assert check['status']=='passed'
    latest=RUN/'L/checkpoints/latest.pt';saved=torch.load(latest,map_location='cpu',weights_only=False);assert saved['status']=='user_stop' and not saved['finished']
    folder=RUN/'engineering/v1_preserved';folder.mkdir(parents=True,exist_ok=False)
    for p in [latest,RUN/'L/checkpoints/best.pt',RUN/'sources.json',RUN/'source_snapshot.zip',RUN/'state.json']:
        shutil.copy2(p,folder/p.name)
    prior_source=read(RUN/'sources.json');checkpoint_sha=sha(latest);old_elapsed=saved['elapsed_seconds'];prior_elapsed=old_elapsed+max(0.,time.time()-latest.stat().st_mtime)
    new_source=snapshot()
    save(RUN/'engineering/migration_v2.json',dict(revision='E101.engineering.v2',authorized_by='User 2026-11-03 requests faster GPU training; same final method and data',registered_utc=utc(),from_checkpoint_sha256=checkpoint_sha,from_steps=saved['progress']['step'],from_cursor=saved['progress']['cursor'],old_elapsed_seconds=old_elapsed,preserved_elapsed_seconds=prior_elapsed,old_sources=prior_source,new_sources=new_source,change='L only: second CUDA graph for guarded capturable AdamW plus unchanged module displacement projection; same FP32 objective/batch32/member order/LR tiers. GPU bias corrections change floating-point order; five-case differences recorded, not bit-exact continuation. F retains old update path.',state_retained=['model','Adam moments and exact scalar step values','RNG','order','cursor','seen','scheduler','best/latest identity','cumulative budget including engineering interruption'],check=check,original_source_and_weights_preserved=str(folder),test_payloads_read=False))
    print('Migration registered at step',saved['progress']['step'],'cursor',saved['progress']['cursor'],'budget',prior_elapsed,flush=True)
if __name__=='__main__':main()
