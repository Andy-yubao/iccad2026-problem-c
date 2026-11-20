"""Compose stage-own-Loss best checkpoints before terminal Q evaluation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import torch
from experiments.E101_final_c5_solver.common import HERE,RUN,PLAN,read,save,checkpoint,sha,check_stop,sources,utc
from experiments.E101_final_c5_solver.train import baseline

def main():
    check_stop()
    for stage in ('L','F'):assert read(RUN/stage/'results.json')['status']=='early_stop'
    expected=read(RUN/'sources.json');current=sources()
    assert all(current.get(n)==h for n,h in expected.items()),'Frozen training dependency changed'
    model,cfg=baseline();lp=RUN/'L/checkpoints/best.pt';fp=RUN/'F/checkpoints/best.pt'
    l=torch.load(lp,map_location='cpu',weights_only=False);f=torch.load(fp,map_location='cpu',weights_only=False)
    model.placer.load_state_dict(l['model'],strict=True);model.refiner.load_state_dict(f['model'],strict=True)
    path=HERE/'delivery/model.pt'
    checkpoint(path,dict(model=model.state_dict(),model_config=cfg,identity=dict(plan_sha256=sha(HERE/'plan.json'),L_best_sha256=sha(lp),F_best_sha256=sha(fp)),selection='Unique stage own-Loss best; no Q or test reselection'))
    (RUN/'evaluation/dev_monitor500/gallery').mkdir(parents=True,exist_ok=True)
    save(RUN/'frozen_selection.json',dict(checkpoint_sha256=sha(path),L_best_epoch=l['progress']['best_epoch'],F_best_epoch=f['progress']['best_epoch'],frozen_utc=utc(),test_payloads_read=False,test_authorization=PLAN['authorization'],test_scope='Repeated registered2000, previously exposed by E101; no test feedback'))
    print('Unique E74+C5 model frozen; dev500 then repeated test2000',flush=True)

if __name__=='__main__':main()
