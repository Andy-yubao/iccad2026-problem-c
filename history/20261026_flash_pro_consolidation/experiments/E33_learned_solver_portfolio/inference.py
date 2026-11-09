"""Standalone gates and a predeclared disagreement-to-E25 consensus variant."""
from .resources import configure
RESOURCE=configure()
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from .routing import Gate
from .io import ROOT,HERE,sha,save,checkpoint,sources,snapshot

class ConsensusGate(nn.Module):
    def __init__(self,models):
        super().__init__();self.models=nn.ModuleList(models);self.last_consensus=None
    def forward(self,x):
        values=[model(x) for model in self.models];probability=torch.stack([v[0].softmax(-1) for v in values]);chosen=probability.argmax(-1)
        agree=(chosen==chosen[:1]).all(0);self.last_consensus=agree
        logits=probability.mean(0).clamp_min(1e-12).log();logits=torch.where(agree[:,None],logits,torch.zeros_like(logits))
        return logits,torch.stack([v[1] for v in values]).mean(0)

def load_gate(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if 'ensemble_members' in state:
        models=[]
        for member in state['ensemble_members']:
            source=ROOT/member['checkpoint']
            if sha(source)!=member['sha256']:raise ValueError('Ensemble member changed')
            item=torch.load(source,map_location='cpu',weights_only=False)
            if 'ensemble_members' in item:raise ValueError('Nested gate ensembles are not registered')
            model=Gate(**item['model_config']).eval();model.load_state_dict(item['model']);models.append(model)
        return ConsensusGate(models).eval(),state
    model=Gate(**state['model_config']).eval();model.load_state_dict(state['model']);return model,state

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--members',nargs=3,type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if not args.output.resolve().is_relative_to(HERE/'runs'):raise ValueError('Output must remain in E33/runs')
    if args.output.exists():raise ValueError('Ensemble manifest immutable')
    states=[torch.load(path,map_location='cpu',weights_only=False) for path in args.members]
    if [s['identity']['seed'] for s in states]!=[0,1,2]:raise ValueError('Use registered seeds 0,1,2 in order')
    for path,state in zip(args.members,states):
        result=json.loads((path.parents[1]/'results.json').read_text())
        if result['status'] not in ('early_stop','max_epochs','budget'):raise ValueError('Ensemble weights are not frozen')
        if state['identity']['inner_split']!=states[0]['identity']['inner_split']:raise ValueError('Ensemble roles differ')
    identity=copy.deepcopy(states[0]['identity']);identity.update(seed='consensus_0_1_2',sources=sources(),assembly='no gradients, unanimous predicted action or E25')
    members=[{'checkpoint':str(path.resolve().relative_to(ROOT)),'sha256':sha(path)} for path in args.members]
    value={'ensemble_members':members,'identity':identity,'model_config':{'ensemble_count':3},'progress':{'best_epoch':[s['progress']['best_epoch'] for s in states]},'assembly_only':True}
    args.output.mkdir(parents=True,exist_ok=False);checkpoint(args.output/'checkpoints/best.pt',value);save(args.output/'manifest.json',{'members':members,'rule':identity['assembly'],'test_read':False});snapshot(args.output/'source_snapshot.zip',identity)
    print('Consensus manifest saved; no new training or outer evaluation',flush=True)

if __name__=='__main__':main()
