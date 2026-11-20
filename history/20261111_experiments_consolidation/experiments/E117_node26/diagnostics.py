"""Cache the same five train members' real twelve layers, without extra dev inference."""
import json
import numpy as np
import torch
from .io import OWNER, ROOT, save, sha
from .model import GeometryModel
from .data import load_rows, collate
from .batches import inputs_only, decode
from .train import seed_all

def main():
    if any((OWNER/p).exists() for p in ('STOP','QUEUE_STOP','runs/node26_seed0/STOP')):raise InterruptedError('STOP')
    seed_all(0)
    plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    path=OWNER/'runs/node26_seed0/checkpoints/best.pt'
    frozen=json.loads((OWNER/'evaluation/frozen_model.json').read_text(encoding='utf-8'))
    assert sha(path)==frozen['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    rows=load_rows('train',plan['preflight_train_indices']);batch,_=collate(rows,'cuda')
    dest=ROOT/'visualization/E117_node26/20261109_train5';dest.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        states=model(inputs_only(batch))
        layers=torch.stack([decode(s,batch)*batch['scale'][:,None,None] for s in states]).cpu().numpy()
    np.savez_compressed(dest/'layers.npz',layers=layers,valid=batch['valid'].cpu().numpy(),
                        reference=(batch['reference']*batch['scale'][:,None,None]).cpu().numpy(),
                        constraints=batch['constraints'].cpu().numpy())
    save(dest/'manifest.json',dict(checkpoint_sha256=frozen['checkpoint_sha256'],best_epoch=frozen['best_epoch'],
        members=[list(r['member']) for r in rows],layers=list(range(1,13)),test_read=False,
        style='G colors only; IDs/G/F/P/L/R/T/B; no MIB; use shared visualization when rendering'))

if __name__=='__main__':main()
