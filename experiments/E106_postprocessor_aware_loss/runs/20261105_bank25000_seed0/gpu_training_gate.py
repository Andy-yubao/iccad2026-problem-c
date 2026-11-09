"""One disposable CUDA Adam step on the SAME five previously checked fit cases."""
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.E106_postprocessor_aware_loss.common import OWNER, RUN, read, save, setup
from experiments.E106_postprocessor_aware_loss.train import ensure_training_authorized
from experiments.E106_postprocessor_aware_loss.model import PostprocessorCritic
from experiments.E106_postprocessor_aware_loss.data import load_case, batch_cases
from experiments.E106_postprocessor_aware_loss.objective import fitting_loss, role_exp_mean


def main():
    ensure_training_authorized()
    setup()
    bank = read(RUN/'bank_results.json')
    assert bank['status']=='complete' and bank['drafts']==25000
    chosen = read(OWNER/'preflight.json')['case_indices']
    rows = [load_case(i) for i in chosen]
    assert len(rows)==5 and all(r['split']=='fit' for r in rows)
    # Six problems is the registered batch shape, with one repeated fit problem.
    rows.append(rows[-1])
    plan = read(OWNER/'plan.json')
    model = PostprocessorCritic(**plan['model']).cuda().train()
    policy = plan['proposed_training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy['lr_tiers'][0], weight_decay=policy['weight_decay'])
    batch = batch_cases(rows, 'cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    values = fitting_loss(model(batch['graph'],batch['states'],group_size=5), batch,
                          read(RUN/'normalization.json'), role_exp_mean('fit'))
    values['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), policy['clip_norm'])
    if not bool(torch.isfinite(norm) & torch.isfinite(values['total']) & (norm<1e4) & (values['total']<1e6)):
        raise FloatingPointError('Disposable CUDA update rejected before AdamW')
    optimizer.step()
    torch.cuda.synchronize()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    save(RUN/'gpu_training_gate.json',dict(status='passed',unique_fit_cases=5,case_indices=chosen,
        batch_problems=6,batch_drafts=30,disposable_optimizer_updates=1,formal_training_updates=0,
        checkpoint_or_optimizer_reused_by_training=False,model_parameters=sum(p.numel() for p in model.parameters()),
        initial_lr=policy['lr_tiers'][0],Loss=float(values['total']),gradient_norm=float(norm),
        step_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        new_postprocessor_calls=0,internal_check_cases=0,test_read=False))
    print('CUDA gate passed: same5 fit cases,one disposable update; formal run starts fresh.',flush=True)


if __name__=='__main__':
    main()
