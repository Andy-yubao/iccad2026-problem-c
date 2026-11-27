"""One rejected batch only, no Adam update; compare exact-RNG old/new graphs."""
import copy,gc,importlib.util,time
from pathlib import Path
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save
from experiments.E91_e74_32k_scratch.model import GeometryModel
from experiments.E91_e74_32k_scratch.data import load_rows
from experiments.E91_e74_32k_scratch.resident import ResidentRows
from experiments.E91_e74_32k_scratch.cuda_step import CapturedStep
from experiments.E91_e74_32k_scratch.loss import per_case
from experiments.E91_e74_32k_scratch.batches import inputs_only
from experiments.E91_e74_32k_scratch.train import restore_rng

def offset(state):
    data=bytes(state.tolist());assert len(data)==16
    return int.from_bytes(data[:8],'little'),int.from_bytes(data[8:],'little')

def main():
    start=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    out=Path(__file__).resolve().parent
    c=torch.load(HERE/'runs/20261031_scratch32k_seed0/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    progress=c['progress'];ids=progress['order'][progress['cursor']:progress['cursor']+32].tolist()
    rows=load_rows('train',ids);g=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146).batch(list(range(32)))
    spec=importlib.util.spec_from_file_location('experiments.E91_e74_32k_scratch._failure_reference',HERE/'runs/speedup_resume_20261031/reference_cuda_step.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    results={};values={};evaluation=None;before_rng=None
    for name,cls in [('fast',CapturedStep),('original',reference.CapturedStep)]:
        m=GeometryModel(**c['model_config']).cuda().train();m.load_state_dict(c['model'])
        original={k:v.detach().clone() for k,v in m.state_dict().items()}
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-4)
        opt.load_state_dict(copy.deepcopy(c['optimizer']));restore_rng(copy.deepcopy(c['rng']))
        step=cls(m,opt,g,'simple')
        if before_rng is None:
            after_original=torch.cuda.get_rng_state()
            step.graph.replay();torch.cuda.synchronize()
            after_probe=torch.cuda.get_rng_state()
            seed,after=offset(after_original);seed2,probe=offset(after_probe)
            assert seed==seed2 and probe>after
            delta=probe-after;assert after>=delta
            before_rng=torch.tensor(list(seed.to_bytes(8,'little')+(after-delta).to_bytes(8,'little')),dtype=torch.uint8)
        torch.cuda.set_rng_state(before_rng)
        step.graph.replay();torch.cuda.synchronize()
        assert torch.equal(torch.cuda.get_rng_state(),c['rng']['cuda'][0])
        scalar={k:float(v.detach()) for k,v in step.values.items()}
        norm=float(step.norm);values[name]=scalar
        results[name]=dict(total=scalar['total'],gradient_norm=norm,finite=bool(step.finite),
            loss_limit_exceeded=scalar['total']>=1000,gradient_limit_exceeded=norm>=10000,
            RNG_rewound_one_exact_graph_offset=True)
        assert all(torch.equal(v,original[k]) for k,v in m.state_dict().items())
        assert {float(s['step']) for s in opt.state.values()}=={float(progress['step'])}
        if name=='fast':
            m.eval()
            with torch.no_grad():
                state=m(inputs_only(g));v=per_case(state,g,'simple')
            evaluation=[dict(index=i,n=r['graph']['n'],eval_loss=float(v['total'][j]),
                eval_reference=float(v['reference'][j]),eval_overlap=float(v['overlap_total'][j]),
                eval_task=float(v['task'][j])) for j,(i,r) in enumerate(zip(ids,rows))]
        del step,opt,m,original;gc.collect();torch.cuda.empty_cache()
    assert values['fast']==values['original']
    assert results['fast']['gradient_norm']==results['original']['gradient_norm']
    save(out/'rejected_batch_replay.json',dict(status='replayed',source_step=progress['step'],
        source_epoch=progress['epoch'],source_cursor=progress['cursor'],unique_actual_cases=32,
        scope_exception='Exactly one rejected batch32 required to recover the real mean-gradient guard; no additional data or parameter updates',
        modes=results,old_new_scalar_losses_and_gradient_norm_bit_exact=True,
        eval_cases=evaluation,eval_is_not_training_Dropout=True,
        no_Adam_update=True,formal_checkpoint_untouched=True,test_read=False,seconds=time.perf_counter()-start))
    print(results,flush=True)
    print('Worst eval cases:',sorted(evaluation,key=lambda r:r['eval_loss'],reverse=True)[:5],flush=True)

if __name__=='__main__':main()
