"""Disposable five-member check of the exact recipe used by the formal queue."""
import gc
import json
import os
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from datasets.random_reference import RandomReference
from . import train
from .augmentation import axis_swap
from .cuda_step import CapturedStep, EagerStep
from .data import REGISTRY, load_rows, records, training_indices
from .evaluation.run import public_graph
from .initialization import load_parent, verify_frozen_prefix
from .io import HERE, save, sha, sources
from .loss import drift_terms, per_case
from .loss_config import ARMS, configuration
from .model import GeometryModel
from .public import inputs_only
from .resident import ResidentRows


def main():
    report = dict(status='preparing', formal_training_started=False, dev_payloads=0, test_payloads=0)
    try:
        plan = json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        state = json.loads((HERE/'queue_state.json').read_text(encoding='utf-8'))
        assert state['status'] == 'prepared' and not (HERE/'runs').exists()
        assert sha(REGISTRY) == plan['registry_sha256']
        assert training_indices() is None and len(records('train')) == plan['train_count'] == 36000
        assert len(records('train_monitor500')) == len(records('dev_monitor500')) == 500
        assert plan['steps_per_epoch'] == 563 and plan['maximum_steps'] == 5630
        for arm in ARMS:
            for key, value in configuration(arm).items():
                if key in plan:
                    assert value == plan[key], key
            assert configuration(arm)['reference_weight'] == plan['weights'][arm]
        assert train.learning_rate(plan, 0) == plan['lr_first']
        assert train.learning_rate(plan, plan['maximum_steps'] - 1) == plan['lr_last']

        train.seed_all(plan['seed'])
        parent = load_parent(plan)['model']
        indices = plan['preflight_train_indices']
        rows = load_rows('train', indices)
        assert len(rows) == 5 and len({tuple(r['member']) for r in rows}) == 5
        pack = ResidentRows(rows,'cuda',max_nodes=120,max_pins=145,max_ports=512)
        batch = pack.batch(range(5))
        model = GeometryModel().cuda().eval()
        model.load_state_dict(parent)
        verify_frozen_prefix(model, parent)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == plan['trainable_parameters']
        with torch.no_grad():
            states = model(inputs_only(batch))
            assert states.shape == (13,5,120,3)
            no_ref = per_case(states,batch,ARMS[1],diagnostics=True)
            with_ref = per_case(states,batch,ARMS[0])
            torch.testing.assert_close(with_ref['total'],no_ref['total'] + no_ref['reference'],rtol=0,atol=0)
            changed_target = dict(batch,target=batch['target']+100)
            torch.testing.assert_close(per_case(states,changed_target,ARMS[1])['total'],no_ref['total'],rtol=0,atol=0)
            assert not torch.equal(per_case(states,changed_target,ARMS[0])['total'],with_ref['total'])
            assert all(not term.any() for term in drift_terms(states[8],states[8],batch))

        # Verify the timed inference path on one of the same five train members.
        item = RandomReference('train')[indices[0]]
        assert item['member'] == list(rows[0]['member'])
        graph = public_graph(item['solver_inputs'])
        m=graph['port_features'].shape[1]
        for key, original in graph.items():
            reference = batch[key][:1].cpu()
            actual = reference[..., :120+m] if key in ('role_hops','role_edges','role_group') else reference
            if key in ('port_features','port_valid'):
                actual = reference[:,:m]
            torch.testing.assert_close(actual,original,rtol=0,atol=0)

        del model, states, no_ref, with_ref, changed_target, batch
        gc.collect()
        torch.cuda.empty_cache()
        sample64 = pack.batch([i % 5 for i in range(64)])
        flags = torch.arange(64,device='cuda') % 2 == 1
        with torch.no_grad():
            swapped = axis_swap(dict(sample64,axis_swap=flags))
        input_prefix = None
        report['arms'] = {}
        for arm in ARMS:
            train.restore_rng(initial_rng) if input_prefix is not None else None
            model = GeometryModel().cuda().train()
            model.load_state_dict(parent)
            optimizer = train.optimizer_for(model)
            verify_frozen_prefix(model,parent)
            with torch.no_grad():
                prefix = model(inputs_only(swapped))[8].detach()
            if input_prefix is None:
                input_prefix = prefix
                initial_rng = train.rng_state()
            else:
                torch.testing.assert_close(prefix,input_prefix,rtol=0,atol=0)
            cap = CapturedStep(model,optimizer,sample64,arm)
            cap.inputs['axis_swap'].copy_(flags)
            assert not optimizer.state
            assert all(torch.equal(p.detach().cpu(),parent[name]) for name,p in model.named_parameters())
            for _ in range(2):
                values,norm = cap.replay()
            tail = EagerStep(model, optimizer, pack.batch([i % 5 for i in range(32)]), arm, cap.projection)
            tail.inputs['axis_swap'].copy_(flags[:32])
            values,norm = tail.replay()
            values,norm = cap.replay()
            assert cap.projection.steps == 4
            verify_frozen_prefix(model,parent)
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                       for p in model.units[2].parameters())
            assert any(not torch.equal(p.detach().cpu(),parent[name])
                       for name,p in model.named_parameters() if name.startswith('units.2.'))
            report['arms'][arm] = dict(accepted_disposable_steps=4, batch_sequence=[64,64,32,64], loss=float(values['total']),
                                      gradient_norm=float(norm))
            print(f'{arm} guarded GPU updates passed with frozen L8',flush=True)
            del tail, cap, model, optimizer, values, norm
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status='passed',sources=sources(),train_members=[r['member'] for r in rows],
                      frozen_prefix_bitwise=True,no_reference_target_invariance=True,
                      eval_public_batch_exact=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        save(HERE/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    main()
