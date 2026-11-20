"""Five train cases, real gradients and CUDA capture, but no optimizer updates."""
import ast
import hashlib
import json
import time
import traceback
import numpy as np
import torch
from .io import OWNER, ROOT, save, sources
from .train import MODEL, seed_all, optimizer_for
from .model import GeometryModel
from .features import graph_features
from .data import load_rows
from .batches import inputs_only, decode
from .resident import ResidentRows
from .loss_config import ARMS, layer_weights
from .loss import per_case, base_per_case
from .cuda_step import CapturedStep
from .augmentation import axis_swap


def digest(model):
    value = hashlib.sha256()
    for k, v in model.state_dict().items():
        value.update(k.encode())
        value.update(v.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def main():
    started = time.perf_counter()
    result = dict(status='error', unique_train_cases=5, dev_payloads=0,
                  test_read=False, optimizer_updates=0, training_started=False)
    try:
        for path in OWNER.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        authorization = json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
        assert not authorization['training_enabled'] and (OWNER/'STOP').exists()
        assert not (OWNER/'runs').exists(), 'Preparation must not create a training run'
        seed_all(0)
        rows = load_rows('train', plan['preflight_train_indices'])
        assert len({tuple(r['member']) for r in rows}) == 5
        max_pins = max(r['graph']['pin_coordinates'].shape[1] for r in rows)
        pack = ResidentRows(rows, 'cuda', max_nodes=120, max_pins=max_pins)
        batch = pack.batch([i % 5 for i in range(32)])
        public = inputs_only(batch)
        assert 'target' not in public and 'reference' not in public and 'mib' not in public
        assert all(torch.equal(r['graph']['mask'], torch.ones_like(r['graph']['mask'])) for r in rows)
        ds_module = __import__('datasets.random_reference', fromlist=['RandomReference'])
        ds = ds_module.RandomReference('train')
        # Re-read one already-counted case to check raw areas and ignored MIB fields.
        item = ds[plan['preflight_train_indices'][0]]
        raw = item['solver_inputs']
        assert torch.equal(rows[0]['graph']['area']*rows[0]['graph']['scale'].square(),
                           torch.as_tensor(raw[1]).float()) or torch.allclose(
                           rows[0]['graph']['area']*rows[0]['graph']['scale'].square(),
                           torch.as_tensor(raw[1]).float(), rtol=1e-6, atol=1e-7)
        changed = list(raw)
        changed[5] = np.asarray(raw[5]).copy()
        changed[5][:,2] = np.arange(raw[0])+100
        original = graph_features(raw)
        alternative = graph_features(tuple(changed))
        assert torch.equal(original['features'], alternative['features'])
        assert all(torch.equal(a,b) for a,b in zip(original['relations'], alternative['relations']))
        assert original['features'].shape[-1] == 16
        results = {}
        expected_digest = None
        expected_output = None
        for arm in ARMS:
            seed_all(0)
            model = GeometryModel(**MODEL).cuda()
            assert sum(p.numel() for p in model.parameters()) == plan['parameters']
            assert all(p.requires_grad for p in model.parameters())
            before = digest(model)
            if expected_digest is None: expected_digest = before
            assert before == expected_digest
            model.eval()
            with torch.no_grad():
                output = model(public)
                assert output.shape == (12,32,120,3)
                if expected_output is None: expected_output = output.detach().cpu()
                else: assert torch.equal(output.cpu(), expected_output)
                rejected = False
                try: model({**public, 'target': batch['target']})
                except ValueError: rejected = True
                assert rejected, 'Network accepted target geometry'
                decoded = decode(output[-1], batch)
                altered_state = output[-1].clone()
                altered_state[...,0] += 1.
                altered_state[...,2] += .2
                altered = decode(altered_state, batch)
                locked = batch['valid'] & batch['constraints'][...,:2].bool().any(-1)
                assert locked.any()
                assert (decoded[locked]-altered[locked]).abs().sum() > 0
                values = per_case(output, batch, arm)
                weights = layer_weights(arm)
                explicit = sum(weight*base_per_case(output[i-1],batch)['total'] for i,weight in weights.items())
                assert torch.allclose(values['total'], explicit, rtol=1e-6,atol=1e-6)
                assert len([k for k in values if k.startswith('layer')]) == len(weights)
                swapping = {**batch, 'axis_swap': torch.ones(32,device='cuda',dtype=torch.bool)}
                twice = axis_swap(axis_swap(swapping))
                for key in batch: assert torch.equal(batch[key], twice[key]), key
            # Direct target supervision covers fixed/preplaced position and shape too.
            leaves = output.detach().clone().requires_grad_(True)
            supervised = base_per_case(leaves[-1], batch)
            (supervised['position']+supervised['shape']).mean().backward()
            assert leaves.grad[-1][locked][...,:2].abs().sum() > 0
            assert leaves.grad[-1][locked][...,2].abs().sum() > 0
            del leaves, supervised, values, explicit, decoded, altered, altered_state
            model.train()
            optimizer = optimizer_for(model)
            torch.cuda.reset_peak_memory_stats()
            cap = CapturedStep(model, optimizer, batch, arm)
            # Deliberately use graph only. cap.replay would update Adam and is forbidden here.
            cap.graph.replay()
            torch.cuda.synchronize()
            assert bool(cap.finite)
            assert float(cap.max_case) < 1000 and float(cap.values['total']) < 1000
            assert float(cap.norm) < 10000
            assert model.layers[0].qkv.weight.grad.abs().sum() > 0
            assert model.layers[-1].qkv.weight.grad.abs().sum() > 0
            assert model.head[0].weight.grad.abs().sum() > 0
            assert digest(model) == before
            assert not optimizer.state and cap.projection.steps == 0
            results[arm] = dict(total=float(cap.values['total']),
                final_total=float(cap.values['final_total']), gradient_norm=float(cap.norm),
                supervised_layers=list(weights), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                initial_parameters_sha256=before, parameters_unchanged=True,
                first_last_blocks_receive_gradients=True, cuda_graph_replayed_without_Adam=True,
                optimizer_updates=0)
            del cap, optimizer, model, output
            torch.cuda.empty_cache()
        result.update(status='passed', sources=sources(), arms=results,
            parameters=plan['parameters'], actual_nodes=[r['graph']['n'] for r in rows],
            members=[list(r['member']) for r in rows], max_loss_ports_in_check=max_pins,
            all_module_prediction=True, no_MIB_input_or_loss=True,
            hard_specifications_not_overwriting_draft=True,
            batch32_FP32=True, same_initial_state_and_eval_output=True,
            real_optimizer_update_tested=False, resume_tested=False,
            check_scope='Only five train members; no epoch, optimizer step, checkpoint or evaluation run')
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        save(OWNER/'preflight.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
