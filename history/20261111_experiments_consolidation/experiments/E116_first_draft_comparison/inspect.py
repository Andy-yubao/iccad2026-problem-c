"""Five frozen train members; forward/backward/capture only, no Adam update."""
import ast
import hashlib
import json
import sys
import time
import traceback
import numpy as np
import torch
from datasets.random_reference16k import RandomReference16k
from . import train as runner
from .io import OWNER, ROOT, save, sources
from .model import ARMS, build_model
from .original_model import GeometryModel as ReferenceOriginal
from .features import graph_features
from .data import load_rows
from .batches import inputs_only, decode
from .resident import ResidentRows
from .loss import base_per_case, per_case
from .cuda_step import CapturedStep
from .augmentation import axis_swap


def digest(model):
    value = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        value.update(key.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def check_launch_blocked(arm):
    before = sys.argv
    try:
        sys.argv = ['E116.train', '--arm', arm]
        try:
            runner.main()
        except PermissionError as error:
            assert 'training is disabled' in str(error)
            assert not (OWNER/'runs').exists()
            return True
        raise AssertionError('Disabled training entry did not reject launch')
    finally:
        sys.argv = before


def main():
    started = time.perf_counter()
    result = dict(status='error', unique_train_cases=5, dev_payloads=0,
                  test_read=False, optimizer_updates=0, training_started=False,
                  real_optimizer_update_tested=False, resume_tested=False)
    try:
        for path in OWNER.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        auth = json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
        assert not auth['training_enabled'] and not auth['actual_updates_authorized']
        assert (OWNER/'STOP').exists() and not (OWNER/'runs').exists()
        blocked = {arm: check_launch_blocked(arm) for arm in ARMS}
        runner.seed_all(0)
        original = build_model('original4')
        original_rng = torch.get_rng_state().clone()
        runner.seed_all(0)
        candidate = build_model('new_first')
        assert torch.equal(original_rng, torch.get_rng_state())
        shared = []
        for key, value in original.state_dict().items():
            if key.startswith('input.') or any(
                    key.startswith(f'layers.{i}.{name}.') for i in range(4)
                    for name in ('norm1', 'norm2', 'qkv', 'output', 'local')):
                assert torch.equal(value, candidate.state_dict()[key]), key
                shared.append(key)
        assert len(shared) == 44
        assert len(original.layers) == len(candidate.layers) == 4
        del original, candidate
        rows = load_rows('train', plan['preflight_train_indices'])
        assert len({tuple(r['member']) for r in rows}) == 5
        assert all(torch.equal(r['graph']['mask'], torch.ones_like(r['graph']['mask'])) for r in rows)
        max_pins = max(r['graph']['pin_coordinates'].shape[1] for r in rows)
        pack = ResidentRows(rows, 'cuda', max_nodes=120, max_pins=max_pins)
        batch = pack.batch([i % 5 for i in range(32)])
        public = inputs_only(batch)
        assert not {'target', 'reference', 'mib', 'pin_coordinates'} & set(public)
        raw = RandomReference16k('train')[plan['preflight_train_indices'][0]]['solver_inputs']
        assert torch.allclose(rows[0]['graph']['area']*rows[0]['graph']['scale'].square(),
                              torch.as_tensor(raw[1]).float(), rtol=1e-6, atol=1e-7)
        changed = list(raw)
        changed[5] = np.asarray(raw[5]).copy()
        changed[5][:, 2] = np.arange(raw[0])+100
        a, b = graph_features(raw), graph_features(tuple(changed))
        assert torch.equal(a['features'], b['features'])
        assert all(torch.equal(x, y) for x, y in zip(a['relations'], b['relations']))
        results = {}
        for arm in ARMS:
            print(f'Checking {arm}: five train cases, no optimizer update', flush=True)
            runner.seed_all(0)
            model = build_model(arm).cuda()
            count = sum(p.numel() for p in model.parameters())
            assert count == plan['parameters_by_arm'][arm]
            assert all(p.requires_grad for p in model.parameters())
            before = digest(model)
            model.eval()
            calls = []
            hook = model.head.register_forward_hook(lambda _m, args, _out: calls.append(tuple(args[0].shape)))
            with torch.no_grad():
                output = model(public)
            hook.remove()
            assert output.shape == (32, 120, 3)
            assert len(calls) == (5 if arm == 'original4' else 1)
            if arm == 'new_first':
                assert calls[0][-1] == 128 and not hasattr(model, 'state_embed')
            else:
                with torch.random.fork_rng(devices=[]):
                    reference = ReferenceOriginal(width=64, layers=4, rectangle=True,
                                                  ffn_dropout=.1, residual_dropout=.05)
                reference.load_state_dict(model.state_dict())
                reference.cuda().eval()
                with torch.no_grad():
                    assert torch.equal(reference(public)[-1], output)
                del reference
            with torch.no_grad():
                for forbidden in ('target', 'reference'):
                    try:
                        model({**public, forbidden: batch[forbidden]})
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Model accepted label input')
                values = per_case(output, batch, arm)
                assert torch.equal(values['total'], base_per_case(output, batch)['total'])
                assert torch.equal(values['total'], per_case(output, batch, ARMS[1-ARMS.index(arm)])['total'])
                locked = batch['valid'] & batch['constraints'][..., :2].bool().any(-1)
                assert locked.any()
                altered = output.clone()
                altered[..., 0] += 1.
                altered[..., 2] += .2
                assert (decode(altered, batch)[locked]-decode(output, batch)[locked]).abs().sum() > 0
                twice = axis_swap(axis_swap({**batch, 'axis_swap': torch.ones(32, device='cuda', dtype=torch.bool)}))
                assert all(torch.equal(batch[key], twice[key]) for key in batch)
            leaves = output.detach().clone().requires_grad_(True)
            supervised = base_per_case(leaves, batch)
            (supervised['position']+supervised['shape']).mean().backward()
            assert leaves.grad[locked][..., :2].abs().sum() > 0
            assert leaves.grad[locked][..., 2].abs().sum() > 0
            del leaves, supervised, values, altered, twice
            model.train()
            optimizer = runner.optimizer_for(model)
            torch.cuda.reset_peak_memory_stats()
            cap = CapturedStep(model, optimizer, batch, arm)
            # cap.replay() calls Adam; replay only the forward/backward graph.
            cap.graph.replay()
            torch.cuda.synchronize()
            assert bool(cap.finite) and float(cap.max_case) < 1000.
            assert float(cap.values['total']) < 1000. and float(cap.norm) < 10000.
            assert model.layers[0].qkv.weight.grad.abs().sum() > 0
            assert model.layers[-1].qkv.weight.grad.abs().sum() > 0
            assert model.head[0].weight.grad.abs().sum() > 0
            if arm == 'new_first':
                assert model.layers[0].ff_gate.weight.grad.abs().sum() > 0
                assert model.layers[-1].relation_bias.coefficients.grad.abs().sum() > 0
            assert digest(model) == before
            assert not optimizer.state and cap.projection.steps == 0
            results[arm] = dict(parameters=count, total=float(cap.values['total']),
                max_case_loss=float(cap.max_case), gradient_norm=float(cap.norm),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                initial_parameters_sha256=before, parameters_unchanged=True,
                prediction_head_calls=len(calls), final_supervision_only=True,
                first_last_blocks_receive_gradients=True,
                cuda_graph_replayed_without_Adam=True, optimizer_updates=0)
            del model, optimizer, cap, output
            torch.cuda.empty_cache()
        assert not (OWNER/'runs').exists()
        result.update(status='passed', sources=sources(), arms=results,
            train_members=[list(r['member']) for r in rows],
            subset_local_indices=plan['preflight_train_indices'],
            actual_nodes=[r['graph']['n'] for r in rows], max_loss_ports_in_check=max_pins,
            common_initial_tensors=shared, common_initial_tensor_count=len(shared),
            common_global_CPU_RNG=True, launch_blocked_before_run_creation=blocked,
            all_module_prediction=True, no_MIB_input_or_loss=True,
            hard_specifications_not_overwriting_draft=True,
            batch32_FP32=True, original_arithmetic_reproduced=True,
            new_prediction_head_called_once=True,
            GPU=torch.cuda.get_device_name(0), torch=str(torch.__version__),
            check_scope='Only five unique frozen16k train members; no Adam step, run, checkpoint, dev/test evaluation or resume test')
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        save(OWNER/'preflight.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources', 'traceback', 'common_initial_tensors')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
