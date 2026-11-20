"""Necessary five-member implementation gate; absolutely zero optimizer steps."""
import ast
import inspect
import json
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np
import torch
from datasets.random_reference import RandomReference
from . import data, runtime, train
from .addresses import address_rows
from .augmentation import axis_swap, flags_for
from .batches import inputs_only, decode
from .control import ROOT, OWNER, ARMS, read, sha, sources
from .cuda_step import CapturedStep
from .io import save
from .loss import per_case, base_per_case
from .model import build_model, relation_descriptor, PureBlock
from .resident import ResidentRows


def forbidden(*args, **kwargs):
    raise AssertionError('Preparation attempted a forbidden raw-data/train/update operation')


def check_descriptor(batch):
    raw = relation_descriptor(inputs_only(batch))
    valid, address = batch['valid'], batch['address']
    b, n = valid.shape
    assert raw.shape == (b, n, 400)
    torch.testing.assert_close(raw[..., :16], batch['features'] * valid[..., None], rtol=0, atol=0)
    for k in range(b):
        active = int(valid[k].sum())
        ids = address[k, :active]
        assert sorted(ids.tolist()) == list(range(active))
        expected_id = torch.eye(128)[ids]
        torch.testing.assert_close(raw[k, :active, 16:144], expected_id, rtol=0, atol=0)
        restored_wire = raw[k, :active, 144:272][:, ids]
        torch.testing.assert_close(restored_wire, batch['adjacency'][k, :active, :active], rtol=0, atol=0)
        labels = batch['constraints'][k, :active, 3]
        expected_group = ((labels[:, None] == labels[None, :]) & (labels[:, None] > 0)).float()
        restored_group = raw[k, :active, 272:400][:, ids]
        torch.testing.assert_close(restored_group, expected_group, rtol=0, atol=0)
        assert raw[k, active:].count_nonzero() == 0
        if active < 128:
            assert raw[k, :active, 144+active:272].count_nonzero() == 0
            assert raw[k, :active, 272+active:400].count_nonzero() == 0
    return raw


def main():
    started = time.perf_counter()
    result = dict(status='error', unique_train_cases=5, optimizer_updates=0,
                  real_optimizer_update_tested=False, training_started=False,
                  dev_tensor_payloads=0, raw_payload_reads=0, test_read=False,
                  postprocessing_calls=0, model_checkpoint_reads=0, resume_tested=False)
    try:
        for path in OWNER.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        plan = read(OWNER / 'plan.json')
        assert not read(OWNER / 'authorization.json')['training_enabled']
        assert (OWNER / 'STOP').is_file() and not (OWNER / 'runs').exists()
        blocked = {}
        for arm in ARMS:
            with patch.object(sys, 'argv', ['train', '--arm', arm]), patch.object(runtime, 'run', forbidden):
                try:
                    train.main()
                except PermissionError:
                    blocked[arm] = True
                else:
                    raise AssertionError('Training entry did not reject preparation-only state')
        result['launch_blocked_before_runtime'] = blocked
        for role in ('test', 'test_monitor500', 'dev'):
            try:
                data.load_rows(role)
            except PermissionError:
                pass
            else:
                raise AssertionError('Disallowed role accepted')
        result['cache_inventory_metadata_only'] = data.cache_inventory()
        subset, _ = data.registries()
        ids = plan['preflight_train_indices']
        parent_ids = [subset['train_parent_indices'][i] for i in ids]
        chunks = [data.cache_folder() / 'train' / f'{i//256:04d}.pt' for i in sorted(set(parent_ids))]
        chunks = sorted(set(chunks))
        before_cache = {p.relative_to(ROOT).as_posix(): sha(p) for p in chunks}
        with patch.object(RandomReference, '__getitem__', forbidden), patch.object(np, 'load', forbidden):
            rows = data.load_rows('train', ids)
        historical = read(ROOT / 'history/20261111_experiments_consolidation/experiments/E116_first_draft_comparison/preflight.json')
        assert [list(r['member']) for r in rows] == historical['train_members']
        assert len({tuple(r['member']) for r in rows}) == 5
        result['members'] = [list(r['member']) for r in rows]
        result['subset_local_indices'] = ids
        result['shared_cache_chunks'] = before_cache
        result['shared_cache_reused_without_rebuild'] = True
        runtime.seed_all(0)
        batch = ResidentRows(rows, 'cpu', max_nodes=120).batch(range(5))
        batch['address'] = address_rows([r['graph']['n'] for r in rows], ids, epoch=0)
        raw = check_descriptor(batch)
        renamed = dict(batch)
        renamed['address'] = address_rows([r['graph']['n'] for r in rows], ids, epoch=1)
        check_descriptor(renamed)
        identity = dict(batch)
        identity['address'] = address_rows([r['graph']['n'] for r in rows], ids)
        check_descriptor(identity)
        assert torch.equal(batch['address'], address_rows([r['graph']['n'] for r in rows], ids, epoch=0))
        group_renamed = dict(batch)
        group_renamed['constraints'] = batch['constraints'].clone()
        label = group_renamed['constraints'][..., 3]
        group_renamed['constraints'][..., 3] = torch.where(label > 0, label * 17 + 5, label)
        torch.testing.assert_close(relation_descriptor(inputs_only(group_renamed)), raw, rtol=0, atol=0)
        flip = dict(batch, axis_swap=torch.tensor([True, False, True, False, True]))
        twice = axis_swap(axis_swap(flip))
        for key in batch:
            torch.testing.assert_close(twice[key], batch[key], rtol=0, atol=0)
        invalid = dict(inputs_only(batch), target=batch['target'])
        try:
            relation_descriptor(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Label accepted by network')
        public = inputs_only(batch)
        assert not {'reference', 'target', 'pin_coordinates', 'wire_b2b'} & set(public)
        assert (batch['mask'][batch['valid']] == 1).all()
        assert list(inspect.signature(PureBlock.forward).parameters) == ['self', 'h', 'valid']
        rng_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        models = {arm: build_model(arm) for arm in ARMS}
        assert torch.equal(rng_before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        common = models['pure'].state_dict()
        assert all(torch.equal(value, models['full'].state_dict()[key]) for key, value in common.items())
        result['common_initial_tensor_count'] = len(common)
        result['common_initial_tensors_exact'] = True
        result['global_CPU_CUDA_RNG_preserved_by_build'] = True
        arms = {}
        for arm, model in models.items():
            count = sum(p.numel() for p in model.parameters())
            assert count == plan['parameters_by_arm'][arm]
            assert all(p.requires_grad for p in model.parameters())
            if arm == 'pure':
                assert all(not hasattr(layer, 'local') and not hasattr(layer, 'relation_bias') for layer in model.layers)
            calls = []
            hook = model.head.register_forward_hook(lambda *args: calls.append(1))
            model.eval()
            output = model(public)
            hook.remove()
            assert len(calls) == 1 and output.shape == (5, 120, 3)
            values = per_case(output, batch, arm)
            torch.testing.assert_close(values['total'], base_per_case(output, batch)['total'], rtol=0, atol=0)
            gradient = torch.autograd.grad(values['total'].sum(), output, retain_graph=True)[0]
            hard = batch['valid'] & ((batch['constraints'][..., 0] != 0) | (batch['constraints'][..., 1] != 0))
            assert gradient[hard].abs().sum() > 0
            p = decode(output, batch)
            assert (p[..., :2][hard] - batch['specs'][..., :2][hard]).abs().sum() > 0
            assert (p[..., 2:][hard] - batch['specs'][..., 2:][hard]).abs().sum() > 0
            values['total'].mean().backward()
            assert model.input[0].weight.grad.abs().sum() > 0
            assert all(layer.qkv.weight.grad.abs().sum() > 0 for layer in model.layers)
            with torch.no_grad():
                original = output.detach()
                permutation = torch.arange(119, -1, -1)
                permuted = {}
                for key, value in public.items():
                    if key in ('adjacency', 'group'):
                        permuted[key] = value[:, permutation][:, :, permutation]
                    elif key == 'scale':
                        permuted[key] = value
                    else:
                        permuted[key] = value[:, permutation]
                torch.testing.assert_close(model(permuted), original[:, permutation], rtol=2e-5, atol=2e-6)
                renamed_output = model(inputs_only(renamed))
                sensitivity = float((renamed_output - original)[batch['valid']].square().mean().sqrt())
            arms[arm] = dict(parameters=count, prediction_head_calls=1, all_six_blocks_receive_gradients=True,
                             CPU_final_loss=float(values['total'].mean().detach()),
                             initial_address_rename_output_RMS=sensitivity,
                             address_invariance_claimed=False)
            print(f'{arm}: CPU input/endpoint/gradient checks passed', flush=True)
        del models, model, output, values, gradient, p
        assert torch.cuda.is_available()
        result['GPU'] = torch.cuda.get_device_name(0)
        pack = ResidentRows(rows, 'cuda', max_nodes=120)
        repeated = [i % 5 for i in range(32)]
        gpu_batch = pack.batch(repeated)
        gpu_batch['address'] = address_rows([rows[i]['graph']['n'] for i in repeated], [ids[i] for i in repeated], epoch=0).cuda()
        flags = torch.as_tensor(flags_for(0, 0, np.asarray([ids[i] for i in repeated]), size=16000), device='cuda')
        with patch.object(torch.optim.AdamW, 'step', forbidden):
            for arm in ARMS:
                runtime.seed_all(0)
                model = build_model(arm).cuda().train()
                before = {k: v.detach().clone() for k, v in model.state_dict().items()}
                optimizer = runtime.optimizer_for(model)
                torch.cuda.reset_peak_memory_stats()
                cap = CapturedStep(model, optimizer, gpu_batch, arm)
                cap.inputs['axis_swap'].copy_(flags)
                values, norm = cap.backward_only()
                torch.cuda.synchronize()
                assert cap.projection.steps == 0 and not optimizer.state
                assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
                assert model.input[0].weight.grad.abs().sum() > 0
                assert all(layer.qkv.weight.grad.abs().sum() > 0 for layer in model.layers)
                try:
                    cap.replay()
                except PermissionError:
                    pass
                else:
                    raise AssertionError('Preparation capture allowed an optimizer update')
                arms[arm].update(CUDA_graph_forward_backward_passed=True, batch_size=32,
                    total=float(values['total']), max_case_loss=float(cap.max_case),
                    unclipped_gradient_norm=float(norm), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    parameters_unchanged=True, optimizer_state_empty=True, optimizer_updates=0)
                print(f'{arm}: FP32 batch32 CUDA graph backward passed, zero optimizer updates', flush=True)
                del cap, optimizer, model, before, values
                torch.cuda.empty_cache()
        for name, digest in before_cache.items():
            assert sha(ROOT / name) == digest
        assert not (OWNER / 'runs').exists() and (OWNER / 'STOP').exists()
        assert not read(OWNER / 'authorization.json')['training_enabled']
        result.update(status='passed', arms=arms, sources=sources(),
                      descriptor_relations_reconstructed_exact=True,
                      endpoint_renaming_consistent=True, group_label_rename_invariant=True,
                      input_row_permutation_with_addresses_equivariant=True,
                      labels_excluded=True, no_F_P_overwrite=True, no_geometric_feedback=True,
                      pure_blocks_have_no_relation_arguments=True, cache_unchanged=True,
                      torch=str(torch.__version__))
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        save(OWNER / 'preflight.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'traceback', 'shared_cache_chunks')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
