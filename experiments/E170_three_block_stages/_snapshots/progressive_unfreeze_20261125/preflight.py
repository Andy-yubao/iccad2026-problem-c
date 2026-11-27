"""Disposable CUDA checks for the protected and largest joint E170 phases."""
from copy import deepcopy
import gc
import json
import time

import torch

from datasets.random_reference import RandomReference
from .base_resident import ResidentRows as BaseRows
from .data import load_rows
from .io import HERE, ROOT, save, sha
from .loss_config import PLAN, PROTECTED_EPOCHS
from .model import GeometryModel
from .public import inputs_only
from .resident import ResidentRows
from .schedule import LR, PlateauController
from .train import BATCH_SIZE, build_capture, optimizer_for, seed_all, unfreeze


def check_schedule():
    for stage in range(1, 5):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=LR[0])
        scheduler = PlateauController(optimizer, stage)
        for epoch in range(15, PROTECTED_EPOCHS + 1):
            scheduler.step(1., epoch)
        scheduler.best = None
        scheduler.lr_bad = scheduler.early_bad = 0
        for epoch in range(PROTECTED_EPOCHS + 1, 50):
            scheduler.step(1., epoch)
            if scheduler.should_stop():
                break
        if not scheduler.should_stop() or optimizer.param_groups[0]['lr'] != (LR[-1] if stage == 4 else 1e-4):
            raise ValueError('E170 LR floor or stopping rule changed')


def probe(model, pack, joint):
    optimizer = optimizer_for(model)
    if joint:
        unfreeze(model, optimizer)
        assert optimizer.param_groups[-1]['lr'] == optimizer.param_groups[0]['lr'] * .2
    torch.cuda.reset_peak_memory_stats()
    before = time.perf_counter()
    capture = build_capture(model, optimizer, pack)
    capture.set_case_count(len(pack.rows))
    prefix = {name: value.detach().clone() for name, value in model.prefix_state().items()}
    values, norm = capture.replay()
    torch.cuda.synchronize()
    if not joint and not all(torch.equal(value, prefix[name]) for name, value in model.prefix_state().items()):
        raise ValueError('Frozen prefix changed on the disposable AdamW step')
    free, _ = torch.cuda.mem_get_info()
    result = dict(stage=model.stage, joint=joint, batch=BATCH_SIZE, active=len(pack.rows),
                  loss=float(values['total']), gradient_norm=float(norm),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes=free,
                  capture_and_update_seconds=time.perf_counter() - before)
    if free < 512 * 2**20:
        raise RuntimeError('Insufficient dedicated VRAM headroom')
    print(json.dumps(result), flush=True)
    return result


def main():
    seed_all(PLAN['seed'])
    check_schedule()
    parent = PLAN['parent_initial']
    checkpoint = ROOT / parent['checkpoint']
    if sha(checkpoint) != parent['checkpoint_sha256']:
        raise ValueError('E165 L0 checksum changed')
    if sha(HERE / 'initial_model.py') != sha(ROOT / 'experiments/E165_joint_l0/initial_model.py'):
        raise ValueError('E170 L0 implementation differs from E165')
    pretrained = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if pretrained['identity']['arm'] != parent['arm'] or pretrained['progress']['best_epoch'] != parent['best_epoch']:
        raise ValueError('Unexpected E165 L0 best')
    monitor = RandomReference('train_monitor500')
    indices = monitor.indices[:5]
    rows = load_rows('train', indices=indices)
    if [list(row['member']) for row in rows] != [monitor.records[i]['member'] for i in indices]:
        raise ValueError('Preflight member order changed')

    # Reuse the measured maximum train dimensions from E165 on the identical
    # members and preprocessing, including the structural-role port padding.
    previous = json.loads((ROOT / 'experiments/E165_joint_l0/runs/e165_backend_seed0/resources.json').read_text())
    max_pins = previous['max_loss_ports_per_block']
    base = BaseRows(rows, 'cuda', max_nodes=120, max_pins=max_pins)
    fixed_roles = ResidentRows(rows, 'cuda', max_nodes=120, max_pins=max_pins, max_ports=1)
    bytes_per_port = 1217
    max_ports = 1 + (previous['row_bytes'] - fixed_roles.row_bytes) // bytes_per_port
    assert max_ports > 0 and fixed_roles.row_bytes + (max_ports - 1) * bytes_per_port == previous['row_bytes']
    assert base.max_pins == max_pins
    pack = ResidentRows(rows, 'cuda', max_nodes=120, max_pins=max_pins, max_ports=max_ports)
    public = inputs_only(pack.batch(range(len(rows))))
    cpu_prefix = pretrained['model']
    reference = GeometryModel(1).cuda()
    reference.load_prefix(cpu_prefix)
    reference.eval()
    with torch.no_grad():
        initial_only = reference.initial_generator(public)[0]
        full_initial = reference(public)[0]
    if not torch.equal(full_initial, initial_only):
        raise ValueError('Frozen S0 differs from the E165 L0 output')
    s1 = probe(reference, pack, joint=False)
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    for stage in range(1, 5):
        model = GeometryModel(stage).cuda()
        model.load_prefix(cpu_prefix)
        cpu_prefix = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    model.shape_neighbor_count = 120
    check = deepcopy(model)
    check.unfreeze()
    geometry_calls = [0] * 4

    def on_geometry(index):
        def count(*_):
            geometry_calls[index] += 1
        return count

    handles = [unit.geometry_attention.register_forward_hook(on_geometry(i)) for i, unit in enumerate(check.stages)]
    outputs = check(public)
    parents = [check.initial_generator.head[-1].weight, *(unit.head.position.weight for unit in check.stages[:-1])]
    gradients = torch.autograd.grad(outputs[-1].square().mean(), parents)
    if geometry_calls != [1] * 4 or not all(bool(torch.isfinite(grad).all() & (grad.abs().max() > 0)) for grad in gradients):
        raise ValueError('Expected one geometry block per stage and final-layout gradients to all earlier outlets')
    for handle in handles:
        handle.remove()
    del check, outputs, parents, gradients
    gc.collect()
    torch.cuda.empty_cache()
    s4 = probe(model, pack, joint=True)
    save(HERE / f'preflight_batch{BATCH_SIZE}.json',
         dict(status='passed', members=[list(row['member']) for row in rows], parent_sha256=sha(checkpoint),
              s0_matches=True, geometry_calls=geometry_calls, gradient_to_prefix=True, stage_probes=[s1, s4],
              max_nodes=120, max_loss_ports_per_block=max_pins, max_role_ports=max_ports,
              gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))


if __name__ == '__main__':
    main()
