"""Independent E42-compatible feature preparation; supervision stays out of inputs."""
import torch
from datasets.reference_layouts import public_hard_reference
from .batches import sample, collate as batch_collate


def prepare(item):
    case=item['case'];graph=sample(case.solver_inputs,torch.as_tensor(item['positions']).float())
    case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
    return {'graph':graph,'member':item['member'],'case':case}


def collate(rows,device):
    return batch_collate([x['graph'] for x in rows],list(range(len(rows))),device),None
