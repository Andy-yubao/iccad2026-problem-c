"""One constructed public-input symmetry counterexample, not a dataset frequency audit."""
import numpy as np
import torch
from .io import ROOT, HERE, save
from .model import GeometryModel, decode
from .batches import sample, collate, inputs_only


def main():
    torch.set_num_threads(4)
    n = 32
    # A cycle with identical module attributes is vertex-transitive. One shared
    # pin connected identically to every module preserves this automorphism.
    inputs = (n, np.ones(n), np.array([[i,(i+1)%n,1.] for i in range(n)]),
              np.array([[0,i,1.] for i in range(n)]), np.array([[4.,2.]]),
              np.zeros((n,5)), np.full((n,4),-1.))
    reference = torch.tensor([[float(i%8),float(i//8),1.,1.] for i in range(n)])
    graph = collate([sample(inputs, reference)], [0], 'cpu')
    state = torch.load(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = GeometryModel(**state['model_config']).eval(); model.load_state_dict(state['model'])
    with torch.no_grad():
        raw = model(inputs_only(graph))
        positions = (decode(raw, graph)[0]*graph['scale'][0]).numpy()
    delta = float((raw-raw[:,:1]).abs().max())
    lo = np.maximum(positions[0,:2],positions[1,:2]); hi = np.minimum(positions[0,:2]+positions[0,2:],positions[1,:2]+positions[1,2:])
    area = float(np.maximum(hi-lo,0).prod()); assert area > .99
    save(HERE/'runs/symmetry_counterexample/results.json', {
        'constructed_case': True, 'module_count':n, 'actual_dataset_cases_read':0, 'test_read':False,
        'raw_state_difference_max':delta, 'overlap_area':area,
        'numerical_note':'The initial 1e-6 bitwise-symmetry tolerance failed on this ring because floating-point reductions need not be exactly permutation-equivariant. The actual diagnostic criterion is overlapping positive-area outputs, not bitwise equality.',
        'predicted_positions':positions.tolist(), 'valid_reference_positions':reference.tolist(),
        'conclusion':'In exact arithmetic, permutation equivariance forces identical states for automorphic nodes. Finite precision can produce small differences, but E42 still overlaps on this admissible constructed case despite a non-overlapping reference. This exposes a symmetry limitation, without quantifying its frequency or explaining observed typical-case error by itself.'})
    print('Constructed symmetry limitation verified', flush=True)


if __name__ == '__main__':
    main()
