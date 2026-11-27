"""Adapt immutable-result audit for fixed-shape E79 without re-evaluation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=ROOT/'experiments/E79_center_refinement'
s=(ROOT/'experiments/E78_connectivity_posttrain/runs/completion/verify_saved.py').read_text(encoding='utf-8')
s=s.replace('experiments.E78_connectivity_posttrain','experiments.E79_center_refinement')
s=s.replace('20261029_e77_connectivity_seed0','20261029_e78_center_seed0')
s=s.replace("parent=ROOT/'experiments/E76_bounded_refinement/runs/20261029_frozen_e74_corrector_seed0'", "parent=ROOT/'experiments/E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0'")
s=s.replace('paired_weighted_Q_vs_E76','paired_weighted_Q_vs_E78').replace('paired_overlap_vs_E76','paired_overlap_vs_E78')
needle="    save(HERE/'runs/completion/inspection.json',result)\n    print('weightedQ'"
assert needle in s
s=s.replace(needle,"""    result['paired_geometry_vs_E78']=review['paired_geometry_vs_E78']
    result['source_shape_deltas500']=review['saved_source_shape_deltas500']
    result['source_R_mean']=review['full_dev']['topology_regression']
    result['source_topology_flip_fraction_mean']=review['full_dev']['topology_flip_fraction']
    save(HERE/'runs/completion/inspection.json',result)
    print('weightedQ'""")
(HERE/'runs/completion/verify_saved.py').write_text(s,encoding='utf-8')
print('E79 audit prepared')
