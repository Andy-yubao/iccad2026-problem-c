"""Copy the completed-parent audit independently, adapt explicit source identities."""
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
t=(ROOT/'experiments/E81_multisource_refinement/runs/audit_saved.py').read_text(encoding='utf-8')
t=t.replace('from experiments.E81_multisource_refinement.','from experiments.E82_global_pressure_refinement.')
t=t.replace("OUT=HERE/'runs/20261029_e42_multisource_seed0'","OUT=HERE/'runs/20261029_e81_pressure_seed0'")
t=t.replace("parents=provenance()['parents'];", "parents=read(ROOT/'experiments/E81_multisource_refinement/provenance.json')['parents'];")
t=t.replace("e80path=ROOT/'experiments/E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer'", "e80path=ROOT/'experiments/E81_multisource_refinement/runs/20261029_e42_multisource_seed0'")
t=t.replace('vs_E80','vs_E81')
needle="    deployment=torch.load(OUT/'deployment.pt'"
at=t.index(needle)
extra="""    parent81=torch.load(ROOT/provenance()['initialization_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    for k,v in parent81['model'].items():
        assert torch.equal(v,best['model'][k]) and torch.equal(v,latest['model'][k]),k
    assert len(parent81['model'])==497
"""
t=t[:at]+extra+t[at:]
t=t.replace("'sources_unchanged':True,", "'sources_unchanged':True,'frozen_E81_tensors_bit_exact_best_and_latest':497,'trainable_pressure_parameters':2304,")
(HERE/'runs/audit_saved.py').write_text(t,encoding='utf-8')
