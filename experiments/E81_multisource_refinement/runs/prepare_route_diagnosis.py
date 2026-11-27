"""Independent copy of the saved-geometry diagnostic; no producer source edits."""
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
text=(ROOT/'experiments/E80_modular_transfer/runs/diagnose_route.py').read_text(encoding='utf-8')
text=text.replace('from experiments.E80_modular_transfer.','from experiments.E81_multisource_refinement.')
text=text.replace("'E80':'E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer'}",
                  "'E80':'E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer','E81':'E81_multisource_refinement/runs/20261029_e42_multisource_seed0'}")
(HERE/'runs/diagnose_route.py').write_text(text,encoding='utf-8')
