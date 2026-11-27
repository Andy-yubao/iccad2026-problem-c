"""Keep frozen evaluation sources intact; fix inherited figure caption only."""
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
text=(HERE/'render_drafts.py').read_text(encoding='utf-8')
text=text.replace('from .dataset_view import','from experiments.E80_modular_transfer.dataset_view import')
text=text.replace('from .io import','from experiments.E80_modular_transfer.io import')
text=text.replace('frozen coherent E78 plus center correction in public E74 frame',
                  'E42 placer plus frozen E79 correction package in public E42 frame')
text=text.replace("if (target/'manifest.json').exists():return",'# Regenerate saved-coordinate gallery to correct inherited caption.')
(HERE/'runs/render_transfer_gallery.py').write_text(text,encoding='utf-8')
