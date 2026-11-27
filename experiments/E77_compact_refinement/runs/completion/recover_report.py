"""Serialize saved-result report scalars; no training/inference/scoring rerun."""
import json
import numpy as np
from experiments.E77_compact_refinement import report
from experiments.E77_compact_refinement.io import save,sha,HERE

def compatible(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)

def normalized_save(path,value):
    save(path,json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False,default=compatible)))

def main():
    original=report.save;report.save=normalized_save
    try:report.main()
    finally:report.save=original
    save(HERE/'runs/completion/report_recovery.json',{
        'cause':'NumPy int32 count in containment violations was not JSON serializable',
        'repair':'Read-only saved-result report; normalize NumPy scalars at serialization boundary',
        'production_report_sha256_unchanged':sha(HERE/'report.py'),
        'original_error_preserved':'runs/report.log','training_rerun':False,'inference_rerun':False,
        'solver_evaluation_rerun':False,'checkpoint_reselection':False,'test_read':False})

if __name__=='__main__':main()
