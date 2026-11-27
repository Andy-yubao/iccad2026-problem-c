"""Independent saved-evidence completion scripts; no changes to frozen runtime."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=ROOT/'experiments/E78_connectivity_posttrain'
source=ROOT/'experiments/E77_compact_refinement/runs/completion/verify_saved.py'
s=source.read_text(encoding='utf-8').replace('experiments.E77_compact_refinement','experiments.E78_connectivity_posttrain')
s=s.replace('20261029_e76_compact_corrector_seed0','20261029_e77_connectivity_seed0')
start=s.index("    charged=read(HERE/'runs/recovery/startup_interruption.json')")
end=s.index("    result['training_seen_count']",start)
s=s[:start]+"    result['training_seconds_note']='New phase2 stage wall including initialization; no interruption charge or inherited-stage reset.'\n"+s[end:]
s=s.replace("result['automatic_report_failed_only_serialization']=True","result['automatic_report_failed_only_serialization']=False")
needle="    save(HERE/'runs/completion/inspection.json',result)\n    print('weightedQ'"
assert needle in s
s=s.replace(needle,"""    previous77=ROOT/'experiments/E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0'
    result['paired_weighted_Q_vs_E77']=paired(flash['records'],read(previous77/'terminal/results.json')['records'],lambda r:r['metrics']['cost'],weighted=True)
    result['paired_overlap_vs_E77']=paired(raw['records'],read(previous77/'full_dev.json')['records'],lambda r:r['overlap_area_fraction'])
    result['paired_geometry_vs_E77']=review['paired_geometry_vs_E77']
    save(HERE/'runs/completion/inspection.json',result)
    print('weightedQ'""")
(HERE/'runs/completion/verify_saved.py').write_text(s,encoding='utf-8')
for name in ('analyze_saved.py','analyze_orders.py'):
    s=(HERE/'runs/diagnosis'/name).read_text(encoding='utf-8')
    if name=='analyze_saved.py':
        s=s.replace("PATHS={'E74'", "PATHS={'E78':'E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0','E74'")
    s=s.replace("'runs/diagnosis/", "'runs/completion/")
    if name=='analyze_orders.py':
        s=s.replace("print(json.dumps(result['tail_cases'],indent=2),flush=True)","print('tail164',result['tail_cases']['164'],flush=True)")
    (HERE/'runs/completion'/name).write_text(s,encoding='utf-8')
print('Completion checks prepared')
