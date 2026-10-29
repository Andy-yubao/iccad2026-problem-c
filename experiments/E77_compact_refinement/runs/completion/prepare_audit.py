"""Adapt prior saved-evidence audit to E77, preserving completed algorithms."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
source=ROOT/'experiments/E76_bounded_refinement/runs/completion/verify_saved.py'
target=Path(__file__).with_name('verify_saved.py')
assert not target.exists()
s=source.read_text(encoding='utf-8')
s=s.replace('experiments.E76_bounded_refinement','experiments.E77_compact_refinement')
s=s.replace('20261029_frozen_e74_corrector_seed0','20261029_e76_compact_corrector_seed0')
s=s.replace("parent=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0'",
    "parent=ROOT/'experiments/E76_bounded_refinement/runs/20261029_frozen_e74_corrector_seed0'")
s=s.replace("'backbone.'","'source.'")
s=s.replace('frozen_encoder_tensors_bit_exact','frozen_source_tensors_bit_exact')
s=s.replace('paired_weighted_Q_vs_E74','paired_weighted_Q_vs_E76')
s=s.replace('paired_overlap_vs_E74','paired_overlap_vs_E76')
needle="    print('weightedQ',q,'overlap2000',overlap,'G',result['common_G2000'],flush=True)"
assert needle in s
s=s.replace(needle,"""    previous74=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0'
    result['paired_weighted_Q_vs_E74']=paired(flash['records'],read(previous74/'terminal/results.json')['records'],lambda r:r['metrics']['cost'],weighted=True)
    result['paired_overlap_vs_E74']=paired(raw['records'],read(previous74/'full_dev.json')['records'],lambda r:r['overlap_area_fraction'])
    review=read(HERE/'runs/review.json')
    result['paired_geometry_vs_E74']=review['paired_geometry_vs_E74']
    result['paired_geometry_vs_E76']=review['paired_geometry_vs_E76']
    result['paired_geometry_vs_E42_gallery']=review['paired_geometry_vs_E42_gallery']
    result['public_frame_containment500']=review['saved_public_frame_containment500']
    result['train_improvement_fraction']=review['train_improvement_fraction']
    result['dev_improvement_fraction']=review['dev_improvement_fraction']
    result['train_dev_relative_gap']=review['train_dev_relative_gap']
    result['best_train_dev']=[review['best_monitor']['train_monitor']['total'],review['best_monitor']['dev_monitor']['total']]
    result['initial_train_dev']=[review['initial_monitor']['train_monitor']['total'],review['initial_monitor']['dev_monitor']['total']]
    charged=read(HERE/'runs/recovery/startup_interruption.json')['charged_interruption_seconds']
    result['budget_charged_interruption_seconds']=charged
    result['measured_resumed_training_stage_seconds']=result['training_seconds']-charged
    result['training_seconds_note']='Budget total includes unknown startup interruption conservatively charged; resumed-stage measurement excludes that charge.'
    result['training_seen_count']=latest['progress']['seen'].sum().item()
    result['terminal_hard_legal']=read(folder/'terminal/summary.json')['hard_legal']
    result['raw_hard_legal']=raw['losses']['raw_hard_legal']
    result['automatic_report_failed_only_serialization']=True
    result['report_recovered_without_production_source_change_or_re_evaluation']=True
    save(HERE/'runs/completion/inspection.json',result)
    print('weightedQ',q,'overlap2000',overlap,'G',result['common_G2000'],flush=True)""")
target.write_text(s,encoding='utf-8')
print('Prepared',target)
