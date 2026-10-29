"""Finalize changed hypothesis before preflight/formal training, retaining rejected proposal."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/E78_connectivity_posttrain'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=read(HERE/'plan.json');write(HERE/'runs/diagnosis/rejected_group_wire_plan.json',p)
p.update(fixed_loss='G+2I+5E+.25*anchored_corridor, where corridor=mean(square(normalized deficit))+.25*max(square deficit),16 ordinary max-path relaxations; fixed from epoch0',
    architecture='E77 full warm weights; only existing21860-parameter corrector trainable. E76 source frozen/eval, E74 public frame retained. Equal-public-area compatible MIB groups share log-aspect; public locked shape governs group when consistent.',
    initial_function='Public MIB-coherent projection of E77, not exact unprojected E77 function. All pretrained weights reused unchanged before optimization.',
    hypothesis='Saved worst traces show primary shape iterations illegal and packed rescue; MIB preparation changes predicted dimensions, and predicted pair orders overfill anchored corridors. Reduce this preparation mismatch and fixed-shape repair pressure together.',
    rejected_proposal='Raw public-group logdet and b2b excess alone: group-proxy change/final grouping-change correlation .00985, raw worst164 b2b ratio1.0068 but returned1.660; no formal training of that proposal.',
    group_goal='Do not invent new group-connectivity guarantee. Encourage coherent MIB geometry and anchor-compatible packing while original contact/group/boundary supervision remains.',
    wire_goal='No new raw b2b term: saved raw wire already close to reference. The rejected implementation remains diagnostic evidence only.',
    limits='MIB tying initially raises intersection in4/5 train probes; learning must recover the tradeoff. Corridor pressure is truncated fixed-shape order diagnostic, not exact SOCP feasibility. Discrete axis/order choices have no straight-through gradient. Free-shape repair may succeed despite positive pressure; frame may be infeasible.')
write(HERE/'plan.json',p)
p=read(HERE/'provenance.json');p['copy_note']='Independent E77 runtime; full E77 best38 weights reused. New compatible equal-area public MIB log-aspect tying plus fixed anchored-corridor pressure; same E76 frozen source and E74 frame, only existing last corrector fine-tuned.';write(HERE/'provenance.json',p)
path=HERE/'initialization.py';s=path.read_text(encoding='utf-8').replace("'mode':'Exact E77 best38; fine-tune same trained corrector under new fixed Loss'","'mode':'Full E77 best38 weights; public MIB projection and new fixed corridor Loss; same trained corrector'")
s=s.replace("'optimizer':'Fresh AdamW only existing last corrector, source state unchanged',","'optimizer':'Fresh AdamW only existing last corrector, source state unchanged',\n        'initial_function':'MIB-coherent projection of E77; not exact unprojected function',")
path.write_text(s,encoding='utf-8')
path=HERE/'loss_config.py';s=path.read_text(encoding='utf-8').replace('+.25 anchored corridor mean-square+.25max-square pressure','+.25*(anchored corridor mean-square+.25max-square) pressure');path.write_text(s,encoding='utf-8')
path=HERE/'train.py';s=path.read_text(encoding='utf-8')
s=s.replace("REVISION = 'E78.frozen_e76_frame_corrector.v1'","REVISION = 'E78.e77_MIB_anchored_posttrain.v1'")
s=s.replace("('common_original_G', 'center_local'","('common_original_G', 'anchored_corridor', 'center_local'")
s=s.replace('Entire E76 frozen/eval; new corrector dropout .05/decay .0003; hard public-predicted E74 frame','E76 source frozen/eval; existing E77 last corrector dropout.05/decay.0003; equal-area public MIB tying and E74 frame')
s=s.replace('G+2 true intersection/public total area+5 positive log-box expansion squared','G+2I+5E+.25 anchored corridor mean-plus-max squared pressure')
s=s.replace('new-corrector Adam; E76 frozen; preprojection residual bounded.15 and final E74 frame','existing E77 last-corrector Adam; E76 frozen; residual bound.15, MIB coherent shape and E74 frame')
path.write_text(s,encoding='utf-8')
path=HERE/'report.py';s=path.read_text(encoding='utf-8')
s=s.replace("'violations':sum(v>3e-6 for v in protrusions)","'violations':int(sum(v>3e-6 for v in protrusions))")
needle="    save(HERE/'runs/review.json',study);"
assert needle in s
s=s.replace(needle,"""    e77_path=ROOT/'experiments/E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0'
    e77=read(e77_path/'terminal/results.json')['records']
    assert [r['member'] for r in records]==[r['member'] for r in e77]
    q77=np.array([r['metrics']['cost'] for r in e77]);d77=q-q77
    study['E77_baseline_flash']=read(e77_path/'terminal/summary.json')
    study['paired_Q_vs_E77']={'better_equal_worse':[int((d77< -1e-8).sum()),int((abs(d77)<=1e-8).sum()),int((d77>1e-8).sum())],'mean_delta':float(d77.mean())}
    study['paired_geometry_vs_E77']=compare(current,[{'index':r['index'],'member':r['member'],**geometry(r['draft_positions'])} for r in e77])
    save(HERE/'runs/review.json',study);""")
s=s.replace('# E78固定预测外框＋内部局部修正结果','# E78 MIB一致输出＋锚点通道后训练结果')
start=s.index('从E76 own-Loss best29后训练')
end=s.index("\n\n完成",start)
s=s[:start]+'从E77 own-Loss best38沿用完整权重，继续训练其既有最后局部修正器；E76源全冻结/eval和E74公开预测外框保持。相容且等公开面积MIB组共享log长宽比。固定Loss为G+2I+5E+.25锚点通道过填平方压力（16次max-path松弛）；训练/监测/选模完全一致。初始输出是MIB一致投影的E77，不能宣称未投影函数保持；组合不能单独归因。'+s[end:]
path.write_text(s,encoding='utf-8')
path=HERE/'render_drafts.py';s=path.read_text(encoding='utf-8').replace('frozen E76 internal correction in public-predicted E74 frame','MIB-coherent E77 posttrain in public E74 frame');path.write_text(s,encoding='utf-8')
print('E78 hypothesis and fixed objective finalized before preflight')
