from pathlib import Path
import json,hashlib
p=Path('experiments/E107_geometry_posttrain');src=Path('experiments/E105_gated_prenorm_placer')
(p/'preflight.py').write_text('from .shortcheck import main\n\nif __name__=="__main__":main()\n',encoding='utf-8')
# Record the unchanged data/model/augmentation/regularization and explicit stage changes.
files=['model.py','features.py','batches.py','data.py','cache.py','dataset_view.py','ports.py','resident.py','augmentation.py','simple_loss.py','group_tree.py','step_control.py','captured_projection.py','predictor.py']
records=[]
for name in files:
    a=hashlib.sha256((src/name).read_bytes()).hexdigest();b=hashlib.sha256((p/name).read_bytes()).hexdigest();assert a==b,name;records.append(dict(file=name,sha256=a,identical=True))
base=json.loads((src/'plan.json').read_text(encoding='utf-8'));plan=json.loads((p/'plan.json').read_text(encoding='utf-8'))
fields=['architecture','parameters','train_count','train_monitor_count','dev_monitor_count','data','batch_size','precision','seed','threads','maximum_seconds','maximum_epochs','optimizer','augmentation','network_ports']
for key in fields:assert plan[key]==base[key],key
alignment=dict(status='passed',files=records,plan_fields={k:plan[k] for k in fields},changes=['warm start E105 best58','new geometry/ref10 objective and frozen train500 scales','fresh AdamW and LR1e-4->5e-5 without protection','frozen parent and clean student trust forward','strict per-batch absolute and relative explosion abort'],test_read=False)
(p/'non_loss_alignment.json').write_text(json.dumps(alignment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan['geometry_loss']['auxiliary_penetration_weight']=.05
plan['minimum_lowest_lr_complete_epochs']=2
plan['normalization_sha256']=hashlib.sha256((p/'normalization.json').read_bytes()).hexdigest()
plan['non_loss_alignment_sha256']=hashlib.sha256((p/'non_loss_alignment.json').read_bytes()).hexdigest()
(p/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Rewrite only E107 report interpretation and bind its E105 parent baseline.
f=p/'evaluate.py';s=f.read_text(encoding='utf-8').replace("BASE=ROOT/'experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0'","BASE=ROOT/'experiments/E105_gated_prenorm_placer/runs/20261105_gated128x16_seed0'")
s=s.replace("old_best=min(old_history,key=lambda h:h['dev_monitor']['total'])", "old_best=history[0]  # E105 parent re-evaluated with the same new Loss at epoch0")
s=s.replace('E104 32k / 128x16','E105 parent / new Loss').replace('E105 32k / 128x16','E107 posttrain / new Loss')
s=s.replace("for hs,name,count in [(old_history,'E105 parent / new Loss',32000),(history,'E107 posttrain / new Loss',32000)]:", "for hs,name,count in [(history,'E107 stage (epoch0 = E105 parent)',32000)]:")
s=s.replace('E105方案一终末评价','E107几何目标ref10后训练终末评价').replace('E104修复后best81已完成500结果只读复用','E105父模型best58已完成500结果只读复用；Loss列使用本阶段epoch0的新Loss基准，非原训练Loss').replace('E104修复版32k / 128×16|E105方案一32k / 128×16','E105父模型（新Loss基准）|E107后训练')
lines=s.splitlines()
for i,line in enumerate(lines):
    if "text+='正常dev早停" in line:lines[i]="    text+='本阶段从E105 best58继续，原参考位置/形状权重改为10%，几何三项主导且加入软约束/线长保护与局部漂移约束；各项固定不调权，train/dev/best/LR/早停共用同一新Loss。训练Loss爆炸任一门禁即在Adam前拒绝并写STOP，只有正常早停才进入本评价。草稿几何与后处理Q分开，父模型结果只读复用，不按Q换checkpoint或再训练。dev已反复开发使用、单seed，不称正式比赛或独立未见泛化；测试封存。\\n\\n'"
s='\n'.join(lines)+'\n'
s=s.replace('E105 scheme1 / frozen Loss-best / depth v1','E107 geometry ref10 / frozen new-Loss best / depth v1').replace("('raw','E105 raw')","('raw','E107 raw')").replace('<title>E105 dev500</title>','<title>E107 dev500</title>')
# Exact raw metrics are always diagnostic, outside end-to-end solve timing.
s=s.replace('from .metrics import module_overlap','from .metrics import module_overlap,union_area')
s=s.replace("raw_overlap=module_overlap(draft),draft_positions", "raw_overlap=module_overlap(draft),draft_positions")
a=s.index('    return dict(count=len(records),weighted_Q=wq');b=s.index('\ndef report(',a)
old=s[a:b];old=old.replace('    return dict(count=', '    result=dict(count=')
old += '''    area=[];blank=[];n10=[]
    for record in records:
        p=np.asarray(record['draft_positions'],dtype=float);total=float(p[:,2:].prod(-1).sum());union=union_area(p)
        bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
        area.append(max(0.,1-union/total));blank.append(max(0.,1-union/bbox));n10.append(record['raw_overlap']['N10'])
    result.update(mean_draft_area_excess=float(np.mean(area)),mean_draft_whitespace=float(np.mean(blank)),mean_draft_N10=float(np.mean(n10)))
    return result

'''
s=s[:a]+old+s[b:]
s=s.replace("comparison=dict(baseline=baseline,current=current", "parent_records=[json.loads((BASE/f'evaluation_depth_dev500/cases/{i:04d}.json').read_text(encoding='utf-8')) for i in range(500)]\n    parent_geometry=summarize(parent_records)\n    comparison=dict(baseline=baseline,parent_geometry=parent_geometry,current=current")
s=s.replace("for name,left,right in pairs:text+=", "pairs += [(key,parent_geometry[key],current[key]) for key in ('mean_draft_area_excess','mean_draft_whitespace','mean_draft_N10')]\n    for name,left,right in pairs:text+=")
# If the new-Loss best is exactly epoch0 parent, reuse its existing500 solves.
s=s.replace("del state\n    OUT.mkdir", "parent_state=torch.load(HERE/plan['parent_checkpoint'],map_location='cpu',weights_only=False)\n    reuse_parent=all(torch.equal(v,parent_state['model'][k]) for k,v in state['model'].items())\n    del parent_state,state\n    OUT.mkdir")
s=s.replace("record=request(predictor,row);records.append(record)", "if reuse_parent:\n                record=json.loads((BASE/f'evaluation_depth_dev500/cases/{i:04d}.json').read_text(encoding='utf-8'));record['reused_parent_result']=True\n            else:record=request(predictor,row)\n            records.append(record)")
s=s.replace("current.update(status='metrics_complete'", "current.update(parent_result_reuse=reuse_parent,status='metrics_complete'")
f.write_text(s,encoding='utf-8')
