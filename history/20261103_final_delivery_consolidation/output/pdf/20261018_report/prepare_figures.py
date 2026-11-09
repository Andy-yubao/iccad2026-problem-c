"""Communication figures from saved development evidence only; no inference/data reads."""
from pathlib import Path
import sys, json, csv, copy, hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
AS=OUT/'assets'; AS.mkdir(exist_ok=True)
H=ROOT/'history/20261018_E10_consolidation/experiments'
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
refpath=H/'intent_aware_repair/runs/20261017_geometry_only_factorial_seed0/evaluation/cases.jsonl'
refs=rows(refpath)
finalpath=H/'E8_fast_robust_postprocess/runs/20261018_robust_dev100_seed0/cases.jsonl'
finals={r['case_id']:r for r in rows(finalpath)}
draftpath=H/'E8_fast_robust_postprocess/runs/20261018_stageA_dev100_seed0/drafts.jsonl'
drafts={r['case_id']:r for r in rows(draftpath)}
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False})
lbpath=ROOT/'references/official/results/C_beta_leaderboard_20260823.csv'
lb=list(csv.DictReader(lbpath.open(encoding='utf-8')))
fig,ax=plt.subplots(figsize=(9.5,3.2))
vals=[float(r['raw_score']) for r in lb]
ax.bar(range(1,11),vals,color='#96b9d0',width=.65)
ax.axhspan(1.211556,1.232217,color='#c45037',alpha=.20,label='本地两组开发面板均值 1.212 / 1.232')
ax.axhline(1.2218865,color='#b33f28',ls='--',lw=1)
ax.set(xticks=range(1,11),xlabel='官方综合排名（Beta，2026-08-23）',ylabel='质量分 / raw_score（越低越好）',ylim=(1,1.58))
for i,v in enumerate(vals,1): ax.text(i,v+.009,f'{v:.3f}',ha='center',fontsize=9)
ax.legend(loc='upper left',frameon=False,fontsize=9)
fig.text(.5,.015,'官方为隐藏集、块数加权汇总；本地为开发集算术均值。不同口径，仅作水平参照。',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.045,1,1));fig.savefig(AS/'leaderboard.png',dpi=190);plt.close(fig)
from visualization.layouts import plot_case, overlap_blocks, geometry_details
def variant(pos,metrics=None):
    return {'positions':pos,'metrics':metrics or {}}
def render(row,path,stages):
    for v in row['variants'].values():
        v['soft_details']=geometry_details(v['positions'],row['constraints'])
    return plot_case(row,path,stages=stages)
median=sorted(finals.values(),key=lambda r:r['variants']['conic_robust']['final']['metrics']['cost'])[49]
r=next(r for r in refs if r['case_id']==median['case_id'])
render({'case_id':r['case_id'],'panel':'Saved dev100; illustrative median-Q case','constraints':r['public_constraints'],
 'variants':{'reference':variant(r['reference'],r['reference_metrics']),
 'raw':variant(drafts[r['case_id']]['draft']),
 'final':median['variants']['conic_robust']['final'] }},AS/'layout_progress.png',stages=(('reference','Official reference'),('raw','Model proposal'),('final','E10 / E8 conic_robust')))
boundpath=H/'E9_model_postprocess_alignment/runs/20261018_official_quality_bound_dev100_seed0_r2/results.json'
bound=json.loads(boundpath.read_text(encoding='utf-8'))
# Try a direct translation of each free block to its required edge(s). Only
# successful hard-legal translations are counted, with original shapes intact.
candidates=[]; total_fail=0; near_fail=0; fail_cases=set(); near_cases=set()
for r in refs:
    p=r['reference']; flags=r['public_constraints']
    xmin=min(q[0] for q in p); ymin=min(q[1] for q in p)
    xmax=max(q[0]+q[2] for q in p); ymax=max(q[1]+q[3] for q in p)
    for i,(q,f) in enumerate(zip(p,flags)):
        code=int(f[4]);x,y,w,h=q
        if not code: continue
        gaps={1:abs(x-xmin),2:abs(x+w-xmax),4:abs(y+h-ymax),8:abs(y-ymin)}
        required=[bit for bit in gaps if code&bit]
        if all(gaps[b]<1e-6 for b in required):continue
        total_fail+=1; fail_cases.add(r['case_id'])
        relative=max(gaps[b]/((xmax-xmin) if b in (1,2) else (ymax-ymin)) for b in required)
        if relative<=.01: near_fail+=1;near_cases.add(r['case_id'])
        if f[0] or f[1]:continue
        new=copy.deepcopy(p)
        if code&1:new[i][0]=xmin
        if code&2:new[i][0]=xmax-w
        if code&4:new[i][1]=ymax-h
        if code&8:new[i][1]=ymin
        if any(v<0 for v in new[i][:2]) or overlap_blocks(new,1e-6):continue
        # All other blocks unchanged, no dimension/area or preplaced change.
        candidates.append({'case_id':r['case_id'],'block':i,'code':code,'relative_gap':relative,
          'gaps':{str(b):gaps[b] for b in required},'before':q,'after':new[i], 'row':r,'positions':new})
candidates.sort(key=lambda c:(c['relative_gap'],c['case_id'],c['block']))
examples=candidates[:3]
if not examples:raise RuntimeError('No direct legal boundary translation found')
for k,c in enumerate(examples,1):
    r=c['row'];m=copy.deepcopy(r['reference_metrics']);m['boundary_violations']-=1;m.pop('cost',None)
    render({'case_id':f"{c['case_id']} | block {c['block']}; translation only; Q not rescored",'panel':'Boundary evidence',
      'constraints':r['public_constraints'],'variants':{'reference':variant(r['reference'],r['reference_metrics']),
      'raw':variant(c['positions'],{}),'final':finals[r['case_id']]['variants']['conic_robust']['final']}},
      AS/f'boundary_{k}.png',stages=(('reference','Official reference'),('raw',f"Translate block {c['block']} only"),('final','Our saved solution')))
# A readable close-up supplements, rather than replaces, the common layout view.
from matplotlib.patches import Rectangle
c=examples[0];r=c['row'];p=r['reference'];i=c['block']; x,y,w,h=c['before'];nx,ny,_,_=c['after']
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
pad=max(w,h)*.4; lim=(min(x,nx)-pad,max(x,nx)+w+pad,min(y,ny)-pad,max(y,ny)+h+pad)
for ax,ps,title in zip(axes,[p,c['positions']],['官方参考：未贴到指定边','只平移此块：贴边且无重叠']):
    for j,(a,b,cw,ch) in enumerate(ps):
        if a+cw<lim[0] or a>lim[1] or b+ch<lim[2] or b>lim[3]:continue
        ax.add_patch(Rectangle((a,b),cw,ch,fc='#edbba7' if j==i else '#dbe3eb',ec='#bd4c2d' if j==i else '#8293a4',lw=1.5 if j==i else .6))
        ax.text(a+cw/2,b+ch/2,str(j),ha='center',va='center',fontsize=8)
    xmin=min(q[0] for q in p);xmax=max(q[0]+q[2] for q in p);ymin=min(q[1] for q in p);ymax=max(q[1]+q[3] for q in p)
    for bit in (1,2,4,8):
        if c['code']&bit:
            if bit in (1,2):ax.axvline(xmin if bit==1 else xmax,color='#be3f2a',ls='--')
            else:ax.axhline(ymax if bit==4 else ymin,color='#be3f2a',ls='--')
    ax.set(xlim=lim[:2],ylim=lim[2:],title=title);ax.set_aspect('equal');ax.tick_params(labelsize=8)
fig.suptitle(f"{c['case_id']}  ·  block {i}  ·  gap {max(c['gaps'].values()):g}",fontsize=10)
fig.tight_layout();fig.savefig(AS/'boundary_detail.png',dpi=200)
axes[0].set_title('Official reference: 1-unit top gap')
axes[1].set_title('Translate one block: boundary satisfied, no overlap')
fig.savefig(AS/'boundary_detail_en.png',dpi=200);plt.close(fig)
# MIB certificate: preserve actual interval widths, and display numeric intervals.
b=bound['cases'][0];g=b['groups'][0]
fig,ax=plt.subplots(figsize=(9,2.8))
for j,(block,interval) in enumerate(zip(g['blocks'],g['area_intervals'])):
    ax.plot(interval,[j,j],lw=8,color='#5289ad',solid_capstyle='round')
    ax.text(interval[1]+6,j,f'[{interval[0]:.2f}, {interval[1]:.2f}]',va='center',fontsize=10)
ax.set(yticks=range(3),yticklabels=[f'block {i}' for i in g['blocks']],xlabel='合法面积区间（目标面积 ±1%）',xlim=(35,310),ylim=(-.55,2.6))
ax.set_title('同一 MIB 组：目标面积 72 / 240 / 56，合法面积区间无交集',fontsize=11)
fig.text(.5,.015,'dev:worker_0/layouts_1344.th:82 · MIB group 1 · 四位小数舍入放宽后的区间仍不相交',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(AS/'mib_intervals.png',dpi=200)
ax.set_xlabel('Permitted area interval (target +/-1%)')
ax.set_title('One MIB group: incompatible target areas 72 / 240 / 56',fontsize=11)
fig.texts[0].set_text('dev:worker_0/layouts_1344.th:82 | group 1 | intervals remain disjoint after round-4 relaxation')
fig.savefig(AS/'mib_intervals_en.png',dpi=200);plt.close(fig)
r=refs[0]
render({'case_id':r['case_id'],'panel':'MIB evidence: blocks 33 / 42 / 63, group 1','constraints':r['public_constraints'],
 'variants':{'reference':variant(r['reference'],r['reference_metrics']),
 'raw':variant(drafts[r['case_id']]['draft']),
 'final':finals[r['case_id']]['variants']['conic_robust']['final']}},AS/'mib_layout.png',
 stages=(('reference','Official reference'),('raw','Model proposal'),('final','Our saved solution')))
evidence={'scope':'Saved dev_quality100 only; no dataset reads, inference, or optimization. Boundary checks are geometry-only translations.',
 'boundary':{'reference_violating_blocks':total_fail,'reference_cases':len(fail_cases),'within_1pct_bbox_blocks':near_fail,'within_1pct_bbox_cases':len(near_cases),'direct_legal_translation_blocks':len(candidates),
 'examples':[{k:v for k,v in c.items() if k not in ('row','positions')} for c in examples]},
 'mib_summary':bound['summary'],'median_layout_case':median['case_id'],
 'reference_dev100_Q_mean':sum(r['reference_metrics']['cost'] for r in refs)/len(refs),
 'better_dev100':sum(finals[r['case_id']]['variants']['conic_robust']['final']['metrics']['cost']<r['reference_metrics']['cost'] for r in refs),
 'sources':[{ 'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [refpath,finalpath,draftpath,lbpath,boundpath]]}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
portable=[]
for c in examples:
    r=c['row']
    portable.append({'case_id':r['case_id'],'public_constraints':r['public_constraints'],
      'area_targets':r['area_targets'],'reference_positions':r['reference'],
      'reference_metrics':r['reference_metrics'],'translated_block':c['block'],
      'translated_positions':c['positions']})
r=refs[0]
portable.append({'case_id':r['case_id'],'public_constraints':r['public_constraints'],
 'area_targets':r['area_targets'],'reference_positions':r['reference'],'reference_metrics':r['reference_metrics'],
 'mib_certificate':bound['cases'][0]})
(OUT/'layout_evidence.json').write_text(json.dumps(portable,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='sources'},ensure_ascii=False,indent=2))
