"""Twenty saved three-stage layouts via visualization/, without inference."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import json,html,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw,ImageFont,ImageOps
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from scoring.evaluation import evaluate_case
from scoring.checks import independent_hard_checks
from visualization.layouts import plot_case,geometry_details
from .io import ROOT,HERE,save,sha

OUT=HERE/'runs/20261020_poor_good_gallery20'
SOURCE=ROOT/'experiments/E25_integrated_baseline/runs/20261020_controls_steady_monitor500/results.json'
CACHE=ROOT/'experiments/E25_integrated_baseline/runs/public_dev500_v1.pt'

def public_features(row):
    inputs=row['case'].solver_inputs;c=np.asarray(inputs[5]);n=len(c)
    return {'n':n,'boundary_count':int((c[:,4]!=0).sum()),'preplaced_count':int((c[:,1]!=0).sum()),
            'fixed_count':int((c[:,0]!=0).sum()),'group_members':int((c[:,3]!=0).sum()),'mib_members':int((c[:,2]!=0).sum())}

def render(task):
    row,path=task;bounds=plot_case(row,path,stages=(('reference','Original reference geometry'),
        ('raw','Frozen E17 network draft'),('final','E25 steady final layout')))
    return Path(path).name,bounds

def summary(records):
    fields=['n','boundary_fraction','preplaced_fraction','fixed_fraction','group_fraction','mib_fraction',
            'raw_center_error','raw_log_ratio_error','raw_overlap_pairs','raw_overlap_area_ratio',
            'raw_group_violations','final_center_error','bbox_expansion','wire_expansion','final_soft',
            'seconds','rescue','reference_aspect','final_aspect','area_cv']
    return {field:{'mean':float(np.mean([r['features'][field] for r in records])),
                   'median':float(np.median([r['features'][field] for r in records]))} for field in fields}

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    result=json.loads(SOURCE.read_text(encoding='utf-8'));rows=result['records']
    cache=torch.load(CACHE,map_location='cpu',weights_only=False);public=cache['rows']
    poor=sorted(rows,key=lambda r:(-r['metrics']['cost'],r['index']))[:10]
    goodpool=[r for r in rows if r['metrics']['cost']<=1.01]
    features={r['index']:public_features(public[r['index']]) for r in rows}
    costs=np.empty((10,len(goodpool)))
    for a,bad in enumerate(poor):
        f=features[bad['index']]
        for b,good in enumerate(goodpool):
            g=features[good['index']]
            costs[a,b]=100*abs(f['n']-g['n'])+sum(abs(f[k]/f['n']-g[k]/g['n'])*10 for k in ['boundary_count','preplaced_count','group_members','mib_members'])+good['index']*1e-7
    aa,bb=linear_sum_assignment(costs);matched={int(a):goodpool[int(b)] for a,b in zip(aa,bb)}
    selection={'poor':'10 highest saved final Q, descending; ties by index',
        'good':'Q <= 1.01, one-to-one matching: module count first, then public constraint densities',
        'matching_cost':'100*abs(delta_n) + 10*sum(abs(delta_constraint_fraction)) + 1e-7*good_index',
        'good_pool_count':len(goodpool),'source_sha256':sha(SOURCE),'registry_sha256':sha(REGISTRY),
        'pairs':[{'rank':i+1,'poor':poor[i]['index'],'good':matched[i]['index']} for i in range(10)],
        'test_read':False,'inference_calls':0,'postprocess_calls':0,'render_processes':2,'threads_per_process':1}
    save(OUT/'selection.json',selection)
    references=ReferenceLayouts('dev_monitor500');lookup={tuple(r['member']):i for i,r in enumerate(references.records)}
    records=[];tasks=[]
    for pair in range(10):
        for cohort,record in [('poor',poor[pair]),('good',matched[pair])]:
            index=record['index'];raw=np.asarray(public[index]['draft']);case=public[index]['case']
            item=references[lookup[tuple(record['member'])]];ref=np.asarray(item['positions'],dtype=float)
            c=np.asarray(case.solver_inputs[5],dtype=int);final=np.asarray(record['positions']);n=len(c)
            raw_metrics=evaluate_case(case,raw.tolist(),0.)['metrics'];raw_metrics['cost']=None
            original_details=geometry_details(ref.tolist(),c.tolist());raw_details=geometry_details(raw.tolist(),c.tolist())
            final_details=geometry_details(final.tolist(),c.tolist())
            publicf=features[index];areas=np.asarray(case.solver_inputs[1],dtype=float);unit=math.sqrt(float(areas.sum())/n)
            error=lambda x:float(np.sqrt(np.mean(np.sum(((x[:,:2]+x[:,2:]/2)-(ref[:,:2]+ref[:,2:]/2))**2,axis=1)))/unit)
            ii,jj=np.triu_indices(n,1);ov=np.minimum(raw[ii,:2]+raw[ii,2:],raw[jj,:2]+raw[jj,2:])-np.maximum(raw[ii,:2],raw[jj,:2]);overlap=np.maximum(ov,0).prod(1)
            refspan=(ref[:,:2]+ref[:,2:]).max(0)-ref[:,:2].min(0);span=(final[:,:2]+final[:,2:]).max(0)-final[:,:2].min(0)
            m=record['metrics'];f={**publicf,'boundary_fraction':publicf['boundary_count']/n,
                'preplaced_fraction':publicf['preplaced_count']/n,'fixed_fraction':publicf['fixed_count']/n,
                'group_fraction':publicf['group_members']/n,'mib_fraction':publicf['mib_members']/n,
                'raw_center_error':error(raw),'final_center_error':error(final),
                'raw_log_ratio_error':float(np.abs(np.log(raw[:,2]/raw[:,3])-np.log(ref[:,2]/ref[:,3])).mean()),
                'raw_overlap_pairs':int((overlap>1e-12).sum()),'raw_overlap_area_ratio':float(overlap.sum()/areas.sum()),
                'raw_group_violations':raw_metrics['grouping_violations'],'bbox_expansion':m['area_gap'],
                'wire_expansion':m['hpwl_gap'],'final_soft':m['total_soft_violations'],'seconds':record['seconds'],
                'rescue':int(any(t['phase']=='full_packed_refine' for t in record['trace'])),
                'reference_aspect':float(max(refspan)/min(refspan)),'final_aspect':float(max(span)/min(span)),
                'area_cv':float(areas.std()/areas.mean())}
            filename=f'{cohort}_{pair+1:02d}_index{index}.png'
            row={'case_id':f'index {index} | '+':'.join(map(str,record['member'])),'panel':f'E25 {cohort} case #{pair+1:02d}',
                'constraints':c.tolist(),'caption':f'n={n} | final Q={m["cost"]:.6f} | bbox {m["area_gap"]:+.1%} | wire {m["hpwl_gap"]:+.1%} | solve {record["seconds"]:.3f}s | Original WH may not match derived MIB areas',
                'variants':{'reference':{'positions':ref.tolist(),'metrics':{},'soft_details':original_details},
                    'raw':{'positions':raw.tolist(),'metrics':raw_metrics,'soft_details':raw_details},
                    'final':{'positions':final.tolist(),'metrics':m,'soft_details':final_details}}}
            records.append({'cohort':cohort,'pair':pair+1,'index':index,'member':record['member'],'image':filename,'Q':m['cost'],
                'features':f,'final_metrics':m,'raw_metrics':raw_metrics,'raw_details':raw_details,'final_details':final_details})
            tasks.append((row,str(OUT/filename)))
    # The shared renderer owns all layout drawing; workers only render saved data.
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(render,task) for task in tasks]
        for count,future in enumerate(as_completed(futures),1):
            filename,bounds=future.result()
            for r in records:
                if r['image']==filename:r['render']=bounds
            if count%5==0:print('Rendered',count,'/20',flush=True)
    for cohort in ['poor','good']:
        selected=[r for r in records if r['cohort']==cohort]
        sheet=Image.new('RGB',(2200,5*720+75),'#e2e8f0');draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26)
        draw.text((20,15),'较差10例：原参考 / 模型草稿 / 最终布局' if cohort=='poor' else '匹配的较好10例：原参考 / 模型草稿 / 最终布局',fill='#0f172a',font=font)
        for k,r in enumerate(selected):
            with Image.open(OUT/r['image']) as picture:thumb=ImageOps.contain(picture.convert('RGB'),(1090,650),Image.Resampling.LANCZOS)
            xx=(k%2)*1100;yy=75+(k//2)*720
            draw.text((xx+12,yy),f'#{r["pair"]} index {r["index"]}   n={r["features"]["n"]}   Q={r["Q"]:.5f}',fill='#991b1b' if cohort=='poor' else '#166534',font=font)
            sheet.paste(thumb,(xx+(1100-thumb.width)//2,yy+42))
        sheet.save(OUT/f'overview_{cohort}.png')
    summaries={cohort:summary([r for r in records if r['cohort']==cohort]) for cohort in ['poor','good']}
    save(OUT/'comparison.json',{'selection':selection,'summary':summaries,'records':records,
        'reference_policy':'Original dataset geometry, displayed without derived-task Q; not silently MIB-repaired',
        'raw_Q_policy':'Not shown: hard-illegal drafts do not have a meaningful feasible Q',
        'coordinate_storage':'No duplicated layout JSON; source cache, saved result and frozen member pointers retained'})
    lines=['# E25：较差10例与匹配较好10例的三图比较','',
        '三栏依次为原始参考几何、冻结E17模型草稿、E25 steady最终布局。只读取已保存草稿/最终布局和20例开发参考，不推理、不修复、不读测试。',
        '差例按固定500例最终Q降序取10例；好例来自Q≤1.01的130例，以模块数优先、约束密度其次做一对一匹配。不是简单地拿小规模好例与大规模差例比较，也不是独立统计检验。',
        '原参考宽高未按MIB派生面积修改，因此参考栏不重新给派生任务Q；非法草稿Q也不展示。颜色为分组、纹理为MIB，红色外框为重叠，L/R/T/B徽记为各自bbox的边界状态。','',
        '[交互图集](gallery.html) · [差例总览](overview_poor.png) · [好例总览](overview_good.png) · [精确比较记录](comparison.json)','',
        '| 配对 | 差例 n / Q | 好例 n / Q | 差例 B/G/M | 好例 B/G/M |','|---:|---|---|---|---|']
    cards=[]
    for pair in range(1,11):
        bad=next(r for r in records if r['pair']==pair and r['cohort']=='poor');good=next(r for r in records if r['pair']==pair and r['cohort']=='good')
        def bgm(r):return '/'.join(str(r['final_metrics'][k]) for k in ['boundary_violations','grouping_violations','mib_violations'])
        lines.append(f'| {pair} | [#{bad["index"]} n={bad["features"]["n"]} / {bad["Q"]:.6f}]({bad["image"]}) | [#{good["index"]} n={good["features"]["n"]} / {good["Q"]:.6f}]({good["image"]}) | {bgm(bad)} | {bgm(good)} |')
        cards.append(f'<section data-pair="{pair}"><h2>配对{pair}　差例 #{bad["index"]}（Q={bad["Q"]:.4f}）／好例 #{good["index"]}（Q={good["Q"]:.4f}）</h2>'+''.join(f'<h3>{"较差" if r["cohort"]=="poor" else "较好"}：n={r["features"]["n"]}，B/G/M={bgm(r)}</h3><a href="{r["image"]}"><img src="{r["image"]}" loading="lazy"></a>' for r in [bad,good])+'</section>')
    lines+=['','## 匹配样本的描述统计','',
        '中心误差单位为该例典型模块边长sqrt(公开总面积/n)；原始重叠面积为逐对相交面积之和/公开总面积，存在多重计数，只用于比较。','',
        '| 特征（均值） | 差例10 | 好例10 |','|---|---:|---:|']
    names={'n':'模块数','boundary_fraction':'边界模块比例','preplaced_fraction':'预放置模块比例','group_fraction':'分组模块比例','mib_fraction':'MIB模块比例','raw_center_error':'草稿中心误差','raw_log_ratio_error':'草稿长宽比log误差','raw_overlap_pairs':'草稿重叠对数','raw_overlap_area_ratio':'草稿归一化重叠面积','raw_group_violations':'草稿分组违反','final_center_error':'最终中心误差','bbox_expansion':'最终面积gap','wire_expansion':'最终线长gap','final_soft':'最终软违反','seconds':'原保存求解秒','rescue':'触发完整打包救援比例','reference_aspect':'参考bbox长宽比','area_cv':'模块面积变异系数'}
    for key,label in names.items():lines.append(f'| {label} | {summaries["poor"][key]["mean"]:.4f} | {summaries["good"][key]["mean"]:.4f} |')
    lines+=['','解释和视觉观察补充于本实验README；均值差异是这20例的描述，不能据此断言因果或泛化。','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    options='<option value="all">全部10对</option>'+''.join(f'<option value="{i}">配对{i}</option>' for i in range(1,11))
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>E25 好坏布局三图比较</title><style>body{font:16px/1.6 system-ui;max-width:1600px;margin:auto;padding:24px;background:#f1f5f9}img{width:100%;height:auto}section{padding:20px;background:white;margin:24px 0}select{font:inherit}header{position:sticky;top:0;background:#f1f5f9;padding:8px;z-index:2}</style><header><h1>E25：较差10例与规模匹配的较好10例</h1><p>三栏：原参考／模型草稿／最终布局；点击图片查看原图。差例按Q最差10例，好例Q≤1.01并匹配规模和约束密度。</p><label>查看 <select id="pair">'+options+'</select></label>　<a href="report.md">统计报告</a></header>'+''.join(cards)+'<script>document.getElementById("pair").onchange=e=>document.querySelectorAll("section").forEach(s=>s.hidden=e.target.value!=="all"&&s.dataset.pair!==e.target.value)</script>',encoding='utf-8')
    print(json.dumps({'summary':summaries,'pairs':selection['pairs']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
