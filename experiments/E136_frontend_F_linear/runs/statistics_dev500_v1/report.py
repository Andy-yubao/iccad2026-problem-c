"""Descriptive Markdown/CSV tables; deliberately no diagnoses or recommendations."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

LOSS_ROWS=(
    ('完整Loss合计','loss.total','number'),
    ('参考中心位置','loss.reference_position','number'),
    ('参考形状（含0.25系数）','loss.reference_shape','number'),
    ('碰撞：穿透深度','loss.collision_penetration','number'),
    ('碰撞：模块平均负担','loss.collision_mean_load','number'),
    ('碰撞：最大超10%负担','loss.collision_max_excess','number'),
    ('线长','loss.wirelength_component','number'),
    ('外框紧凑','loss.compactness_component','number'),
    ('Grouping','loss.grouping_weighted','number'),
    ('Boundary','loss.boundary_weighted','number'),
    ('F/P尺寸','loss.public_shape_weighted','number'),
    ('P位置','loss.public_preplaced_position_weighted','number'),
)
GEOMETRY_ROWS=(
    ('精确重复覆盖率 (A−U)/A','geometry.duplicate_coverage','percent'),
    ('成对交面积/模块总面积','geometry.pair_intersection','percent'),
    ('逐模块覆盖率均值','geometry.covered_module_mean','percent'),
    ('逐模块覆盖率P95','geometry.covered_module_p95','percent'),
    ('逐模块最大覆盖率','geometry.covered_module_max','percent'),
    ('覆盖率>10%的模块占比','geometry.covered_module_over10','percent'),
    ('实际模块总面积/公开面积总和','geometry.actual_over_public_area','number'),
    ('精确并集/公开面积总和','geometry.union_over_public_area','number'),
    ('外框/公开面积总和','geometry.bbox_over_public_area','number'),
    ('外框/reference外框','geometry.bbox_reference','number'),
    ('外框长宽比','geometry.bbox_aspect','number'),
    ('留白/外框 (B−U)/B','geometry.whitespace_bbox','percent'),
    ('留白/公开面积总和','geometry.whitespace_public','percent'),
    ('HPWL/reference HPWL','geometry.hpwl_reference','number'),
    ('HPWL总值','geometry.hpwl_total','number'),
    ('内部HPWL','geometry.hpwl_internal','number'),
    ('外部HPWL','geometry.hpwl_external','number'),
    ('模块 |log宽高比| P95','geometry.abs_log_aspect_p95','number'),
    ('长宽比>10的模块比例','geometry.aspect_over10_fraction','percent'),
    ('F/P尺寸相对误差均值','geometry.fixed_dimension_relative_mean','percent'),
    ('F/P尺寸相对误差最大值','geometry.fixed_dimension_relative_max','percent'),
    ('P左下角偏移/√规格面积 均值','geometry.preplaced_position_local_mean','percent'),
    ('P左下角偏移/√规格面积 最大值','geometry.preplaced_position_local_max','percent'),
    ('P中心偏移/√规格面积 均值','geometry.preplaced_center_local_mean','percent'),
    ('组接触MST瓶颈/局部尺度','geometry.group_contact_bottleneck','percent'),
    ('组接触MST边平均距离/局部尺度','geometry.group_contact_mean','percent'),
    ('组外框/组模块总面积','geometry.group_group_bbox_over_area','number'),
    ('同组重复覆盖率','geometry.group_same_group_overlap','percent'),
    ('组连通率（1%容差）','geometry.group_connected_0.01','percent'),
    ('组MST边近接触比例（1%容差）','geometry.group_connectivity_0.01','percent'),
    ('组外框内异组覆盖占比','geometry.group_foreign_bbox_fraction','percent'),
    ('组外框空隙内异组覆盖占比','geometry.group_foreign_empty_bbox_fraction','percent'),
    ('组接触树路径受阻比例','geometry.group_tree_blocked_fraction','percent'),
    ('组接触树移动扫掠异组负担','geometry.group_tree_swept_foreign_load','percent'),
    ('Boundary最大要求边距/局部尺度 均值','geometry.boundary_gap_local','percent'),
    ('Boundary最大要求边距P95','geometry.boundary_gap_local_p95','percent'),
    ('Boundary近边比例（1%容差）','geometry.boundary_near_0.01','percent'),
    ('Boundary外侧模块比例','geometry.boundary_outside_fraction','percent'),
    ('Boundary外侧模块比例P95','geometry.boundary_outside_fraction_p95','percent'),
    ('Boundary向外移动扫掠阻挡负担','geometry.boundary_swept_obstacle_load','percent'),
    ('预放置边界锚点外侧深度/局部尺度','geometry.boundary_anchor_outside_depth','percent'),
)
SCORE_ROWS=(
    ('硬合法','score.is_feasible','percent'),
    ('重叠违规对数','score.overlap_violations','number'),
    ('尺寸/位置硬违规模块数','score.dimension_violations','number'),
    ('Boundary违规数','score.boundary_violations','number'),
    ('Grouping违规数','score.grouping_violations','number'),
    ('软违规比例（无MIB）','score.violations_relative','percent'),
    ('原评分Q（草稿包含硬约束罚分）','score.cost','number'),
    ('原记录完整阶段耗时 ms','complete_ms','number'),
)
MOVE_ROWS=(
    ('后处理中心位移均值/√总面积','move.move_global_mean','percent'),
    ('后处理中心位移P95/√总面积','move.move_global_p95','percent'),
    ('后处理中心位移最大值/√总面积','move.move_global_max','percent'),
    ('后处理 |Δlog宽高比| 均值','move.shape_log_change_mean','number'),
    ('后处理 |Δlog宽高比| P95','move.shape_log_change_p95','number'),
)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def get(row,key):
    current=row
    for part in key.split('.',1):
        if not isinstance(current,dict):return None
        current=current.get(part)
    return current

def flat(row):
    output={}
    for category,values in row.items():
        if isinstance(values,dict):
            for key,value in values.items():
                if isinstance(value,(int,float,bool)) and math.isfinite(float(value)):output[category+'.'+key]=float(value)
        elif isinstance(values,(int,float,bool)) and math.isfinite(float(values)):output[category]=float(values)
    return output

def format_value(value,kind='number'):
    if value is None:return '—'
    if kind=='percent':return f'{value*100:.3f}%'
    if value!=0 and abs(value)<1e-6:return f'{value:.2e}'
    return f'{value:.6f}'

def table(rows,columns,labels):
    text='| 指标 | Reference | '+' | '.join(labels)+' |\n|---|'+ '|'.join('---:' for _ in columns)+'|\n'
    for label,key,kind in rows:
        text+='| '+label.replace('|','\\|')+' | '+' | '.join(format_value(get(row,key),kind) for row in columns)+' |\n'
    return text+'\n'

def compose(reference,draft,final,labels,header):
    text=header+'\n\n'
    text+='## S12草稿：布局域Loss拆项\n\n'+table(LOSS_ROWS,[reference,*draft],labels)
    text+='## S12草稿：实际保存布局的几何与约束\n\n'+table(GEOMETRY_ROWS+SCORE_ROWS,[reference,*draft],labels)
    text+='## 原后处理结果：布局域Loss拆项\n\n'+table(LOSS_ROWS,[reference,*final],labels)
    text+='## 原后处理结果：几何、约束与移动量\n\n'+table(GEOMETRY_ROWS+SCORE_ROWS+MOVE_ROWS,[reference,*final],labels)
    return text

def build_report(manifest):
    models=manifest['models'];keys=[m['key'] for m in models];labels=[m['label'] for m in models]
    losses={key:read(HERE/f'losses/{key}.json')['rows'] for key in keys}
    coordinate={stage:read(HERE/f'coordinate_losses/{stage}.json')['rows'] for stage in ['reference',
        *[stage+'.'+key for stage in ('draft','final') for key in keys]]}
    records=[];collection={'reference':[]}
    for stage in ('draft','final'):
        for key in keys:collection[stage+'.'+key]=[]
    folder=HERE/'tables';folder.mkdir(exist_ok=True)
    for i in range(500):
        geo=read(HERE/f'geometry/{i:04d}.json')
        reference=dict(geo['reference'],loss=coordinate['reference'][i]['loss'],training_loss=losses['baseline'][i]['reference'])
        record=dict(index=i,member=geo['member'],counts=geo['counts'],reference=reference,draft={},final={})
        collection['reference'].append(flat(reference))
        for key in keys:
            if losses[key][i]['member']!=geo['member']:raise ValueError('Loss/geometry member mismatch')
            for stage in ('draft','final'):
                row=dict(geo[stage][key],loss=coordinate[stage+'.'+key][i]['loss'],training_loss=losses[key][i][stage])
                if stage=='final':row['move']=geo['movement'][key]
                record[stage][key]=row;collection[stage+'.'+key].append(flat(row))
        records.append(record)
        header=f"# dev {i:04d} · N={geo['counts']['n']}\n\n成员：`{geo['member'][0]} : {geo['member'][1]}`。F/P模块={geo['counts']['fixed_or_preplaced']}，P={geo['counts']['preplaced']}，组={geo['counts']['groups']}，边界模块={geo['counts']['boundary_modules']}。Loss主表在实际保存坐标上用同项同系数计算，不再次decode/clamp；原训练域总Loss另列。只列数值，不作原因解释。"
        text=compose(reference,[record['draft'][key] for key in keys],[record['final'][key] for key in keys],labels,header)
        training_rows=tuple((label,key.replace('loss.','training_loss.',1),kind) for label,key,kind in LOSS_ROWS)
        text+='## 原训练域S12 Loss拆项（监测口径）\n\n'+table(training_rows,[reference,*[record['draft'][key] for key in keys]],labels)
        text+='## 后处理重编码的原训练公式（含decoder限制，非实际几何）\n\n'+table(training_rows,[reference,*[record['final'][key] for key in keys]],labels)
        (folder/f'dev_{i:04d}.md').write_text(text,encoding='utf-8')
    statistics={}
    for column,rows in collection.items():
        statistic={}
        for key in sorted({k for row in rows for k in row}):
            values=np.asarray([row[key] for row in rows if key in row],dtype=float)
            statistic[key]=dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=float(values.max()))
        statistics[column]=statistic
    # Recover nested mean dictionaries for exactly the same table schema.
    def mean_row(column):
        row={}
        for name,value in statistics[column].items():
            if '.' in name:
                category,key=name.split('.',1);row.setdefault(category,{})[key]=value['mean']
            else:row[name]=value['mean']
        return row
    reference=mean_row('reference');draft=[mean_row('draft.'+key) for key in keys];final=[mean_row('final.'+key) for key in keys]
    heading='# 共同 dev500 逐题配对统计：每题归一后取均值\n\nReference为实际参考布局计算值。每组使用已冻结own-Loss-best，草稿取S12。主表为同项同系数的布局诊断Loss：直接使用实际Reference／已存草稿／已存后处理XYWH，不再次decode或clamp；沿用完整public-priority-v1的全部项和系数，未乘监督层权重。原训练域Loss另表保留并核对原监测值。几何度量始终使用实际保存坐标。'
    summary=compose(reference,draft,final,labels,heading)
    training_rows=tuple((label,key.replace('loss.','training_loss.',1),kind) for label,key,kind in LOSS_ROWS)
    summary+='## 原训练域S12 Loss拆项（与主表布局域分开）\n\n'+table(training_rows,[reference,*draft],labels)
    summary+='## 后处理XYWH重编码的原训练公式（含decoder限制，非实际几何）\n\n'+table(training_rows,[reference,*final],labels)
    from scoring.official import official_evaluator
    reference_costs=[record['reference']['score']['cost'] for record in records]
    reference_weighted=official_evaluator().compute_total_score(reference_costs,[record['counts']['n'] for record in records])
    summary+='## 后处理Q与耗时分布（原结果）\n\n| 指标 | Reference | '+' | '.join(labels)+' |\n|---|'+ '|'.join('---:' for _ in range(6))+'|\n'
    for label,key in (('平均Q','mean_Q'),('加权Q','weighted_Q'),('混合Q','mixed_Q'),('Q的P95','p95_Q'),('最坏Q','max_Q'),('完整均时ms','mean_complete_seconds'),('完整P95ms','p95_complete_seconds'),('最大耗时ms','max_complete_seconds'),('超过250ms例数','over250ms')):
        values=[m['result'][key]*(1000 if 'seconds' in key else 1) for m in models]
        ref_values=statistics['reference']['score.cost']
        ref=(ref_values['mean'] if key=='mean_Q' else reference_weighted if key=='weighted_Q' else
             .5*(ref_values['mean']+reference_weighted) if key=='mixed_Q' else ref_values['p95'] if key=='p95_Q' else ref_values['max'] if key=='max_Q' else None)
        summary+='| '+label+' | '+format_value(ref)+' | '+' | '.join(format_value(value) for value in values)+' |\n'
    summary+='\n按适用题目计算指标均值；无适用对象记缺失，不补0。完整精度、有效题数n以及逐题表见JSON/CSV和tables目录。模块P95、最大值等先逐题计算再对题目取均值，不与全体模块混池。组连通/扫掠为既有1%边长接触代理，非联合可行性证明；Q与约束计数沿用原评分，reference另用同口径计算。reference没有求解耗时或后处理位移，以—表示。\n'
    (HERE/'summary.md').write_text(summary,encoding='utf-8')
    (HERE/'summary.json').write_text(json.dumps(dict(statistics=statistics,models=models,counts=dict(cases=500,models=5)),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    index='# 500题逐题对照表\n\n数值列最左侧为Reference；每题均有草稿/原后处理的Loss拆项、几何和约束表。不作原因分析。\n\n[全体均值表](summary.md) · [完整精度CSV](all_cases.csv)\n\n| 题号 | N | 基线Q | A Q | B Q | C Q | D Q |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for record in records:
        index+=f"| [dev {record['index']:04d}](tables/dev_{record['index']:04d}.md) | {record['counts']['n']} | "+' | '.join(f"{record['final'][key]['score']['cost']:.6f}" for key in keys)+' |\n'
    (HERE/'index.md').write_text(index,encoding='utf-8')
    columns=sorted({k for rows in collection.values() for row in rows for k in row})
    with (HERE/'all_cases.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dev_index','source_shard','source_row','stage','model',*columns]);writer.writeheader()
        for record in records:
            common=dict(dev_index=record['index'],source_shard=record['member'][0],source_row=record['member'][1])
            writer.writerow(dict(common,stage='reference',model='reference',**flat(record['reference'])))
            for stage in ('draft','final'):
                for key in keys:writer.writerow(dict(common,stage=stage,model=key,**flat(record[stage][key])))
    # A compact directly reusable selection of Markdown tables for the chat.
    compact_geometry=tuple(row for row in GEOMETRY_ROWS if row[1] in {
        'geometry.duplicate_coverage','geometry.covered_module_mean','geometry.covered_module_max','geometry.covered_module_over10',
        'geometry.bbox_over_public_area','geometry.whitespace_bbox','geometry.hpwl_reference','geometry.fixed_dimension_relative_mean',
        'geometry.preplaced_position_local_mean','geometry.group_contact_bottleneck','geometry.group_tree_blocked_fraction',
        'geometry.boundary_gap_local','geometry.boundary_swept_obstacle_load'})
    chat='草稿Loss（500题均值）\n\n'+table(LOSS_ROWS,[reference,*draft],labels)
    chat+=table((('原训练域S12总Loss（另列）','training_loss.total','number'),),[reference,*draft],labels)
    chat+='草稿几何（500题均值）\n\n'+table(compact_geometry,[reference,*draft],labels)
    chat+='后处理Loss（500题均值）\n\n'+table(LOSS_ROWS,[reference,*final],labels)
    chat+='后处理几何（500题均值）\n\n'+table(compact_geometry+MOVE_ROWS+tuple(row for row in SCORE_ROWS if row[1] in ('score.is_feasible','score.boundary_violations','score.grouping_violations','score.cost')),[reference,*final],labels)
    (HERE/'chat_tables.md').write_text(chat,encoding='utf-8')
    (HERE/'reporting_sources.json').write_text(json.dumps(dict(report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_weighted_Q=reference_weighted,reference_mean_Q=float(np.mean(reference_costs)),
        reference_mixed_Q=.5*(reference_weighted+float(np.mean(reference_costs)))),indent=2)+'\n',encoding='utf-8')
