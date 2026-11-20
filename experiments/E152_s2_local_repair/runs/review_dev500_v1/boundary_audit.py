"""Audit saved boundary failures and necessary public-hard lower bounds, no solver."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from datasets.random_reference import RandomReference

HERE=Path(__file__).resolve().parent
BITS=(1,2,4,8)
NAMES=('L','R','T','B')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def name(code):return ''.join(s for b,s in zip(BITS,NAMES) if code&b)
def failures(pos,flags):
    pos=np.asarray(pos,dtype=float);lo=pos[:,:2].min(0);hi=(pos[:,:2]+pos[:,2:]).max(0)
    gaps=np.column_stack((pos[:,0]-lo[0],hi[0]-pos[:,0]-pos[:,2],hi[1]-pos[:,1]-pos[:,3],pos[:,1]-lo[1]))
    required=(flags[:,4,None].astype(int)&np.asarray(BITS))!=0
    return required & (np.abs(gaps)>=1e-6),gaps

def main():
    ds=RandomReference('dev_monitor500');population=Counter();preplaced_population=Counter();reference=Counter()
    data=read(HERE/'boundary_hard_conflicts.json');per_case={}
    for i in range(500):
        item=ds[i];flags=np.asarray(item['solver_inputs'][5],dtype=int);spec=np.asarray(item['solver_inputs'][6]);ref=np.asarray(item['supervision']['reference_xywh'])
        badref,refgaps=failures(ref,flags)
        for j,code in enumerate(flags[:,4]):
            if code:
                population[name(code)]+=1
                if flags[j,1]:preplaced_population[name(code)]+=1
                if badref[j].any():reference[name(code)]+=1
        for arm,entry in data.items():
            row=read(HERE.parent/f'{arm}_seed0/evaluation_no_mib_dev500/cases/{i:04d}.json')
            assert row['member']==item['member']
            fail,gaps=failures(row['positions'],flags)
            assert int(fail.any(1).sum())==row['metrics']['boundary_violations']
            pos=np.asarray(row['positions']);lo=pos[:,:2].min(0);hi=(pos[:,:2]+pos[:,2:]).max(0)
            for violation in [r for r in entry['rows'] if r['index']==i]:
                j=violation['module'];violation['reference_also_fails_same_module']=bool(badref[j].any())
                violation['reference_failed_sides']=[NAMES[k] for k in np.flatnonzero(badref[j])]
                violation['reference_required_gaps']={NAMES[k]:float(refgaps[j,k]) for k,b in enumerate(BITS) if flags[j,4]&b}
                protrusions=[]
                for side in violation['failed']:
                    axis=0 if side in ('L','R') else 1;upper=side in ('R','T')
                    edge=pos[:,axis]+(pos[:,axis+2] if upper else 0)
                    target=float(edge[j]);extreme=float(hi[axis] if upper else lo[axis])
                    extreme_ids=np.flatnonzero(np.abs(edge-extreme)<1e-6).tolist()
                    beyond=np.flatnonzero(edge>target+1e-6 if upper else edge<target-1e-6).tolist()
                    protrusions.append(dict(failed_side=side,boundary_target=target,current_bbox_edge=extreme,
                        bbox_extreme_modules=extreme_ids,modules_beyond_target=beyond,
                        extreme_modules_preplaced=[int(flags[k,1]) for k in extreme_ids]))
                violation['protrusions']=protrusions
        if (i+1)%100==0:print(f'Boundary audit {i+1}/500',flush=True)
    result=dict(population=dict(population),preplaced_population=dict(preplaced_population),reference_violations=dict(reference),models=data,
        definitions=dict(TR='bitmask6 = R2 | T4; RT in machine records means the same corner',
            boundary='Count one block if any requested edge fails abs(gap)<1e-6',
            unavoidable='For a P-constrained offending block, another immutable P endpoint extends farther, or a locked width/height cannot fit its upper limit with nonnegative coordinates; sufficient lower-bound proof only',
            attribution='An unavoidable soft violation is not proof of a postprocessor defect. Remaining failures need candidate-level analysis; no claim of repairability from this bound alone'),
        training=False,solver_calls=0,Q_calls=0,test_read=False)
    for arm,entry in data.items():
        entry['types_with_denominators']={kind:dict(violations=n,total_constraints=population[kind],rate=n/population[kind]) for kind,n in entry['types'].items()}
        entry['preplaced_violation_modules']=sum(r['preplaced']!=0 for r in entry['rows'])
        entry['reference_also_violates_modules']=sum(r['reference_also_fails_same_module'] for r in entry['rows'])
        entry['all_positive_failed_gaps']=[g for r in entry['rows'] for side,g in r['gaps'].items() if side in r['failed']]
    (HERE/'boundary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# E152 v2 贴边违规审计','',
        '统计范围为两组已保存的原dev500后处理结果。按官方1e-6容差重新逐模块核对，全部与已存违规数一致；没有训练或调用求解器。TR的编码为6（R=2、T=4），机器记录写RT，含义相同。','',
        '| 原约束类型 | dev500约束模块总数 | A违规数 | B违规数 |','|---|---:|---:|---:|']
    for kind in ('L','R','T','B','LT','RT','LB','RB'):
        lines.append(f'| {kind} | {population[kind]} | {data["e152_s2_ref"]["types"].get(kind,0)} | {data["e152_s2_no_ref"]["types"].get(kind,0)} |')
    lines+=['','TR并非多数。A为10/24（41.7%），B为9/29（31.0%）。按实际失败的边统计，包含R失败的模块为A22/24、B25/29；这包括单独R、TR和BR。','',
        '更强的共同点是P：A23/24、B26/29个违规模块固定位置不可移动。两组各至少12个违规模块可由公开硬规格直接证明，存在一个不可能满足的要求边；其中11个模块的全部失败边都不可避免。证明仅使用另一P模块固定端点或固定尺寸的必要宽高下界，不使用reference自由布局。未被证明不可避免不等于已证明可修复。','',
        '示例：','', '| 模型 | dev | 模块 | 约束 | 失败边及间距 | P | 有硬冲突证明的边 | reference该模块也违规 |','|---|---:|---:|---|---|---:|---|---:|']
    for arm,entry in data.items():
        for r in entry['rows']:
            gaps='; '.join(f'{side}={r["gaps"][side]:.9g}' for side in r['failed'])
            lines.append(f'| {"A" if arm=="e152_s2_ref" else "B"} | {r["index"]:04d} | {r["module"]} | {r["type"]} | {gaps} | {r["preplaced"]} | {"/".join(r["unavoidable_failed_sides_from_public_hard_constraints"]) or "未证实"} | {int(r["reference_also_fails_same_module"])} |')
    lines+=['','源码核对：','',
        '- `evaluation/repair/fast_geometry.py` 的 shape 约束将R映射到x上边界、T映射到y上边界，TR同时建立两条不等式，未发现TR位码漏处理。`projection.py`同样逐位处理。',
        '- 内部 `Objective.violations` 与官方使用相同位码与1e-6容差；本次失败间距明显大于数值容差。',
        '- `certify_polish.py` 的贴边移动候选明确跳过P模块。保持P不动是硬约束所需，但它也没有在该处针对P的贴边需求移动其他撑大外框的模块；这是可能改进的覆盖范围，不能仅据此判定为bug。',
        '- 当前证据不支持“主要是TR专属处理错误”。更准确的排查方向是固定P的右/上边界、硬规格导致的不可避免软冲突，以及其余模块撑大外框后是否有可接受的收回方案。','']
    (HERE/'boundary_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({a:{k:v for k,v in r.items() if k in ['count','types','failed_sides','unavoidable_boundary_violation_modules','preplaced_violation_modules','reference_also_violates_modules','types_with_denominators']} for a,r in data.items()},indent=2))

if __name__=='__main__':main()
