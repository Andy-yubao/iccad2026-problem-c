"""Render requested matched dev10 and E159 own-best worst dev10; no training/tests."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,json,random,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from experiments.E158_role_initial3.model import GeometryModel as E158
from experiments.E159_role_initial128.model import GeometryModel as E159
from experiments.E159_role_initial128.data import load_rows,collate
from experiments.E159_role_initial128.public import inputs_only
from experiments.E159_role_initial128.batches import decode
from experiments.E159_role_initial128.loss import per_case
from visualization.layouts import plot_case,render

EXP=HERE.parents[1]
PAIRED=EXP/'runs/visualization_compare10_v1'
WORST=EXP/'runs/visualization_loss_worst10_v1'
for p in (PAIRED,WORST):p.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
COLS=[['total','总Loss'],['reference_weighted','Reference'],['overlap_weighted','重叠×0.25'],
      ['wire_compact_weighted','线长/紧凑'],['grouping_weighted','Grouping'],['boundary_weighted','Boundary'],
      ['public_shape_weighted','F/P尺寸'],['public_preplaced_position_weighted','P位置'],['mib_weighted','MIB Loss']]

def best_record(experiment,arm):
    frozen=read(experiment/'frozen_initial.json')
    progress=read(experiment/f'runs/{arm}_seed0/progress.json')
    best=next(h for h in progress['history'] if h['epoch']==frozen['best_epoch'])
    return frozen,best

def load_model(factory,frozen):
    checkpoint=torch.load(ROOT/frozen['checkpoint'],map_location='cpu',weights_only=False)
    model=factory(**checkpoint['model_config']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    return model

def predict(model,batch,n):
    states=model(inputs_only(batch))
    loss={k:float(v[0]) for k,v in per_case(states,batch,'e159_initial').items()}
    loss.update(objective_total=loss['total'],directly_supervised=True,supervision_weight=1.,objective_name='E158/E159同口径L0')
    positions=(decode(states[0],batch)[0,:n]*batch['scale'][0]).cpu().numpy().astype(float).tolist()
    return positions,loss

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    f158,h158=best_record(ROOT/'experiments/E158_role_initial3','e158_initial')
    f159,h159=best_record(EXP,'e159_initial')
    recorded159=h159['dev_monitor']['per_case_total']
    worst=sorted(range(len(recorded159)),key=lambda i:(-recorded159[i],i))[:10]
    ordinary=sorted(random.Random(20261121).sample([i for i in range(500) if i not in worst],10))
    indices=sorted(set(ordinary+worst))
    rows=load_rows('dev_monitor500',indices=indices)
    m158=load_model(E158,f158);m159=load_model(E159,f159)
    print('Matched dev10',ordinary,'E159 worst10',worst,flush=True)
    predictions={};members={};details=[]
    with torch.inference_mode():
        for index,row in zip(indices,rows):
            batch,_=collate([row],'cuda');n=int(row['inputs'][0])
            p159,l159=predict(m159,batch,n)
            item=dict(index=index,n=n,member=list(row['member']),constraints=np.asarray(row['inputs'][5],dtype=int).tolist(),
                reference=row['supervision_xywh'].tolist(),E159=p159,loss159=l159,
                monitor_loss159=recorded159[index])
            if index in ordinary:
                p158,l158=predict(m158,batch,n)
                item.update(E158=p158,loss158=l158,monitor_loss158=h158['dev_monitor']['per_case_total'][index])
            predictions[index]=item
            print('Captured',index,flush=True)
    cases=[]
    for index in ordinary:
        d=predictions[index];a=d['loss158'];b=d['loss159']
        case=dict(label=f'dev{index:03d} · E158 {a["total"]:.6f} → E159 {b["total"]:.6f}',
            group='同题10例',member=d['member'],n=d['n'],constraints=d['constraints'],
            reference=d['reference'],reference_label='监督参考（形状修正）',
            draft=d['E158'],draft_label='E158 · best58 · L0',final=d['E159'],
            final_label='E159 · best47 · L0',final_caption=f'网络初稿，总Loss：{b["total"]:.6f}\n相对E158：{b["total"]-a["total"]:+.6f}',
            model_losses=[dict(a,label='E158 · best58'),dict(b,label='E159 · best47')],
            loss_columns=COLS,loss_label='E158 L0 Loss',
            loss_caption='两行分别是同题两个模型的实际L0 Loss及已计训练系数贡献；不是模型中间层。',
            objective_note='三栏依次为监督参考、E158网络初稿、E159网络初稿。两模型都只有一个L0输出，没有执行后处理。点击模块可在三栏同步定位。',
            view_kind='network_comparison')
        plot_case(dict(title=case['label'],constraints=d['constraints'],variants={
            'reference':dict(positions=d['reference']),'E158':dict(positions=d['E158']),'E159':dict(positions=d['E159'])}),
            PAIRED/f'png/dev{index:03d}.png',stages=[('reference','监督参考'),('E158',f'E158 · Loss {a["total"]:.6f}'),('E159',f'E159 · Loss {b["total"]:.6f}')])
        cases.append(case)
        details.append(dict(category='paired',index=index,member=d['member'],E158=a['total'],E159=b['total'],delta=b['total']-a['total']))
    write(PAIRED/'data.json',dict(title='E158 与 E159 · 同题10例 · L0对照',
        note='seed20261121从排除E159最差10题后的dev500中随机抽取；同题、同坐标范围、同配色。右栏是E159网络初稿，不是后处理。',
        identity=dict(E158=f158,E159=f159,indices=ordinary,excluded_worst=worst),cases=cases))
    cases=[]
    for rank,index in enumerate(worst,1):
        d=predictions[index];loss=d['loss159']
        case=dict(label=f'Loss最差#{rank:02d} · dev{index:03d} · {d["monitor_loss159"]:.6f}',group='E159 Loss最差10例',
            member=d['member'],n=d['n'],constraints=d['constraints'],reference=d['reference'],reference_label='监督参考（形状修正）',
            layers=[d['E159']],layer_ids=[0],layer_labels=['L0 · E159 best47'],layer_losses=[loss],
            hide_missing_final=True,loss_columns=COLS,loss_label='E159 L0 Loss',
            loss_caption='按best47原固定dev500逐题总Loss降序选取；相同Loss按dev编号升序。显示实际系数后的单题贡献。',
            objective_note='这是500个验证例中E159 own-best原监测Loss最差的10例，与同题对照10例互不重复。没有后段或后处理。')
        render(case,WORST/f'png/rank{rank:02d}_dev{index:03d}.png',layer=0,title=case['label'])
        cases.append(case);details.append(dict(category='worst',rank=rank,index=index,member=d['member'],E159=loss['total'],monitor_loss159=d['monitor_loss159']))
    write(WORST/'data.json',dict(title='E159 · best47 · dev500 Loss最差10例',
        note='从best47原dev500监测记录按总Loss降序取前10题，展示形状修正监督参考与实际L0，未执行后处理。',
        identity=dict(E159=f159,indices=worst,ranking='best47 dev500 per_case_total descending, index ascending'),cases=cases))
    selection=dict(seed=20261121,ordinary_indices=ordinary,worst_indices=worst,selection_population='fixed dev500',
        ordinary_rule='uniform sample from remaining490 after excluding worst10',worst_rule='own-best47 recorded total descending, index ascending',
        records=details,E158=f158,E159=f159,model_forward_calls_E158=10,model_forward_calls_E159=20,
        tests_executed=False,optimizer_steps=0,postprocessing_calls=0,test_read=False)
    write(HERE/'selection.json',selection)
    write(PAIRED/'selection.json',selection);write(WORST/'selection.json',selection)
    with (HERE/'selected_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['category','rank','index','member','E158','E159','delta','monitor_loss159']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(details)
    print('Completed requested matched10 and worst10 galleries',flush=True)

if __name__=='__main__':main()
