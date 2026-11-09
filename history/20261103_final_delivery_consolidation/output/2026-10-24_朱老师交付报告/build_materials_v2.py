"""E42 report assets: official reference, network explanation and three-phase layouts.

Only the three missing example drafts may be generated once. Cached drafts are
reused thereafter. No final solve, training or official scoring is performed.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
os.environ['E42_THREADS']='2'
os.environ['MPLBACKEND']='Agg'
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
FONT=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams.update({'font.family':FONT,'font.size':11,'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
E42=ROOT/'experiments/E42_joint_regularization'
RUN=E42/'runs/20261024_final_test1000_seed20261024'
TRAIN=E42/'runs/20261023_rectangle_w96d12_ab_seed0'
EVIDENCE=HERE/'evidence';ASSETS=HERE/'materials'
RESULTS=ROOT/'references/official/results'
WEIGHTS=TRAIN/'checkpoints/best.pt'
WEIGHT_SHA='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


COPIES={
 'protocol.md':'docs/protocol.md',
 'official_problem_statement.md':'references/official/problem_statement.md',
 'official_qa.md':'references/official/official_qa.md',
 'E42_final_summary.json':str((RUN/'summary.json').relative_to(ROOT)),
 'E42_final_config.json':str((RUN/'config.json').relative_to(ROOT)),
 'E42_final_audit.json':str((RUN/'audit.json').relative_to(ROOT)),
 'E42_environment.json':str((RUN/'environment.json').relative_to(ROOT)),
 'E42_training_results.json':str((TRAIN/'results.json').relative_to(ROOT)),
 'E42_training_config.json':str((TRAIN/'config.json').relative_to(ROOT)),
 'E42_frozen_model.json':str((TRAIN/'frozen.json').relative_to(ROOT)),
 'E42_full_dev.json':str((TRAIN/'full_dev.json').relative_to(ROOT)),
 'E42_completed_analysis.json':str((TRAIN/'completed_analysis.json').relative_to(ROOT)),
 'E42_model.py':'experiments/E42_joint_regularization/model.py',
 'E42_features.py':'experiments/E42_joint_regularization/features.py',
 'E42_inference.py':'experiments/E42_joint_regularization/inference.py',
 'E42_solver.py':'experiments/E42_joint_regularization/solver.py',
 'E42_losses.py':'experiments/E42_joint_regularization/losses.py',
 'E42_train.py':'experiments/E42_joint_regularization/train.py',
 'mean_area_and_registry.py':'datasets/reference_layouts.py',
 'scoring_evaluation.py':'scoring/evaluation.py',
 'official_evaluator.py':'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
}
for file in RESULTS.iterdir():
    if file.is_file():COPIES[file.name]=file.relative_to(ROOT).as_posix()


def savefig(fig,name):
    fig.savefig(ASSETS/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    if name in ('network_structure','official_results_reference'):fig.savefig(ASSETS/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def network_figure():
    fig,ax=plt.subplots(figsize=(13.5,7.1));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.5,.96,'网络怎样得到一张布局草稿',ha='center',fontsize=20,weight='bold')
    boxes=[(.025,.65,.21,.20,'① 每个模块各有一份输入\n面积、固定/预放置、边界要求\n再加模块之间的连接与分组关系'),
           (.275,.65,.20,.20,'② 先猜一次位置和形状\n每块输出：中心横坐标、纵坐标\n以及决定长宽比的一个数'),
           (.515,.65,.20,.20,'③ 连续做12轮修正\n查看其他模块与自己的关系\n更新每块的信息，再改一次猜测'),
           (.755,.65,.22,.20,'④ 输出网络草稿\n用目标面积和长宽比得到宽高\n恢复固定尺寸、预放置位置')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009',fc='#eaf3f3',ec='#126b73',lw=1.4))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.8,linespacing=1.7)
    for a,b in zip(boxes[:-1],boxes[1:]):ax.add_patch(FancyArrowPatch((a[0]+a[2]+.01,.75),(b[0]-.01,.75),arrowstyle='-|>',mutation_scale=17,color='#126b73'))
    ax.add_patch(FancyBboxPatch((.035,.12),.58,.41,boxstyle='round,pad=.012',fc='#f5f8fb',ec='#9aabb7'))
    ax.text(.06,.475,'一轮修正具体做什么？',fontsize=14,weight='bold')
    ax.text(.06,.405,'先看当前猜测：两块相距多远、边缘有无空隙、是否重叠；\n再看任务关系：连接强不强、是否同组、是否要求同形。\n网络据此决定“该多参考谁的信息”，加权汇总其他模块的信息，\n另汇总线连、同组、同形三类邻居，再更新自己的位置和形状。',fontsize=11.4,va='top',linespacing=1.85)
    ax.add_patch(FancyBboxPatch((.67,.12),.295,.41,boxstyle='round,pad=.012',fc='#fff6e8',ec='#bd934d'))
    ax.text(.692,.475,'草稿之后还有几何优化',fontsize=13,weight='bold')
    ax.text(.692,.405,'草稿可能有重叠，不能直接交付。\n优化器调整位置、宽高与相互方位，\n修复硬约束并改善线长、面积、接触。\n最后检查全部硬约束，得到最终布局。',fontsize=10.8,va='top',linespacing=1.9)
    ax.add_patch(FancyArrowPatch((.86,.64),(.82,.55),arrowstyle='-|>',mutation_scale=17,color='#bd934d'))
    ax.text(.5,.035,'每块的输入先转成96个内部数值；12轮对应12个分别学习参数的更新单元。参考答案仅用于训练与评分。',ha='center',fontsize=10.4,color='#475569')
    savefig(fig,'network_structure')


def result_figures(rows,summary):
    with (RESULTS/'C_beta_leaderboard_20260823.csv').open(encoding='utf-8-sig',newline='') as h:
        beta=[{k:float(v) if k not in ('rank','n_feasible') else int(v) for k,v in r.items()} for r in csv.DictReader(h)]
    for r in beta:r['reported_mean_seconds']=r['total_runtime']/100
    write(ASSETS/'official_beta_reference.json',{'source':'references/official/results/C_beta_leaderboard_20260823.csv',
        'cases':100,'test_role':'official hidden Beta','raw_score_interpretation':'working quality-only interpretation, numerically consistent with time-factor floor; no standalone archive column definition',
        'rows':beta,'rank3_total_over_raw':beta[2]['total_score']/beta[2]['raw_score'],
        'rank10_total_over_raw':beta[9]['total_score']/beta[9]['raw_score'],
        'not_same_data_or_hardware_as_local_E42':True})
    with (ASSETS/'official_beta_reference.csv').open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=list(beta[0]));w.writeheader();w.writerows(beta)
    fig,(ax,card)=plt.subplots(1,2,figsize=(12.8,5.9),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    ax.plot([r['raw_score'] for r in beta],range(1,11),'o',color='#126b73',ms=7)
    for r in beta:ax.text(r['raw_score']+.012,r['rank'],f"{r['raw_score']:.4f}",va='center',fontsize=10)
    ax.set(yticks=range(1,11),yticklabels=[f'官方综合第{i}名' for i in range(1,11)],xlabel='归档 raw_score（越低越好）',xlim=(1.05,1.61),ylim=(10.7,.2),title='官方 Beta：100例隐藏测试')
    ax.grid(axis='x',alpha=.2);card.axis('off')
    card.text(.06,.88,'本地 E42：1000例',fontsize=18,weight='bold')
    card.text(.06,.70,f"加权 Q    {summary['weighted_Q']:.6f}",fontsize=19,color='#a16d1d')
    card.text(.06,.55,f"平均求解  {summary['seconds_mean']:.6f} 秒/例\n硬合法     1000 / 1000",fontsize=14,linespacing=2)
    card.text(.06,.31,'本地输入经过MIB平均面积处理；\n官方使用隐藏数据和另一套硬件。\n两侧用于了解数值量级，不能排在同一榜单。',fontsize=12,linespacing=1.8,color='#475569')
    fig.suptitle('官方历史成绩作为参照，保留数据与硬件差异',fontsize=17)
    savefig(fig,'official_results_reference')
    q=np.array([r['metrics']['cost'] for r in rows]);t=np.array([r['seconds'] for r in rows]);y=np.arange(1,1001)/1000
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.7),layout='constrained')
    axes[0].plot(np.sort(q),y,color='#126b73',lw=2)
    axes[0].set(xlabel='每例 Q（越低越好）',ylabel='累计样例比例',title='E42质量分布：全部1000例',ylim=(0,1.02))
    axes[0].text(.33,.35,f"95%的样例Q不超过 {summary['p95_Q']:.3f}\n最差一例Q为 {summary['max_Q']:.3f}",transform=axes[0].transAxes,fontsize=11)
    axes[1].plot(np.sort(t),y,color='#126b73',lw=2)
    axes[1].set(xscale='log',xlabel='每例完整求解时间 / 秒（对数刻度）',ylabel='累计样例比例',title='E42耗时分布：全部1000例',ylim=(0,1.02))
    axes[1].text(.05,.70,f"平均 {summary['seconds_mean']:.3f} 秒\n95%的样例不超过 {summary['seconds_p95']:.3f} 秒\n最长 {summary['seconds_max']:.3f} 秒",transform=axes[1].transAxes,fontsize=11)
    for ax in axes:ax.grid(alpha=.18)
    savefig(fig,'E42_test_distribution')
    return beta


def phase_figures(rows,config):
    from datasets.public_member_inputs import public_member_inputs
    from datasets.worksets import FrozenMemberReader
    from visualization.layouts import plot_case,geometry_details
    rules=[('typical','按E42最终Q排序第500/1000例',sorted(rows,key=lambda r:(r['metrics']['cost'],r['index']))[499]),
           ('worst_quality','E42最终Q最大例',max(rows,key=lambda r:(r['metrics']['cost'],-r['index']))),
           ('slowest','E42完整耗时最大例',max(rows,key=lambda r:(r['seconds'],-r['index'])))]
    members=[r['member'] for _,_,r in rules]
    cache=ASSETS/'E42_draft_examples.json';new_calls=0
    if cache.exists():
        saved=read(cache);assert saved['checkpoint_sha256']==WEIGHT_SHA and [r['member'] for r in saved['records']]==members
        drafts={r['index']:r['raw_positions'] for r in saved['records']}
    else:
        import torch
        from experiments.E42_joint_regularization.model import GeometryModel
        from experiments.E42_joint_regularization.inference import predict
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
        assert sha(WEIGHTS)==WEIGHT_SHA
        state=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
        model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'],strict=True)
        values=[];drafts={}
        for _,_,r in rules:
            inputs=public_member_inputs(config['records'][r['index']],'local_test',authorize_local_test=True)
            raw,_=predict(model,inputs);new_calls+=1;drafts[r['index']]=raw.tolist()
            values.append({'index':r['index'],'member':r['member'],'raw_positions':raw.tolist()})
        write(cache,{'checkpoint_sha256':WEIGHT_SHA,'purpose':'Three illustration drafts reconstructed after the completed benchmark; no final solve or rescore.',
                     'generated_draft_count':3,'CPU_threads':1,'reference_used_by_model':False,'records':values})
    reader=FrozenMemberReader(members,'local_test',identity=config['registry_sha256'],allow_local_test=True)
    descriptions=[]
    for ordinal,(label,rule,r) in enumerate(rules):
        inputs=public_member_inputs(config['records'][r['index']],'local_test',authorize_local_test=True)
        constraints=np.asarray(inputs[5]).tolist()
        reference=reader.case(ordinal).reference_positions
        variants={'reference':{'positions':reference,'soft_details':geometry_details(reference,constraints)},
                  'raw':{'positions':drafts[r['index']],'soft_details':geometry_details(drafts[r['index']],constraints)},
                  'final':{'positions':r['positions'],'metrics':r['metrics'],'soft_details':geometry_details(r['positions'],constraints)}}
        row={'panel':'E42 | three phases','case_id':f'index {r["index"]}', 'constraints':constraints,'variants':variants,
             'caption':'Original reference | reconstructed frozen-model draft | saved benchmark final. Only final metrics are reported.'}
        plot_case(row,ASSETS/f'layout_{label}.png',stages=(('reference','Original reference'),('raw','E42 model draft'),('final','E42 final layout')))
        descriptions.append({'asset':f'materials/layout_{label}.png','selection_rule':rule,'index':r['index'],'member':r['member'],'n':r['n'],
            'E42_Q':r['metrics']['cost'],'E42_seconds':r['seconds'],'E42_metrics':r['metrics'],
            'final_coordinate_source':(RUN/f'cases/{r["index"]:04d}.json').relative_to(ROOT).as_posix(),
            'raw_coordinate_source':'materials/E42_draft_examples.json','reference_source':'Original selected local-test member, WHXY converted to XYWH.',
            'raw_draft_note':'Regenerated for illustration after benchmark, same frozen weights and inference source. Not a stored benchmark draft.',
            'reference_and_raw_official_scores':'not recomputed; n/a','final_solver_reexecuted':False})
    write(ASSETS/'layout_examples.json',descriptions)
    return descriptions,new_calls


def factbase(summary,config,training,extra,beta):
    facts=[]
    def add(i,topic,fact,source,limit):facts.append({'id':i,'topic':topic,'fact':fact,'evidence':source,'claim_boundary':limit})
    add('F01','最终1000例结果',summary,['evidence/E42_final_summary.json'],'本地MIB平均面积任务；不是官方隐藏集成绩。')
    add('F02','官方比较对象','Beta为100例隐藏测试；Alpha为100例公开验证集的接口演练。Beta归档包含raw_score和total_score，Alpha仅包含total_score。',['evidence/C_beta_evaluation_report_hidden_final.txt','evidence/C_beta_leaderboard_20260823.csv','evidence/C_Alpha_Top5_20260720.csv'],'用Beta作主要历史参照，不将Alpha带时间总分与本地Q直接相减。')
    add('F03','官方质量与速度',beta,['materials/official_beta_reference.json','evidence/C_beta_leaderboard_20260823.csv'],'total_runtime/100仅为归档平均时间描述。数据和硬件不同，不可据本地数值推断比赛名次或同预算领先。')
    add('F04','raw_score解释','本材料按无时间质量分理解raw_score；官方综合第3与第10名的total_score/raw_score均约0.7，与运行时间奖励下限吻合。',['materials/official_beta_reference.json','evidence/C_beta_evaluation_report_hidden_final.txt'],'归档没有对raw_score列单独下定义；这是有数值支持的工作解释，不冒充额外官方声明。')
    add('F05','官方硬件','Beta报告记载赛方为每次提交提供专用48核CPU和NVIDIA A100 80GB。',['evidence/C_beta_evaluation_report_hidden_final.txt'],'不知道各队实际用了多少资源；本地最终评价使用4个单线程CPU进程。')
    add('F06','数据变换','MIB组内取算术平均面积，保留组总面积和固定/预放置规格；仍不兼容者排除。原参考几何和评分分母保留。',['evidence/protocol.md','evidence/mean_area_and_registry.py'],'改变了面积输入及样本分布。参考列是原任务答案，不保证对变换后面积仍然硬合法。')
    add('F07','训练与测试规模','v5为训练16000、开发2000、本地测试2000例，训练/开发各固定500例监测；原始分片80/10/10不变。最终1000从余下1900中按实际模块数分层选取，排除曾看过草稿的100例。',['evidence/protocol.md','evidence/E42_final_config.json'],'开发集用于研究；本次模型评价是在同面板前序结果已知后追加，不能称新的独立盲测；剩余900例封存。')
    add('F08','输入长什么样','每块17个输入数值，包括面积、公开尺寸/位置、固定和预放置标记、边界要求、同组/同MIB数量、连接总强度及外部引脚加权平均位置；另外保留模块两两连接、同组和同MIB关系矩阵。',['evidence/E42_features.py','evidence/E42_inference.py'],'外部引脚在输入特征中被聚合，不声称网络逐一读取了全部引脚节点。自由块参考位置不进入输入。')
    add('F09','网络信息流','每块17个数映射为96个内部数值，先预测中心x/y及对数长宽比；经过12个独立参数的更新单元，每个单元按内容和当前矩形关系加权汇总其他模块，并另汇总连接/分组/MIB邻居，随后更新位置形状猜测。',['evidence/E42_model.py'],'注意力权重通过训练学习，不是保证强连接必然权重大；12层是一次前向内部的12轮，不是对新样例训练12次。')
    add('F10','矩形信息与输出','两两关系由原7维增至19维，新增当前宽高、边缘空隙、穿透及固定/预放置标记。自由块宽高由面积与长宽比恢复；固定/预放置按公开规格覆盖。模型共1471539参数。',['evidence/E42_model.py'],'参数量与隐藏宽度用于说明规模，本身不能解释方法。草稿仍可能重叠。')
    add('F11','训练事实',training,['evidence/E42_training_results.json','evidence/E42_training_config.json'],'54轮、27000步、best51、正常早停，约112.72分钟；训练记录postprocessing_calls=0不代表后续评价未执行。')
    add('F12','训练目标和模型选择','训练使用位置、形状与几何约束误差；选模综合geometry和occupancy（冲突、拥挤、空洞、松散程度），训练与监测不运行求解器。Dropout0.05、AdamW decay0.0003，单seed0、batch32。',['evidence/E42_losses.py','evidence/E42_train.py','evidence/E42_training_config.json'],'occupancy参与选模，不在本次E42反向目标中；联合正则化不能拆开归因两项的独立效应。')
    add('F13','泛化证据','相对未正则化的同结构12层，完整开发代理0.468025→0.453814，降低约3.04%；训练—开发差距缩小约34.39%。',['evidence/E42_completed_analysis.json','evidence/E42_full_dev.json'],'代理不是比赛Q；仅联合配置、单种子证据，不能用它代替最终1000例评价。')
    add('F14','几何优化器','根据当前草稿选择左右/上下分离关系，数值优化位置与形状；必要时固定宽高调整坐标、另造可行起点，再保留硬合法候选并做数值修正。',['evidence/E42_solver.py'],'算法只用公开输入目标，不用真实参考Q逐例选布局；限时优化的路径会受运行速度影响。')
    add('F15','网络和优化器各自作用',extra,['materials/derived_metrics.json'],'开发500例草稿0例硬合法，经后处理500例硬合法；最终100%合法属于整个流程，不归因于网络单独输出。')
    add('F16','速度口径',config['time_scope'],['evidence/E42_final_config.json','evidence/E42_environment.json'],'平均单例0.222522秒；批量墙钟109.14秒含初始化、IO、评分，不能除以1000替代单例延迟。原代码未更改，运行库线程钳制为1。')
    add('F17','Q解释','Q越低越好；硬非法记10；先算每例的线长/面积正向差和软违反惩罚，再按exp(模块数/12)加权。中性时间Q取消了速度奖励。',['evidence/official_problem_statement.md','evidence/scoring_evaluation.py'],'Q=1表示此评分下无正几何差和软违反，不是数学全局最优证明；Q1.028不能解释成比最优只差2.8%。')
    add('F18','三相图来源','选择最终Q排序第500例、最终Q最大例、完整耗时最大例；参考来自对应原始样本；草稿仅补生成这3例并缓存；最终布局、最终Q和耗时沿用原1000例记录。',['materials/layout_examples.json','materials/E42_draft_examples.json'],'草稿是报告阶段重建，不是原计时运行保存的草稿；参考/草稿未重新评分，图中指标n/a；没有重跑最终求解。')
    write(HERE/'factbase.json',{'date':'2026-10-24','revision':'v2: E42 and official results','facts':facts})
    lines=['# Factbase：E42方法、结果与官方参照','','日期：2026-10-24；修订v2。所有官方比较数据来自仓库references/official/results。主报告不用内部旧版本作比较。','']
    for f in facts:
        fact=f['fact'] if isinstance(f['fact'],str) else '精确字段见factbase.json对应条目和下列证据。'
        lines += [f'## {f["id"]} {f["topic"]}','',fact,'','证据：'+' · '.join(f'[{p}]({p})' for p in f['evidence']),'','解释边界：'+f['claim_boundary'],'']
    (HERE/'factbase.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    EVIDENCE.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    # Remove only superseded copied evidence/assets inside this report bundle.
    for p in EVIDENCE.iterdir():
        if p.is_file() and p.name not in COPIES:
            assert p.resolve().parent==EVIDENCE.resolve();p.unlink()
    for name in ('method_flow.png','method_flow.svg','test1000_comparison.png','test1000_comparison.svg','generalization_curves.png','block_count_results.csv'):
        p=ASSETS/name
        if p.exists():assert p.resolve().parent==ASSETS.resolve();p.unlink()
    copied=[]
    for name,source in COPIES.items():
        src=ROOT/source;dst=EVIDENCE/name;shutil.copyfile(src,dst)
        copied.append({'bundle_path':dst.relative_to(HERE).as_posix(),'repository_source':src.relative_to(ROOT).as_posix(),'sha256':sha(src)})
    s=read(RUN/'summary.json');c=read(RUN/'config.json');t=read(TRAIN/'results.json')
    rows=[read(RUN/f'cases/{i:04d}.json') for i in range(1000)]
    assert all(r['member']==m['member'] for r,m in zip(rows,c['records']))
    original=read(TRAIN/'terminal/results.json')['records']
    extra={'dev500_raw_hard_legal':sum(r['raw_hard_legal'] for r in original),'dev500_final_hard_legal':sum(r['metrics']['is_feasible'] for r in original),
        'test1000_zero_soft_violation_cases':sum(r['metrics']['total_soft_violations']==0 for r in rows),
        'test1000_total_MIB_violations':sum(r['metrics']['mib_violations'] for r in rows),
        'mean_inference_seconds':float(np.mean([r['inference_seconds'] for r in rows])),
        'mean_postprocess_seconds':float(np.mean([r['postprocess_seconds'] for r in rows]))}
    write(ASSETS/'derived_metrics.json',extra)
    network_figure();beta=result_figures(rows,s);examples,calls=phase_figures(rows,c)
    factbase(s,c,t,extra,beta)
    write(HERE/'evidence_index.json',{'date':'2026-10-24','revision':'v2','copies':copied,
        'new_draft_forward_calls_this_build':calls,'cached_illustration_drafts_total':3,
        'final_solver_calls':0,'new_official_score_calls':0,'new_training_calls':0,
        'test_members_read_for_three_phase_figures':[x['member'] for x in examples],
        'coordinate_storage':'Only new draft coordinates are stored here once. Reference and final coordinates remain in original sources.'})
    print(json.dumps({'revision':'v2','evidence_count':len(copied),'facts':18,'new_draft_forwards':calls,'final_solver_calls':0,'official_score_calls':0},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
