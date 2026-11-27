"""Finish the paused audit using saved predictions only: no model/solver calls."""
import copy,csv,hashlib,html,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from experiments.E81_multisource_refinement.data import load_rows
from experiments.E81_multisource_refinement.runs.placer_corrector_audit import OUT,PATHS,SOURCE,win_counts

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    original=read(OUT/'audit.json');result=copy.deepcopy(original)
    source=read(OUT/'source_metrics.json');rows=load_rows('dev')
    assert [list(r['member']) for r in rows]==source['members']
    public=[];effective=[];loss_area=[]
    for r in rows:
        inp=r['case'].solver_inputs;c=np.asarray(inp[5]);s=np.asarray(inp[6]);g=r['graph']
        public.append(np.asarray(inp[1],dtype=float).sum())
        effective.append(np.where(c[:,:2].any(1),s[:,2:].prod(1),inp[1]).astype(float).sum())
        loss_area.append(np.asarray(g['area'],dtype=float).sum()*float(g['scale'])**2)
    public,effective,loss_area=map(np.asarray,(public,effective,loss_area))
    raw={n:read(PATHS[n]/'full_dev.json')['records'] for n in ('E74','E79','E80','E81','E82')}
    old=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/forwards.json')['models']['E42']
    bases={'E42':old,'E74':raw['E74']};after={}
    for n in bases:
        a=np.array([r['overlap']*r['required_area'] for r in source['models'][n]])/public
        b=np.array([r['overlap_area_fraction'] for r in bases[n]])
        after[n]=float(abs(a-b).max());assert after[n]<2e-6,(n,after[n])
    p42=np.array([r['overlap_area_fraction'] for r in old]);p74=np.array([r['overlap_area_fraction'] for r in raw['E74']]);better=p74<p42-1e-6
    consistency={}
    for n in ('E81','E82'):
        a=np.array([r['overlap_area_fraction'] for r in raw[n]])
        check=np.array([r['loss_true_intersection'] for r in raw[n]])*loss_area/public
        consistency[n]=float(abs(a-check).max());assert consistency[n]<2e-6,(n,consistency[n])
        b=np.array([r['loss_P74_true_intersection'] for r in raw[n]])*loss_area/public
        result['same_case_shared_corrector'][n].update({
          'P74_corrected_overlap':float(b.mean()),'P42_corrected_overlap':float(a.mean()),
          'among_those_P74_corrected_better_equal_worse':win_counts((b-a)[better]),
          'source_difference_corrected_difference_spearman':float(spearmanr(p74-p42,b-a).statistic),
          'overlap_denominator':'sum public input areas, both sources; P74 saved Loss converted from effective graph areas'})
    result['denominator_reconciliation']={
      'finding':'Historical geometry diagnostic uses sum(public input areas). Exact-intersection Loss uses graph areas with locked public specs. The initial source reconstruction used effective required areas. These differ slightly; no checkpoint/coordinate discrepancy is needed to explain the apparent error.',
      'comparison_denominator':'sum public input areas, matching historical geometry diagnostics',
      'frame_necessary_certificate_denominator':'sum mandatory effective areas, retaining locked public dimensions',
      'max_absolute_source_difference_after_conversion':after,
      'max_absolute_P42_loss_vs_diagnostic_after_conversion':consistency,
      'effective_over_public_min_max':[float((effective/public).min()),float((effective/public).max())],
      'historical_files_changed':False,'original_audit_sha256':sha(OUT/'audit.json'),
      'all_quantile_tables':'Already used historical diagnostic denominators consistently; unchanged',
      'note':'No training/Loss change; this only reconciles diagnostic reporting.'}
    result['status']='read_only_audit_complete_reconciled';result['utc']=datetime.now(timezone.utc).isoformat()
    save(OUT/'audit_reconciled.json',result)
    with (OUT/'area_denominators.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['member','public_area','mandatory_effective_area','graph_loss_physical_area'])
        for i,m in enumerate(source['members']):w.writerow([str(m),public[i],effective[i],loss_area[i]])

    manifest=read(OUT/'four_stage_manifest.json');assert len(manifest['fixed100'])==100
    fixed=[r['id'] for r in read(PATHS['E81']/'draft_comparison100/manifest.json')['cases']]
    assert manifest['fixed100']==fixed and len(manifest['cases'])==105
    checked=0
    for case in manifest['cases']:
        axes=[]
        for n,v in case['models'].items():
            assert sha(OUT/v['image'])==v['image_sha256']
            assert len(v['axes_bounds'])==4;axes.append(np.asarray(v['axes_bounds']))
            checked+=1
        assert all(np.allclose(a,axes[0],rtol=0,atol=1e-12) for a in axes)
    assert checked==420
    for n,digest in original['source_result_sha256'].items():assert sha(PATHS[n]/'terminal/results.json')==digest
    save(OUT/'artifact_verification.json',{'images_verified':checked,'fixed100_exact':True,
      'targeted5_posthoc':manifest['targeted'],'shared_axes_verified':True,'historical_terminal_hashes_unchanged':True,
      'reconciled_audit_sha256':sha(OUT/'audit_reconciled.json'),'new_training_solver_Q_calls':0})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tables=result['models']['E81']['source_overlap_quintiles'];x=np.arange(5)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    axs[0].bar(x-.18,[r['base_mean']*100 for r in tables],.36,label='Placer E42',color='#8595a5')
    axs[0].bar(x+.18,[r['corrected_mean']*100 for r in tables],.36,label='After F81',color='#287b94')
    axs[0].set(xticks=x,xticklabels=['Best 20%','Q2','Q3','Q4','Worst 20%'],ylabel='Pair-intersection / public area (%)',title='Same 2,000 cases; 400 per group')
    axs[0].legend(frameon=False)
    y=np.array([r['overlap_area_fraction'] for r in raw['E81']])
    axs[1].hexbin(p42*100,y*100,gridsize=40,mincnt=1,cmap='Blues')
    axs[1].plot([0,18],[0,18],ls='--',color='#a65945',lw=1)
    axs[1].set(xlabel='Placer E42 overlap (%)',ylabel='After F81 overlap (%)',title='Residual error tracks source error',ylim=(0,11),xlim=(0,18))
    fig.savefig(OUT/'source_quality.png',dpi=160);plt.close(fig)

    md=[];web=[]
    def heading(t):md.append('## '+t);web.append('<h2>'+html.escape(t)+'</h2>')
    def para(t):md.append(t);web.append('<p>'+html.escape(t)+'</p>')
    def table(headers,data):
        md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in data))
        web.append('<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in data)+'</tbody></table>')
    heading('暂停状态与核查范围')
    para('实验和 problem-c 自动跟进已暂停。没有启动新训练、新候选、后处理求解或Q评分；本地测试继续封存。原best/latest、优化器、预算和历史结果保留。此前E81/E82均已完成，本次不恢复。')
    para('本次分析冻结dev2000及已有dev-monitor500结果。仅为补齐布局网络输出，做了一次冻结E42/E74公开输入推理；修正输出与后处理输出均读取原保存坐标。开发集已反复用于研究，结论不代表独立未见泛化。')
    heading('四阶段图的含义')
    table(['列','内容','边界'],[
      ['1','参考布局','公开MIB平均面积变换前的原参考，仅用于展示'],
      ['2','布局网络原始输出','解码后、进入修正F之前；E79用E74，其余用E42'],
      ['3','修正网络输出','完整F流程，包括既有MIB形状协调和框内投影，尚未进入后处理器'],
      ['4','已有后处理结果','读取此前保存结果；可能大幅重排、改变拓扑或走救援']])
    para('保留原固定100例及原顺序，另列5个事后定向反例。四套方案共420张图，每例四列及跨方案共用坐标范围、模块ID和颜色。不含评分控件；历史三阶段图不覆盖。当前布局网络就是E42时，不再重复加入一列E42。')
    heading('初始布局差，是否修不回来？')
    para('证据支持“初始布局越差，通常留下越多残余错误”，但尚不支持一个明确的“超过阈值就修不回来”的结论。当前修正器在全部2000例上都降低了成对交面积，最差一组也都改善。这里的“初始差”暂用重叠或原G描述，不能代替用户的整体布局评价。')
    table(['E42初始重叠分组','例数','布局网络','E81修正后','相对降低','修正后仍>5%'],[
      [(['最好20%','第2组','第3组','第4组','最差20%'][i]),r['count'],f"{r['base_mean']:.2%}",f"{r['corrected_mean']:.2%}",f"{r['relative_reduction_fraction']:.2%}",r['corrected_overlap_gt5percent']] for i,r in enumerate(tables)])
    para('初始重叠与E81修正后剩余重叠的Spearman相关为0.897；控制模块数、端口连接比例与图密度后的偏秩相关仍为0.795。按原G分组也有类似趋势。不过，案例本身难度、连接与锚点结构仍会同时影响两个阶段，相关关系不是因果证明。')
    para('最差初始20%在已有评分面板中有113例：布局网络直接接后处理的加权Q为1.049072，增加E81修正后为1.030484，91胜/4平/18负。这与“差输入完全没有收益”不符；仍不能保证每例改善。Q均为后处理后的值，越低越好。')
    web.append('<img class="chart" src="source_quality.png" alt="初始与修正后重叠分层及散点密度">');md.append('![初始质量与残余错误](source_quality.png)')
    heading('同一案例、同一修正F，替换布局网络')
    table(['冻结F','E42+F交面积','E74+同一F交面积','1871例更好E74输入中，修正后更好/更差'],[
      [n,f"{v['P42_corrected_overlap']:.3%}",f"{v['P74_corrected_overlap']:.3%}",f"{v['among_those_P74_corrected_better_equal_worse'][0]} / {v['among_those_P74_corrected_better_equal_worse'][2]}"] for n,v in result['same_case_shared_corrector'].items()])
    para('E74原始交面积在1871/2000例优于E42；换成E74后，同一F的平均残余交面积更低。这在同时见过两个来源的F81/F82上仍成立，支持上游质量会影响下游。不过外框也随来源改变，且E79的F只在E74上训练，不能把差异全部归因于初始重叠。更好输入也有数百例得到更差修正结果。F81/F82的E74来源只复用保存的Loss分项，没有新增该来源的终末Q。')
    heading('是否是继承的外框根本装不下？')
    para('以公开面积及固定规格计算必须保留的模块总面积A，以源预测框面积为B，E42/E74均0/2000例出现B<A；最小B/A分别为1.001284和1.006781。暂未发现“总面积不够”的简单不可行证据。但B≥A远不足以保证可装下：预放置锚点、固定长宽和通道拥挤仍可能阻止合法排布。不能据此排除框限制，也不能声称已证明存在合法解。')
    heading('布局、修正与后处理必须分别评价')
    table(['配置','草稿交面积/dev2000','已有flash加权Q/500','最大Q'],[
      ['E42布局直接接后处理','6.302%','1.025860','1.927279'],
      ['E74布局直接接后处理','5.086%','1.022511','见原保存报告'],
      ['E74+F79（E79）','2.358%','1.016383','1.738220'],
      ['E42+同一F79（E80）','2.785%','1.017329','2.200946'],
      ['E42+加宽多来源F81','2.491%','1.016242','1.825773'],
      ['E42+F82','2.445%','1.018035','2.056040']])
    para('以上布局器和修正后原始草稿仍0/2000硬合法，已保存的后处理面板500/500合法。修正器有效降低重叠，但没有取代合法化。E81相对E80精确空白/B从6.621%降到6.325%（458例改善、42例退化），500例保持自身预测框；约0.279秒平均总求解时间尚无速度优势。F81部署约171.3万参数，原E79约589.5万；加宽、多来源、联合形状与通道Loss为组合，无法单独归因于F容量。')
    para('反例164：E81→E82原始交面积1.366%→1.337%，修复后外框更紧，但主形状候选分别29/29和28/28失败，都依赖packed救援；最终分组违反6→9，线长差距64.19%→65.81%，Q1.825773→2.056040。更少重叠仍可得到更差最终解。四阶段图能揭示后处理的大幅重排，不能把第四列的合法性算作第三列网络已合法。')
    heading('统计口径核对')
    para('发现并核清一个小口径差异：历史交面积诊断以公开输入总面积归一化，训练的真实交面积Loss以包含固定规格的图面积归一化。初版补齐源输出用了后者，两者最大表观差约4.85e-5。统一为历史公开总面积后，E42/E74最大差分别为'+f"{after['E42']:.3g}/{after['E74']:.3g}"+'；初始分组表本来就采用一致的历史诊断口径，保持不变。同一F跨来源表中P74的Loss记录已换算后比较。没有改训练Loss、原结果或模型。框的面积必要条件仍按必须满足的规格面积计算。')
    heading('当前判断与后续边界')
    para('当前应保留布局模型＋修正模型的分工假设，同时把布局器的输入质量列为必须独立核查的因素。已有证据不支持继续单纯加大修正模型就能补救任意坏布局，也不支持直接认定需要推倒这条路线。先由用户检查四阶段图，区分错误的全局次序、形状不相容、框内拥堵、局部碰撞和后处理重排，再决定后续实验。未来若恢复，需要重新明确授权；本次不启动任何新训练或扰动扫描。')
    para('本核查范围内，图像来源/模型身份/历史终末哈希已核对；未实施固定同框的因果干预，也未验证完全不同布局器家族或真正不可修复的阈值。开发改善不等于正式基线升级，E42/E50仍冻结。')
    heading('产物与复现入口')
    md.append('[四阶段图册](index.html) · [核对后统计](audit_reconciled.json) · [逐例数据](case_table.csv) · [面积口径](area_denominators.csv) · [图像身份](four_stage_manifest.json) · [产物校验](artifact_verification.json)')
    web.append('<p><a href="index.html">四阶段图册</a> · <a href="audit_reconciled.json">核对后统计</a> · <a href="case_table.csv">逐例数据</a> · <a href="area_denominators.csv">面积口径</a> · <a href="four_stage_manifest.json">图像身份</a> · <a href="artifact_verification.json">产物校验</a></p>')
    para('源码：runs/placer_corrector_audit.py补齐公开源输出；runs/render_four_stage_audit.py仅用保存坐标绘图；runs/finish_paused_audit.py完成归一化核对、图像哈希校验和报告。原始audit.json与图册manifest均保留，核对后统计另存audit_reconciled.json。')
    (OUT/'report.md').write_text('# 布局模型与修正模型全面核查\n\n'+'\n\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>布局与修正模型核查</title><style>body{max-width:1100px;margin:35px auto;padding:0 22px;font:17px/1.8 system-ui;color:#203040}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:8px;border:1px solid #ced7df;text-align:left}th{background:#eef3f6}h2{margin-top:35px}.chart{width:100%}a{color:#126482}</style><h1>布局模型与修正模型：暂停后的全面核查</h1>'+''.join(web)+'</html>',encoding='utf-8')
    print(json.dumps({'images_verified':checked,'denominator_conversion_max_error':after,'same_f':result['same_case_shared_corrector'],'report':str(OUT/'report.html')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
