"""Standalone scientific figures from continuous geometry diagnostics."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/draft_quality_20261112';sys.path.insert(0,str(ROOT))
COLORS={'reference':'#64748b','E119':'#2563eb','E120':'#e07823'}
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def finish(fig,name):
    fig.savefig(HERE/name,dpi=190,bbox_inches='tight');plt.close(fig)

def main():
    summary=read('summary.json');pooled=read('pooled.json');corr=read('correlations.json')
    gradients=read('train5_gradients.json');layers=read('train5_layers.json')
    analysis=read('analysis.json');keys=['reference_reference','E119_draft','E120_draft']
    labels=['reference','E119 草稿','E120 草稿'];colors=[COLORS[k] for k in ('reference','E119','E120')]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,key,title,scale in zip(axes.flat,['duplicate_coverage','whitespace_bbox','bbox_reference','hpwl_reference'],
        ['重复覆盖面积 / 实际模块总面积','真实留白 / 外框面积','外框面积 / reference 外框面积','加权总线长 / reference 线长'],[100,100,1,1]):
        values=[summary[k][key]['mean']*scale for k in keys]
        bars=ax.bar(labels,values,color=colors,width=.58)
        for b,v in zip(bars,values):ax.annotate(f'{v:.3f}'+('%' if scale==100 else ''),(b.get_x()+b.get_width()/2,v),xytext=(0,5),textcoords='offset points',ha='center')
        ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if scale==1:ax.axhline(1,color='#94a3b8',ls='--');ax.set_ylim(min(values)-.015,max(values)+.015)
        else:ax.set_ylim(0,max(values)*1.22+.01)
    fig.suptitle('固定 dev500：连续草稿几何面板（逐实例等权平均）',fontsize=15)
    fig.supxlabel('面积、线长偏低需结合重叠和固定规格偏差判断；不计算草稿官方评分。',fontsize=10)
    finish(fig,'geometry.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    eps=np.array([0,.005,.01,.02,.05,.1,.2,.5,1,2])
    for key,label,color in zip(keys,labels,colors):
        axes[0].plot(eps*100,[100*pooled[key]['group_curves'][str(float(e))]['connected_fraction'] for e in eps],marker='o',label=label,color=color)
        axes[1].plot(eps*100,[100*pooled[key]['boundary_curves'][str(float(e))] for e in eps],marker='o',label=label,color=color)
    for ax in axes:
        ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,5,10,20,50,100,200],['0','1','5','10','20','50','100','200'])
        ax.set_ylim(-2,102);ax.grid(alpha=.2);ax.legend();ax.set_ylabel('比例 (%)')
    axes[0].set(title='Grouping：接触瓶颈低于该尺度的组',xlabel='允许成对相对平移 / 模块尺度 (%)')
    axes[1].set(title='Boundary：最差要求方向缺口低于该尺度的模块',xlabel='贴边缺口 / 模块尺度 (%)');axes[1].set_xlim(0,50)
    fig.suptitle('几何接近程度：不是官方约束通过率',fontsize=15)
    fig.supxlabel('接触长度 = 对应短边的 1%；接触瓶颈忽略全局协同冲突。角约束按较差方向。',fontsize=10)
    finish(fig,'structure_curves.png')
    features=['duplicate_coverage','group_contact_bottleneck','group_foreign_bbox_fraction','group_tree_blocked_fraction',
        'boundary_gap_local','boundary_outside_fraction','boundary_anchor_outside_depth','fixed_dimension_relative_mean']
    names=['重复覆盖','组接触瓶颈','组外框异组占用','候选接触受阻','贴边缺口','外侧模块比例','锚点外侧突出','固定尺寸偏差']
    targets=['final_Q','final_group_violations','final_boundary_violations','move_global_mean','repair_seconds']
    tnames=['最终 Q','最终分组缺陷','最终边界缺陷','实际调整量','修复时间']
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for ax,which,title in zip(axes,['E119','E120','paired'],['E119：草稿参数→最终结果','E120：草稿参数→最终结果','同一实例：E120−E119 的配对变化']):
        source=corr['paired_deltas'] if which=='paired' else corr['per_model'][which]
        values=np.array([[source[f][t]['spearman'] if source[f][t] else np.nan for t in targets] for f in features])
        im=ax.imshow(values,vmin=-.6,vmax=.6,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(5),tnames,rotation=35,ha='right');ax.set_yticks(range(len(names)),names);ax.set_title(title)
        for i in range(len(names)):
            for j in range(5):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(values[i,j])>.4 else '#111827')
    fig.colorbar(im,ax=axes,label='Spearman 秩相关',shrink=.7)
    fig.supxlabel('全部 500 例保留；相关不等于因果。最终指标仅复用旧结果，不对草稿评分。',fontsize=10)
    finish(fig,'correlations.png')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    plots=[('common_loss','共同原 Loss',1),('duplicate_coverage','重复覆盖 (%)',100),('group_contact_bottleneck','组接触瓶颈 (% 模块尺度)',100),
        ('boundary_gap_local','边界缺口 (% 模块尺度)',100),('coordinate_clipped_fraction','坐标通道被解码裁剪 (%)',100),('hidden_gradient_norm','隐状态梯度范数',1)]
    for ax,(key,title,scale) in zip(axes.flat,plots):
        for m in ('E119','E120'):
            for case in range(5):
                rows=sorted([r for r in layers if r['model']==m and r['case']==case],key=lambda r:r['layer'])
                ax.plot(range(13),[r[key]*scale for r in rows],color=COLORS[m],alpha=.15,lw=.8)
            means=[np.mean([r[key]*scale for r in layers if r['model']==m and r['layer']==k]) for k in range(13)]
            ax.plot(range(13),means,marker='o',color=COLORS[m],label=m,lw=2)
        ax.axvline(4,color='#64748b',ls='--',alpha=.5);ax.axvline(12,color='#64748b',ls='--',alpha=.5)
        ax.set(title=title,xlabel='S0 → L1 … L12',xticks=[0,2,4,6,8,10,12]);ax.grid(alpha=.15);ax.legend()
    fig.suptitle('原固定五个训练例：粗线为均值，细线为单例',fontsize=15)
    fig.supxlabel('只在 L4、L12 直接监督；CPU/FP32/eval、零更新。中间层不是被要求单调下降的优化器。',fontsize=10)
    finish(fig,'layer_diagnostics.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    terms=['reference','overlap','grouping','boundary','wire_compact'];tn=['reference','碰撞','grouping','boundary','线长+紧凑']
    x=np.arange(5)
    for m,shift in [('E119',-.16),('E120',.16)]:
        gs=[r for r in gradients if r['model']==m]
        vals=[analysis['gradient_five_case_summary'][m]['parameter_norm_means'][t] for t in terms]
        axes[0].bar(x+shift,vals,.3,color=COLORS[m],alpha=.8,label=m)
        for i,t in enumerate(terms):axes[0].scatter(np.full(5,i+shift),[r['parameter_gradient_norms'][t]['total'] for r in gs],s=16,color=COLORS[m],edgecolor='white',linewidth=.4,zorder=5)
    axes[0].set(xticks=x,xticklabels=tn,yscale='log',title='实际加权目标对全部参数的梯度范数');axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    pairs=['overlap__reference','grouping__reference','boundary__reference','grouping__overlap','boundary__overlap']
    pn=['reference 与 碰撞','reference 与 grouping','reference 与 boundary','碰撞 与 grouping','碰撞 与 boundary']
    for m,shift in [('E119',-.14),('E120',.14)]:
        gs=[r for r in gradients if r['model']==m]
        for i,k in enumerate(pairs):
            vals=[r['parameter_gradient_cosines'][k] for r in gs]
            axes[1].scatter(vals,np.full(5,i+shift),color=COLORS[m],s=20,alpha=.6)
            axes[1].scatter(np.median(vals),i+shift,color=COLORS[m],s=70,marker='|')
    axes[1].set(yticks=range(5),yticklabels=pn,xlim=(-1.05,1.05),xlabel='梯度方向余弦（负值表示一阶冲突）',title='五例方向：点为单例，竖线为中位数');axes[1].axvline(0,color='#94a3b8',ls='--');axes[1].grid(axis='x',alpha=.2)
    fig.supxlabel('Loss 数值占比不能代替梯度占比。范数不可相加；五个训练例不代表总体。',fontsize=10)
    finish(fig,'gradients.png')
    layout_examples()
    print('figures complete',flush=True)

def layout_examples():
    cache=OUT/'visual_examples.json'
    if cache.exists():examples=json.loads(cache.read_text(encoding='utf-8'))
    else:
        from datasets.random_reference import RandomReference
        ds=RandomReference('dev_monitor500');examples=[]
        # Reuse historical audit sample list, fixed before these new metrics.
        for idx in (0,21,358,481,499):
            item=ds[idx];c=np.asarray(item['solver_inputs'][5],int).copy();c[:,2]=0
            ex=dict(index=idx,member=item['member'],constraints=c.tolist(),reference=item['supervision']['reference_xywh'].tolist())
            for m,exp,arm in [('E119','E119_global_initial_generator','global2'),('E120','E120_node12_collision','node12')]:
                r=json.loads((ROOT/'experiments'/exp/f'runs/{arm}_seed0/evaluation_no_mib_dev500/cases/{idx:04d}.json').read_text(encoding='utf-8'))
                ex[m]=dict(draft=r['draft_positions'],final=r['positions'],final_Q=r['metrics']['cost'])
            examples.append(ex)
        cache.write_text(json.dumps(examples,ensure_ascii=False)+'\n',encoding='utf-8')
    for ex in examples:
        c=np.asarray(ex['constraints']);groups=sorted(set(c[:,3])-{0});cmap=plt.get_cmap('Set3');colors={g:cmap(i%12) for i,g in enumerate(groups)}
        allp=np.concatenate([np.asarray(ex['reference'])]+[np.asarray(ex[m][s]) for m in ('E119','E120') for s in ('draft','final')])
        low=allp[:,:2].min(0);high=(allp[:,:2]+allp[:,2:]).max(0);span=max(high-low);pad=.025*span
        fig,axes=plt.subplots(2,3,figsize=(15,12))
        fig.subplots_adjust(left=.04,right=.985,bottom=.115,top=.93,hspace=.16,wspace=.12)
        for row,m in enumerate(('E119','E120')):
            for col,(p,title) in enumerate([(ex['reference'],'reference'),(ex[m]['draft'],m+' 草稿'),(ex[m]['final'],m+f" 最终布局，原 Q={ex[m]['final_Q']:.4f}")]):
                ax=axes[row,col];p=np.asarray(p)
                pl=p[:,:2].min(0);ph=(p[:,:2]+p[:,2:]).max(0)
                ax.add_patch(Rectangle(pl,*(ph-pl),fill=False,ls='--',ec='#64748b',lw=.8))
                for i,(rect,flags) in enumerate(zip(p,c)):
                    x,y,w,h=rect;fixed,pre,_,g,code=flags
                    ax.add_patch(Rectangle((x,y),w,h,facecolor=colors.get(g,'#e5e7eb'),edgecolor='#475569',lw=.45,alpha=.72))
                    text=str(i)+(f'\nG{g}' if g else '')+('\n'+('F' if fixed else '')+('P' if pre else '') if fixed or pre else '')
                    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=4.4,color='#111827',clip_on=True)
                    for bit,letter,px,py in ((1,'L',x,y+h/2),(2,'R',x+w,y+h/2),(4,'T',x+w/2,y+h),(8,'B',x+w/2,y)):
                        if code&bit:ax.text(px,py,letter,fontsize=4.5,color='white',ha='center',va='center',bbox=dict(facecolor='#334155',edgecolor='white',linewidth=.2,pad=.2),clip_on=True)
                ax.set(xlim=(low[0]-pad,high[0]+pad),ylim=(low[1]-pad,high[1]+pad),title=title);ax.set_aspect('equal');ax.tick_params(labelsize=7)
        legend=[Patch(facecolor=colors[g],label=f'G{g}') for g in groups]+[Patch(facecolor='#e5e7eb',label='无分组')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.052),ncol=len(legend),fontsize=9)
        fig.suptitle(f"dev index {ex['index']} | {len(c)} 模块 | 同尺度、同编号、颜色仅区分 G 组",fontsize=15)
        fig.text(.5,.025,'F=固定形状，P=预放置；L/R/T/B 仅标要求方向，不对草稿作官方约束评分。',ha='center',fontsize=10)
        finish(fig,f"case_{ex['index']:04d}.png")

if __name__=='__main__':main()
