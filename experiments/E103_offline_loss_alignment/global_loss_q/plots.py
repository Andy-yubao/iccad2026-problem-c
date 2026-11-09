"""Standalone scientific figures; full-Q view plus explicitly labeled body zoom."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RUN,KEYS

NAMES=['位置监督','形状监督','穿透深度','平均模块重叠','最差超限重叠','线长','紧凑度','边界贡献','分组贡献','MIB贡献']

def draw(summary):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    data=np.load(RUN/'analysis_arrays.npz');hold=data['split']=='check';q=data['q'][hold];models=data['models'][hold]
    labels=[('E98_original','old_loss','E98原Loss'),('current_v1','current_loss','当前模型训练Loss'),('global_fitted','new_loss','新全局拟合Loss')]
    fig,axes=plt.subplots(2,3,figsize=(16,8.7),constrained_layout=True)
    colors={'enhanced_best71':'#2878B5','current_best29':'#E78433'}
    aliases={'enhanced_best71':'E98增强best71','current_best29':'当前best29'}
    for column,(label,key,title) in enumerate(labels):
        report=summary['comparisons'][label]
        z=(data[key][hold]-report['plot_mean'])/report['plot_std']
        for row in range(2):
            ax=axes[row,column]
            for model in colors:
                active=models==model
                ax.scatter(z[active],q[active],s=13,alpha=.5,c=colors[model],label=aliases[model],edgecolors='none')
            ax.grid(alpha=.18);ax.set_xlabel('Loss标准分数（使用拟合集均值/标准差）');ax.set_ylabel('后处理Q（越低越好）')
            if row==0:
                ax.set_title(title,fontweight='bold')
                ax.set_ylim(.98,max(1.3,float(q.max())*1.03))
                ax.text(.03,.97,f"检查集 Pearson r = {report['check']['pearson']:.3f}\n检查集 Spearman ρ = {report['check']['spearman']:.3f}",
                    transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
                ax.legend(loc='upper right',fontsize=8)
            else:
                ax.set_xlim(-2,4);ax.set_ylim(.995,1.25)
                visible=int(((z>=-2)&(z<=4)&(q<=1.25)).sum())
                ax.set_title(f'主体放大：显示{visible}/600点',fontsize=10)
    fig.suptitle('跨题目Loss–Q对比｜600个未参与拟合的独立样本（两模型各300）\n上排保留全部高尾；下排仅缩放视窗。相关系数始终使用完整600点，不删除Q=1或高Q样本。',fontsize=13)
    fig.savefig(RUN/'loss_q_comparison.png',dpi=170)
    fig.savefig(RUN/'loss_q_comparison.pdf');plt.close(fig)
    stats=summary['term_statistics'];order=np.argsort([abs(s['all_descriptive']['pearson']) for s in stats])[::-1]
    y=np.arange(10);fig,axes=plt.subplots(1,2,figsize=(13,6.4),constrained_layout=True)
    axes[0].barh(y-.17,[stats[i]['all_descriptive']['pearson'] for i in order],height=.32,label='Pearson r',color='#2878B5')
    axes[0].barh(y+.17,[stats[i]['all_descriptive']['spearman'] for i in order],height=.32,label='Spearman ρ',color='#E78433')
    axes[0].set_yticks(y,[NAMES[i] for i in order]);axes[0].invert_yaxis();axes[0].axvline(0,color='grey',lw=.8)
    axes[0].set_xlabel('单项与Q的全局相关系数（全部新2000样本，描述统计）');axes[0].legend()
    values=[stats[i]['nonnegative_standardized_beta'] for i in order]
    axes[1].barh(y,values,color='#3F9B76');axes[1].set_yticks(y,[NAMES[i] for i in order]);axes[1].invert_yaxis()
    axes[1].set_xlabel('联合非负回归标准化系数（仅1400拟合样本）')
    for j,value in enumerate(values):axes[1].text(value+.006,j,f'{value:.3f}',va='center',fontsize=9)
    axes[1].set_xlim(0,max(values)*1.2+.03)
    for ax in axes:ax.grid(axis='x',alpha=.18)
    fig.suptitle('单项关联与联合拟合贡献｜二者不等同于因果重要性；共线项可能互相替代',fontsize=12)
    fig.savefig(RUN/'term_correlations.png',dpi=170);fig.savefig(RUN/'term_correlations.pdf');plt.close(fig)
