"""PNG export of the same E6 grouping/black-frame style used by the viewer."""
from pathlib import Path

from .style import EDGE, GRAY, SIDES, boundary_failures, colors, normalize

STAGES = (('reference','Reference'),('raw','Model draft'),('final','Postprocessed'))

def geometry_details(positions,constraints):
    failed = boundary_failures(positions,constraints)
    return dict(boundary=[dict(block=i,failed_bits=bits) for i,bits in enumerate(failed) if constraints[i][4]])

def plot_case(row,destination,*,stages=None,show_metrics=False):
    """Compatibility with saved evaluation callers; MIB is never displayed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    from matplotlib.patches import Patch,Rectangle
    from matplotlib.lines import Line2D
    from .style import rectangles

    stages = stages or STAGES
    flags = row['constraints']
    palette = colors(flags)
    variants = row['variants']
    frames = [variants[k]['positions'] for k,_ in stages if variants[k].get('positions') is not None]
    if not frames:
        raise ValueError('No saved rectangles to draw')
    for p in frames:
        rectangles(p,len(flags),'stage')
    allp = [r for p in frames for r in p]
    lo = [min(r[j] for r in allp) for j in (0,1)]
    hi = [max(r[j]+r[j+2] for r in allp) for j in (0,1)]
    span = max(hi[j]-lo[j] for j in (0,1))*1.11
    cx,cy = ((lo[j]+hi[j])/2 for j in (0,1))
    limits = row.get('axis_limits',[cx-span/2,cx+span/2,cy-span/2,cy+span/2])
    if len(limits)!=4 or any(limits[j*2]>lo[j] or limits[j*2+1]<hi[j] for j in (0,1)):
        raise ValueError('Shared axes would crop a rectangle')
    legend = [Patch(facecolor=color,edgecolor=EDGE,label=f'Grouping G{g}') for g,color in palette.items()]
    legend += [Patch(facecolor=GRAY,edgecolor=EDGE,label='No grouping'),Patch(facecolor='#15803d',label='L/R/T/B satisfied'),
               Patch(facecolor='#b91c1c',label='L/R/T/B violated'),Line2D([],[],color='#64748b',ls='--',label='Own layout bbox'),
               Patch(facecolor='white',label='Number: module ID; F: fixed shape; P: preplaced')]
    rows = (len(legend)+4)//5
    fig,axes = plt.subplots(1,len(stages),figsize=(6.5*len(stages),8.5+rows*.3),squeeze=False)
    fig.subplots_adjust(left=.04,right=.985,top=.88,bottom=.12+rows*.035,wspace=.18)
    for ax,(name,title) in zip(axes[0],stages):
        p = variants[name].get('positions')
        ax.set(xlim=limits[:2],ylim=limits[2:],title=title,xlabel='x',ylabel='y')
        ax.set_aspect('equal',adjustable='box')
        ax.spines[['top','right']].set_visible(False)
        if p is None:
            ax.text(.5,.5,'Not saved',transform=ax.transAxes,ha='center')
            continue
        failed = boundary_failures(p,flags)
        ownlo = [min(r[j] for r in p) for j in (0,1)]
        ownhi = [max(r[j]+r[j+2] for r in p) for j in (0,1)]
        ax.add_patch(Rectangle(ownlo,ownhi[0]-ownlo[0],ownhi[1]-ownlo[1],fill=False,edgecolor='#64748b',ls='--',lw=.8))
        for i,(r,f) in enumerate(zip(p,flags)):
            x,y,w,h = r
            ax.add_patch(Rectangle((x,y),w,h,facecolor=palette.get(str(f[3]),GRAY),edgecolor=EDGE,lw=.65))
            lines = [str(i)]
            if f[3]>0:lines.append(f'G{int(f[3])}')
            if f[0] or f[1]:lines.append(' '.join(t for yes,t in ((f[0],'F'),(f[1],'P')) if yes))
            ax.text(x+w/2,y+h/2,'\n'.join(lines),ha='center',va='center',fontsize=5.5,color='#111827',linespacing=1.0)
            locations = {1:((x,y+h/2),(3,0),'left','center'),2:((x+w,y+h/2),(-3,0),'right','center'),
                         4:((x+w/2,y+h),(0,-3),'center','top'),8:((x+w/2,y),(0,3),'center','bottom')}
            for bit,letter in SIDES:
                if not f[4]&bit:continue
                xy,offset,ha,va = locations[bit]
                ax.annotate(letter,xy,xytext=offset,textcoords='offset points',ha=ha,va=va,fontsize=5.5,color='white',weight='bold',
                            bbox=dict(boxstyle='round,pad=.1',fc='#b91c1c' if bit in failed[i] else '#15803d',ec='white',lw=.3))
    fig.suptitle(str(row.get('title',f"{row.get('panel','')} | {row.get('case_id','')}")),fontsize=14,y=.965)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.035),ncol=5,fontsize=8,frameon=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=170,facecolor='white')
    result = dict(axis_limits=limits,axes_bounds=[list(ax.get_position().bounds) for ax in axes[0]],grouping_colors=palette)
    plt.close(fig)
    return result

def render(case,destination,*,layer=None,title='Saved layouts'):
    c = normalize(case)
    if layer is not None and layer not in c['layer_ids']:
        raise ValueError('Requested intermediate layer was not saved')
    draft = c['layers'][c['layer_ids'].index(layer)] if layer is not None else c['draft']
    label = (c.get('layer_labels',[f'S{i}' for i in c['layer_ids']])[c['layer_ids'].index(layer)]
             if layer is not None else 'Model draft / final layer')
    row = dict(title=title,constraints=c['constraints'],axis_limits=c['bounds'],
               variants={key:dict(positions=p) for key,p in [('reference',c.get('reference')),('raw',draft),('final',c.get('final'))]})
    stages=(('reference',c.get('reference_label','Reference')),('raw',label),('final','Postprocessed'))
    if c.get('hide_missing_final') and c.get('final') is None:
        stages=stages[:-1]
    return plot_case(row,destination,stages=stages)

def plot_evaluation_case(row,name,destination):
    value = row['variants'][name]
    return plot_case(dict(panel=row['panel'],case_id=row['case_id'],constraints=row['constraints'],
        variants=dict(reference=dict(positions=row['reference_positions']),raw=dict(positions=value.get('raw_positions')),
                      final=dict(positions=value.get('final',{}).get('positions')))),destination)
