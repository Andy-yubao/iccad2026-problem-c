"""Plot saved Q distributions and worst layouts, with no solving or training."""
import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
import numpy as np
from .data import HERE,ROOT,reader,sha,save


def collect():
    rows=[];sources=[]
    for run in ['20261019_certified_v2_dev100_seed0','20261019_frozen_confirmation500_seed0']:
        path=HERE/'runs'/run/'results.json';sources.append(path)
        data=json.loads(path.read_text(encoding='utf-8'))
        for r in data['cases']:
            _,shard,index=r['case_id'].split(':')
            v=r['variants']['stable']
            rows.append({'case_id':r['case_id'],'split':'dev','cohort':'development600',
                         'member':[shard,int(index)],'metrics':v['metrics'],'positions':v['positions'],
                         'source':str(path.relative_to(ROOT)),'status':'evaluated'})
    bank=HERE/'runs/teacher_train16000_v1'
    path=bank/'records.jsonl';sources.append(path);unknown=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);m=r.get('metrics',r.get('rejected_metrics'))
        if m is None:
            unknown.append(r);continue
        rejected='error' in r
        label=bank/(f"rejected/{r['index']:05d}.npz" if rejected else r['label_file'])
        if not rejected and sha(label)!=r['label_sha256']:
            raise ValueError(f'Changed label: {label}')
        rows.append({'case_id':r.get('case_id',f"train:{r['member'][0]}:{r['member'][1]}"),
                     'split':'train','cohort':'training_prefix3376','member':r['member'],'metrics':m,
                     'label':label,'source':str(path.relative_to(ROOT)),
                     'status':'rejected_Q_above_2' if rejected else 'accepted_teacher'})
    assert len({r['case_id'] for r in rows})==len(rows)
    return rows,sources,unknown


def distribution(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    groups=[('development600','开发600例','#2563eb'),('training_prefix3376','训练任务前缀3376例','#e58b20')]
    fig,axes=plt.subplots(1,2,figsize=(15,6.5),gridspec_kw={'width_ratios':[1.05,1]})
    largest=max(r['metrics']['cost'] for r in rows);right=np.ceil(largest/.05)*.05+.025
    bins=np.arange(1.,right+.025,.025);stats={}
    for cohort,title,color in groups:
        values=np.array([r['metrics']['cost'] for r in rows if r['cohort']==cohort])
        n=len(values);ordered=np.sort(values)
        stats[cohort]={'count':n,'mean':float(values.mean()),'median':float(np.median(values)),
            **{f'p{p}':float(np.percentile(values,p)) for p in [90,95,99]},
            'max':float(values.max()),'above_1_8':int((values>1.8).sum()),'above_2':int((values>2).sum())}
        axes[0].hist(values,bins=bins,weights=np.ones(n)*100/n,color=color,alpha=.42,
                     edgecolor=color,linewidth=.6,label=f'{title}（n={n}）')
        x=np.unique(np.r_[np.linspace(1,right,1600),ordered])
        survival=(n-np.searchsorted(ordered,x,side='left'))/n*100
        good=survival>0
        axes[1].step(x[good],survival[good],where='post',color=color,linewidth=2,label=f'{title}（n={n}）')
        axes[1].scatter([values.max()],[100/n],color=color,s=35,zorder=5)
    for ax in axes:
        ax.set_xlim(.99,right);ax.set_xlabel('Q（越低越好）');ax.grid(axis='y',color='#cbd5e1',alpha=.5,zorder=0)
        for threshold in [1.8,2.]:ax.axvline(threshold,color='#64748b',linestyle='--',linewidth=.9)
    axes[0].set(title='整体分布：两组分别归一化',ylabel='样例占比（%）')
    axes[0].yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    axes[0].legend(loc='upper right',frameon=False)
    axes[1].set(title='右尾分布：超过各Q水平的频率',ylabel='Q ≥ 横坐标的样例占比（%）',yscale='log')
    axes[1].set_ylim(.015,115);axes[1].legend(loc='upper right',frameon=False)
    axes[1].annotate(f"新停止样例\nQ={stats['training_prefix3376']['max']:.4f}",
        xy=(stats['training_prefix3376']['max'],100/stats['training_prefix3376']['count']),
        xytext=(1.91,.11),arrowprops={'arrowstyle':'->','color':'#b91c1c'},color='#b91c1c',fontsize=10)
    text='\n'.join([f"{title}: 中位 {stats[key]['median']:.4f}｜P95 {stats[key]['p95']:.4f}｜P99 {stats[key]['p99']:.4f}｜最大 {stats[key]['max']:.4f}"
                    for key,title,_ in groups])
    fig.suptitle('E14 已保存结果的 Q 分布',fontsize=18,y=.98)
    fig.text(.5,.905,text,ha='center',va='top',fontsize=10,color='#334155',linespacing=1.8)
    fig.text(.055,.025,'开发与训练分开展示，不混合估计泛化表现。训练为遇到首个 Q>2 即停止的顺序前缀，包含该拒绝结果。\n'
             'Q 沿用原参考分母、采用中性时间；属于 MIB 修正任务的诊断，未读取封存测试、未重新求解。',fontsize=9,color='#475569')
    fig.subplots_adjust(top=.80,bottom=.17,left=.065,right=.98,wspace=.24)
    fig.savefig(out/'q_distribution.png',dpi=180,facecolor='white')
    fig.savefig(out/'q_distribution.svg',facecolor='white');plt.close(fig)
    return stats


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    rows,sources,unknown=collect();stats=distribution(rows,out)
    from visualization.layouts import plot_case,geometry_details
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    worst=sorted(rows,key=lambda r:(-r['metrics']['cost'],r['case_id']))[:10]
    tiles=[];manifest=[]
    fontpath='C:/Windows/Fonts/msyh.ttc'
    titlefont=ImageFont.truetype(fontpath,25);smallfont=ImageFont.truetype(fontpath,19)
    for rank,r in enumerate(worst,1):
        raw=reader([r['member']],r['split']).case(0)
        c=np.asarray(raw.solver_inputs[5],dtype=int).tolist()
        if 'positions' in r:positions=r['positions']
        else:
            with np.load(r['label'],allow_pickle=False) as label:positions=label['positions'].tolist()
        m=r['metrics'];name=f'worst_{rank:02d}.png'
        row={'case_id':r['case_id'],'panel':f'E14 observed worst #{rank:02d}', 'constraints':c,
             'variants':{'final':{'positions':positions,'metrics':m,'soft_details':geometry_details(positions,c)}}}
        bounds=plot_case(row,out/name,stages=(('final','E14 saved output - rejected by quality gate' if r['status'].startswith('rejected') else 'E14 saved output'),))
        image=Image.open(out/name).convert('RGB');w,h=image.size
        x,y,ww,hh=bounds['axes_bounds'][0]
        crop=image.crop((int(max(0,x-.055)*w),int(max(0,1-y-hh-.025)*h),
                         int(min(1,x+ww+.025)*w),int(min(1,1-y+.035)*h)))
        tile=Image.new('RGB',(500,650),'white');draw=ImageDraw.Draw(tile)
        colour='#b91c1c' if r['status'].startswith('rejected') else '#0f172a'
        draw.text((16,12),f"#{rank:02d}   Q = {m['cost']:.4f}",fill=colour,font=titlefont)
        draw.text((16,52),r['case_id'].removeprefix('train:'),fill='#334155',font=smallfont)
        thumb=ImageOps.contain(crop,(472,486),Image.Resampling.LANCZOS)
        tile.paste(thumb,((500-thumb.width)//2,88+(486-thumb.height)//2))
        draw.text((16,584),f"HPWL {m['hpwl_gap']:+.1%}   bbox {m['area_gap']:+.1%}",fill='#334155',font=smallfont)
        draw.text((16,616),f"B/G/M = {m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']}"+
                  ('   已拒绝' if r['status'].startswith('rejected') else ''),fill=colour,font=smallfont)
        tiles.append(tile)
        entry={k:v for k,v in r.items() if k not in ['positions','label']}
        entry.update(rank=rank,image=name,block_count=len(positions),
                     geometry_factor=1+.5*(max(0,m['hpwl_gap'])+max(0,m['area_gap'])),
                     soft_factor=float(np.exp(2*m['violations_relative'])))
        if 'label' in r:entry['saved_layout_file']=str(r['label'].relative_to(ROOT));entry['saved_layout_sha256']=sha(r['label'])
        manifest.append(entry)
        print(rank,r['case_id'],round(m['cost'],6),flush=True)
    sheet=Image.new('RGB',(2500,1380),'#e2e8f0');draw=ImageDraw.Draw(sheet)
    draw.text((25,12),'E14 全部已观察结果中 Q 最高的 10 例（含触发停止的拒绝结果）',fill='#0f172a',font=ImageFont.truetype(fontpath,32))
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%5)*500,70+(i//5)*655))
    sheet.save(out/'worst10_overview.png')
    save(out/'summary.json',{'cohorts':stats,'count':len(rows),'unscored_errors':unknown,'worst10':manifest,
        'sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'selection':'top10 descending E14 Q across all observed train/dev outputs, including quality-rejected output',
        'no_inference':True,'local_test_read':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    with (out/'all_q.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case_id','cohort','status','Q','HPWL_gap','bbox_gap','boundary','grouping','MIB'])
        for r in rows:
            m=r['metrics'];writer.writerow([r['case_id'],r['cohort'],r['status'],m['cost'],m['hpwl_gap'],m['area_gap'],m['boundary_violations'],m['grouping_violations'],m['mib_violations']])
    cards=[]
    for r in manifest:
        m=r['metrics'];cards.append(f"<section id='case{r['rank']}'><h2>#{r['rank']:02d} · Q={m['cost']:.6f}</h2><p>{html.escape(r['case_id'])} · {r['block_count']} 模块 · {'质量门槛拒绝，未入库' if r['status'].startswith('rejected') else '已生成教师标签'}</p><p>HPWL gap {m['hpwl_gap']:+.2%} · bbox gap {m['area_gap']:+.2%} · B/G/M {m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']}</p><a href='{r['image']}'><img loading='lazy' src='{r['image']}' alt='排名{r['rank']}布局'></a></section>")
    (out/'gallery.html').write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>E14 Q分布与最差10例</title><style>body{max-width:1250px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172033;background:#f4f7fb}section{background:white;padding:24px;margin:28px 0;border-radius:14px}img{max-width:100%;height:auto}h1,h2{line-height:1.3}section img{max-height:1100px;display:block;margin:auto}</style><h1>E14 Q分布与最差10例</h1><p>开发600例与已处理训练3376例分开展示；最差10按全部已观察结果的E14最终Q排名。含Q>2触发停止的拒绝结果。没有重新求解、训练或读取封存测试。</p><section><img src='q_distribution.png' alt='Q分布'></section><section><img src='worst10_overview.png' alt='最差10例总览'></section>"+''.join(cards)+"</html>",encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
