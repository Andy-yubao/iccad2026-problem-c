"""Render saved layout stages only; no models, scoring or postprocessing."""
import json
from pathlib import Path
import numpy as np

def publish(data, folder, inline_path=None, *, update=False, template_path=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=update)
    for c in data['cases']:
        assert len(c['layers'])==12 and all(len(p)==c['n'] for p in [c['reference'],c['final'],*c['layers']])
        p=np.concatenate([c['reference'],c['final'],*c['layers']]);lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0)
        mid=(lo+hi)/2;span=max(hi-lo)*1.08
        c['bounds']=[float(mid[0]-span/2),float(mid[0]+span/2),float(mid[1]-span/2),float(mid[1]+span/2)]
        # Keep the confirmed E6 grouping style; exclude MIB from display metadata.
        c['constraints']=[list(flags[:2])+[0]+list(flags[3:]) for flags in c['constraints']]
        from matplotlib import colormaps
        groups=sorted({f[3] for f in c['constraints'] if f[3]>0})
        c['group_colors']={str(g):[int(round(v*255)) for v in colormaps['Set3'](i%12)[:3]] for i,g in enumerate(groups)}
        def boundary_failures(rectangles):
            q=np.asarray(rectangles);lo=q[:,:2].min(0);hi=(q[:,:2]+q[:,2:]).max(0)
            distances=np.stack((abs(q[:,0]-lo[0]),abs(q[:,0]+q[:,2]-hi[0]),abs(q[:,1]+q[:,3]-hi[1]),abs(q[:,1]-lo[1])),axis=1)
            return [[bit for bit,distance in zip((1,2,4,8),row) if flags[4]&bit and distance>=1e-6] for row,flags in zip(distances,c['constraints'])]
        c['boundary_status']=dict(reference=boundary_failures(c['reference']),layers=[boundary_failures(p) for p in c['layers']],final=boundary_failures(c['final']))
    (folder/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def rounded(value):
        if isinstance(value,float):return round(value,5)
        if isinstance(value,list):return [rounded(v) for v in value]
        if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
        return value
    template=Path(template_path or Path(__file__).parent/'three_phase_viewer_template.html').read_text(encoding='utf-8')
    html=template.replace('__DATA__',json.dumps(rounded(data),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    assert len(html.encode('utf-8'))<1_000_000
    if inline_path is not None:
        Path(inline_path).write_text(html,encoding='utf-8')
    shell='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4+12 三相图</title><style>:root{--font-size-base:15px;--foreground:#1c2838;--background:#fff;--border:#b9c4d0;--muted-foreground:#586779;--red:#cf2537}body{font:15px/1.5 system-ui;margin:24px}.viz-controls,.viz-row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.form-label{display:grid;gap:5px}select{font:inherit;padding:6px;max-width:100%}.form-range{flex:1;min-width:160px}.form-check{display:flex;align-items:center;gap:6px}h2,h3,p{margin:0}</style>'
    (folder/'index.html').write_text(shell+html+'</html>',encoding='utf-8')
    return folder


def render_preview(data, folder, case_index=5):
    """Use the confirmed original E6 renderer with all MIB fields excluded."""
    from .layouts import plot_case
    c=data['cases'][case_index]
    stages=(('reference','Reference layout'),('raw','Draft layer12'),('final','Postprocessed output'))
    variants={}
    for name,positions,status in [('reference',c['reference'],c['boundary_status']['reference']),('raw',c['layers'][-1],c['boundary_status']['layers'][-1]),('final',c['final'],c['boundary_status']['final'])]:
        variants[name]=dict(positions=positions,soft_details=dict(boundary=[dict(block=i,failed_bits=bits) for i,bits in enumerate(status)]))
    row=dict(panel=c['group'],case_id=str(c['member']),constraints=c['constraints'],axis_limits=c['bounds'],variants=variants,caption='Same numeric module IDs and G grouping colors; F/P and boundary badges; common axes.')
    plot_case(row,Path(folder)/'worst01.png',stages=stages,show_metrics=False)
