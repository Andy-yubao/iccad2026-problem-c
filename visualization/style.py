"""Shared display semantics; no data access, inference, repair or scoring."""
import math

PALETTE = ('#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462',
           '#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f')
GRAY = '#e5e7eb'
EDGE = '#000000'
SIDES = ((1,'L'),(2,'R'),(4,'T'),(8,'B'))

def colors(constraints):
    groups = sorted({int(f[3]) for f in constraints if f[3] > 0})
    return {str(g):PALETTE[i % len(PALETTE)] for i,g in enumerate(groups)}

def rectangles(value, n, name):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f'{name}: expected {n} rectangles')
    for row in value:
        if len(row) != 4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in row) or min(row[2:]) <= 0:
            raise ValueError(f'{name}: invalid XYWH')
    return value

def boundary_failures(p, constraints):
    lo = [min(r[j] for r in p) for j in (0,1)]
    hi = [max(r[j]+r[j+2] for r in p) for j in (0,1)]
    return [[bit for bit,d in zip((1,2,4,8),(r[0]-lo[0],hi[0]-r[0]-r[2],hi[1]-r[1]-r[3],r[1]-lo[1]))
             if int(f[4]) & bit and abs(d) >= 1e-6] for r,f in zip(p,constraints)]

def normalize(case):
    """Validate saved coordinates and add display metadata; never invent a layer."""
    c = dict(case)
    flags = c['constraints']
    n = c.get('n',len(flags))
    if not isinstance(n,int) or n <= 0 or len(flags) != n or any(len(f)!=5 for f in flags):
        raise ValueError('constraints must have N rows of [F,P,ignored,G,boundary]')
    if any(any(not isinstance(v,(int,float)) or not math.isfinite(v) or v != int(v) for v in f) for f in flags):
        raise ValueError('Constraint flags must be finite integers')
    # MIB is intentionally absent from all display metadata.
    c['constraints'] = [[int(f[0]),int(f[1]),0,int(f[3]),int(f[4])] for f in flags]
    c['n'] = n
    layers = c.get('layers',[])
    if layers and not isinstance(layers,list):
        raise ValueError('layers must be an ordered list of actual readouts')
    c['layers'] = [rectangles(p,n,f'layer {i+1}') for i,p in enumerate(layers)]
    layer_ids=c.get('layer_ids',list(range(1,len(layers)+1)))
    if (not isinstance(layer_ids,list) or len(layer_ids)!=len(layers)
        or any(type(i) is not int or i<0 for i in layer_ids)
        or any(b<=a for a,b in zip(layer_ids,layer_ids[1:]))):
        raise ValueError('layer_ids must match saved layers and increase strictly')
    c['layer_ids']=layer_ids
    if c.get('layer_labels') is not None:
        if (not isinstance(c['layer_labels'],list) or len(c['layer_labels'])!=len(layers)
            or any(not isinstance(label,str) or not label for label in c['layer_labels'])):
            raise ValueError('layer_labels must name every actual saved layer')
    if c.get('layer_losses') is not None:
        if len(c['layer_losses'])!=len(c['layers']):raise ValueError('Layer Loss count must match actual saved layers')
        if any(not isinstance(v.get('total'),(int,float)) or not math.isfinite(v['total']) for v in c['layer_losses']):
            raise ValueError('Invalid per-layer Loss')
    if layers:
        c['draft'] = c['layers'][-1]
    elif c.get('draft') is not None:
        c['draft'] = rectangles(c['draft'],n,'draft')
    else:
        raise ValueError('A saved draft or saved layers is required')
    for name in ('reference','final'):
        if c.get(name) is not None:
            c[name] = rectangles(c[name],n,name)
    stages = [p for p in (c.get('reference'),c['draft'],c.get('final')) if p is not None] + c['layers']
    lo = [min(r[j] for p in stages for r in p) for j in (0,1)]
    hi = [max(r[j]+r[j+2] for p in stages for r in p) for j in (0,1)]
    span = max(hi[j]-lo[j] for j in (0,1))*1.11
    cx,cy = ((lo[j]+hi[j])/2 for j in (0,1))
    c['bounds'] = [cx-span/2,cx+span/2,cy-span/2,cy+span/2]
    c['group_colors'] = colors(c['constraints'])
    c['boundary_status'] = {name:boundary_failures(c[name],c['constraints']) if c.get(name) is not None else None
                            for name in ('reference','draft','final')}
    c['boundary_status']['layers'] = [boundary_failures(p,c['constraints']) for p in c['layers']]
    return c
