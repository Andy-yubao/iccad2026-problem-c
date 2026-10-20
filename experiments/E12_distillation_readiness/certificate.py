"""Teacher label extraction: actual contacts, satisfied boundaries, shape classes."""
import numpy as np


def extract(positions,p):
    from .backend.frozen.global_refine import topology
    x=np.asarray(positions);n=p.n;edges=topology(x,p.constraints)
    boundary=[];lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0)
    for i in range(n):
        mask=0
        for bit,a,up in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
            gap=hi[a]-x[i,a]-x[i,a+2] if up else x[i,a]-lo[a]
            if p.constraints[i,4]&bit and abs(gap)<1e-6:mask|=bit
        boundary.append(mask)
    labels=list(range(n));next_label=n;contacts=[];components=list(range(n))
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group);mapping={}
        for i in ids:
            key=tuple(round(float(v),8) for v in x[i,2:])
            if key not in mapping:mapping[key]=next_label;next_label+=1
            labels[i]=mapping[key]
    for group in np.unique(p.constraints[:,3]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==group);parent={int(i):int(i) for i in ids}
        def find(i):
            while parent[i]!=i:i=parent[i]
            return i
        ranked=[]
        for k,i in enumerate(ids):
            for j in ids[:k]:
                overlap=np.minimum(x[i,:2]+x[i,2:],x[j,:2]+x[j,2:])-np.maximum(x[i,:2],x[j,:2])
                touching=(overlap[0]>=0 and overlap[1]>0) or (overlap[0]>0 and overlap[1]>=0)
                if not touching:continue
                axis=int(np.argmin(np.abs(overlap)));before,after=(int(j),int(i)) if x[j,axis]+x[j,axis+2]/2 <= x[i,axis]+x[i,axis+2]/2 else (int(i),int(j))
                gap=abs(x[after,axis]-x[before,axis]-x[before,axis+2])
                ranked.append((0 if p.preplaced[i] and p.preplaced[j] else 1,gap,before,after,axis))
        for _,gap,i,j,a in sorted(ranked):
            if find(i)==find(j):continue
            parent[find(i)]=find(j)
            if not (p.preplaced[i] and p.preplaced[j]):contacts.append([i,j,a])
        for i in ids:components[i]=find(int(i))
    return {'edges':[[int(v) for v in e] for e in edges],'contacts':contacts,
            'boundary':boundary,'mib_labels':labels,'group_components':components,'certified':True}


def coordinate_shapes(widths,p,labels,family,areas,logs):
    out=widths.copy();locked=p.fixed|p.preplaced
    for label in sorted(set(labels)):
        ids=np.flatnonzero(np.asarray(labels)==label)
        if len(ids)<2:continue
        rigid=ids[locked[ids]]
        if len(rigid):out[ids]=p.specs[rigid[0],2:]
        else:
            lo=float(np.max(p.areas[ids]*.9900002));hi=float(np.min(p.areas[ids]*1.0099998))
            if lo>hi:continue
            area=float(np.clip(np.exp(np.log(areas[ids]).mean()),lo,hi));ratio=float(np.exp(logs[ids].mean()))
            out[ids]=[np.sqrt(area*ratio),np.sqrt(area/ratio)]
    return out
