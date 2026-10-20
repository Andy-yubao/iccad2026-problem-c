"""Predicted/public structure only; no teacher inputs in normal inference."""
import numpy as np
from .targets import decode_prediction
from .backend.frozen.global_refine import topology,contact_tree

def compatible(ids,p):
    ids=np.asarray(ids,dtype=int);locked=p.fixed|p.preplaced;rigid=ids[locked[ids]]
    if len(rigid):
        shape=p.specs[rigid[0],2:]
        if np.max(np.abs(p.specs[rigid,2:]-shape))>1e-8:return False
        free=ids[~locked[ids]]
        return not len(free) or bool((np.abs(shape.prod()/p.areas[free]-1)<=.0099998).all())
    return float(np.max(p.areas[ids]*.9900002))<=float(np.min(p.areas[ids]*1.0099998))

def shape_classes(geometry,p,logits=None):
    n=p.n;parent=list(range(n));members={i:[i] for i in range(n)}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j):
        a,b=find(i),find(j)
        if a==b:return
        group=members[a]+members[b]
        if not compatible(group,p):return
        parent[b]=a;members[a]=group;del members[b]
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group)
        if logits is not None:
            scored=[(float((logits[i,j]+logits[j,i])/2),int(i),int(j)) for k,i in enumerate(ids) for j in ids[:k]]
            for score,i,j in sorted(scored,reverse=True):
                if score>0.:join(i,j)
        else:
            locked=p.fixed|p.preplaced;rigid=ids[locked[ids]]
            for k,i in enumerate(rigid):
                for j in rigid[:k]:join(int(i),int(j))
            free=ids[~locked[ids]]
            for i in free:
                choices=[j for j in rigid if compatible([int(i),int(j)],p)]
                if choices:
                    j=min(choices,key=lambda j:abs(np.log(p.specs[j,2]/p.specs[j,3])-geometry[i,2]));join(int(i),int(j))
            remaining=[int(i) for i in free if not any(locked[j] for j in members[find(int(i))])]
            for k,i in enumerate(remaining):
                for j in remaining[:k]:join(i,j)
    return [find(i) for i in range(n)]

def decode(out,p,structure):
    geometry=np.asarray(out['geometry']);labels=shape_classes(geometry,p,out['mib'] if structure else None)
    seed=decode_prediction(geometry,p,labels);edges=topology(seed,p.constraints);contacts=[];dropped=0
    boundary=[];lo=seed[:,:2].min(0);hi=(seed[:,:2]+seed[:,2:]).max(0)
    for i in range(p.n):
        mask=0
        for k,(bit,a,up) in enumerate([(1,0,False),(2,0,True),(4,1,True),(8,1,False)]):
            if not p.constraints[i,4]&bit:continue
            accept=out['boundary'][i,k]>0 if structure else abs((hi[a]-seed[i,a]-seed[i,a+2]) if up else (seed[i,a]-lo[a]))<=.1*np.sqrt(p.areas[i])
            if accept:mask|=bit
        boundary.append(mask)
    if structure:
        logits=np.asarray(out['contact']);centers=seed[:,:2]+seed[:,2:]/2;ranked=[]
        for i in range(p.n):
            for j in range(i):
                if p.constraints[i,3]<=0 or p.constraints[i,3]!=p.constraints[j,3]:continue
                symmetric=(logits[i,j]+logits[j,i,[2,3,0,1,4]])/2
                prob=np.exp(symmetric-symmetric.max());prob/=prob.sum();code=int(np.argmax(prob))
                if code==4 or prob[code]<.5:continue
                a=code%2;before,after=(i,j) if code<2 else (j,i)
                if centers[before,a]>=centers[after,a]:dropped+=1;continue
                ranked.append((float(prob[code]),before,after,a))
        parent=list(range(p.n))
        def root(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for score,i,j,a in sorted(ranked,reverse=True):
            if root(i)==root(j):continue
            parent[root(i)]=root(j)
            if not (p.preplaced[i] and p.preplaced[j]):contacts.append([i,j,a])
    else:
        for i,j,a in contact_tree(seed,p,edges):
            t=1-a;gap=abs(seed[j,a]-seed[i,a]-seed[i,a+2])
            tg=max(0.,seed[i,t]-seed[j,t]-seed[j,t+2],seed[j,t]-seed[i,t]-seed[i,t+2])
            if max(gap,tg)<=.1*min(np.sqrt(p.areas[i]),np.sqrt(p.areas[j])) and not (p.preplaced[i] and p.preplaced[j]):contacts.append([int(i),int(j),int(a)])
    cert={'certified':True,'edges':edges,'contacts':contacts,'boundary':boundary,'mib_labels':labels}
    return seed,cert,{'contact_count':len(contacts),'boundary_bit_count':sum(int(v).bit_count() for v in boundary),
      'mib_class_count':len(set(labels)),'dropped_direction_conflicts':dropped,'teacher_information_used':False}
