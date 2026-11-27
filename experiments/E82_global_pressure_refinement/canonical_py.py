"""Canonical binary edge arithmetic for numerical contact, never a scoring change."""
import numpy as np


def canonicalize(pos,p,tolerance=1e-6):
    out=pos.copy();locked=p.fixed|p.preplaced
    public=[*p.specs[locked,2:].reshape(-1),*p.specs[p.preplaced,:2].reshape(-1)]
    exponent=max([36,*[float(v).as_integer_ratio()[1].bit_length()-1 for v in public]])
    quantum=2.**(-exponent)
    units=[[int(round(float(v)/quantum)) for v in row] for row in out]
    # Public specifications are exactly representable on this dyadic grid.
    for i in np.flatnonzero(locked):units[i][2:]=[int(round(float(v)/quantum)) for v in p.specs[i,2:]]
    for axis in (0,1):
        n=p.n;values=[r[axis] for r in units]
        values.extend([min(values),max(units[i][axis]+units[i][axis+2] for i in range(n))])
        parent=list(range(n+2));offset=[0]*(n+2);anchor={}
        for i in np.flatnonzero(p.preplaced):anchor[int(i)]=int(round(float(p.specs[i,axis])/quantum))
        if abs(values[n]*quantum)<tolerance:anchor[n]=0
        def find(i):
            if parent[i]!=i:
                root,delta=find(parent[i]);offset[i]+=delta;parent[i]=root
            return parent[i],offset[i]
        def join(i,j,difference):
            ri,di=find(i);rj,dj=find(j)
            if ri==rj:return dj-di==difference
            delta=difference+di-dj
            if ri in anchor and rj in anchor and anchor[rj]-anchor[ri]!=delta:return False
            parent[rj]=ri;offset[rj]=delta
            if rj in anchor:
                anchor[ri]=anchor[rj]-delta;del anchor[rj]
            return True
        events=[]
        t=1-axis
        orthogonal=(np.minimum(out[:,None,t]+out[:,None,t+2],out[None,:,t]+out[None,:,t+2])-
                    np.maximum(out[:,None,t],out[None,:,t]))
        gap=out[None,:,axis]-out[:,None,axis]-out[:,None,axis+2]
        near=(orthogonal>tolerance)&(np.abs(gap)<tolerance)
        np.fill_diagonal(near,False)
        aa,bb=np.nonzero(near)
        events.extend((abs(float(gap[a,b])),int(a),int(b),units[a][axis+2]) for a,b in zip(aa,bb))
        for i in range(n):
            for endpoint,delta in ((n,0),(n+1,units[i][axis+2])):
                distance=values[endpoint]-values[i]-delta
                if abs(distance*quantum)<tolerance:events.append((abs(distance*quantum),i,endpoint,delta))
        for _,i,j,d in sorted(events):join(i,j,d)
        components={}
        for i in range(n+2):
            root,delta=find(i);components.setdefault(root,[]).append((i,delta))
        for root,members in components.items():
            base=anchor.get(root,int(round(float(np.median([values[i]-d for i,d in members])))))
            if root not in anchor:base=max(base,max(-d for i,d in members if i<n) if any(i<n for i,d in members) else base)
            for i,d in members:
                if i<n:units[i][axis]=base+d
    out=np.asarray(units,dtype=float)*quantum
    out[locked,2:]=p.specs[locked,2:];out[p.preplaced]=p.specs[p.preplaced]
    return out
