"""Exact dyadic contact arithmetic in a checked int64 native kernel.

No fastmath or changed geometry/score tolerances. Out-of-range inputs retain
the original arbitrary-precision Python implementation. Compilation and its
startup cost must be reported separately by the resident solver initializer.
"""
import numpy as np
from .canonical_py import canonicalize as python_canonicalize
try:
    from numba import njit
    AVAILABLE=True
except ImportError:
    AVAILABLE=False
    def njit(**kwargs):
        return lambda function:function


@njit(cache=True)
def _find(parent,offset,i):
    root=i;total=np.int64(0)
    while parent[root]!=root:
        total+=offset[root];root=parent[root]
    original=total
    while parent[i]!=i:
        following=parent[i];old=offset[i];parent[i]=root;offset[i]=total
        total-=old;i=following
    return root,original


@njit(cache=True)
def _before(distance,a,b,delta,j,d0,a0,b0,delta0):
    if distance[j]!=d0:return distance[j]>d0
    if a[j]!=a0:return a[j]>a0
    if b[j]!=b0:return b[j]>b0
    return delta[j]>delta0


@njit(cache=True)
def _kernel(pos,specs,locked,preplaced,quantum,tolerance):
    n=len(pos);units=np.empty((n,4),np.int64)
    for i in range(n):
        for j in range(4):units[i,j]=np.int64(np.rint(pos[i,j]/quantum))
        if locked[i]:
            for j in (2,3):units[i,j]=np.int64(np.rint(specs[i,j]/quantum))
    for axis in (0,1):
        values=np.empty(n+2,np.int64)
        lower=units[0,axis];upper=units[0,axis]+units[0,axis+2]
        for i in range(n):
            values[i]=units[i,axis];lower=min(lower,values[i]);upper=max(upper,values[i]+units[i,axis+2])
        values[n]=lower;values[n+1]=upper
        parent=np.arange(n+2);offset=np.zeros(n+2,np.int64)
        anchored=np.zeros(n+2,np.bool_);anchors=np.zeros(n+2,np.int64)
        for i in range(n):
            if preplaced[i]:anchored[i]=True;anchors[i]=np.int64(np.rint(specs[i,axis]/quantum))
        if abs(float(lower)*quantum)<tolerance:anchored[n]=True;anchors[n]=0
        maximum=n*n+2*n;distance=np.empty(maximum);ea=np.empty(maximum,np.int64);eb=ea.copy();ed=ea.copy();count=0
        tangent=1-axis
        for a in range(n):
            for b in range(n):
                if a==b:continue
                overlap=min(pos[a,tangent]+pos[a,tangent+2],pos[b,tangent]+pos[b,tangent+2])-max(pos[a,tangent],pos[b,tangent])
                gap=(pos[b,axis]-pos[a,axis])-pos[a,axis+2]
                if overlap>tolerance and abs(gap)<tolerance:
                    distance[count]=abs(gap);ea[count]=a;eb[count]=b;ed[count]=units[a,axis+2];count+=1
        for i in range(n):
            for endpoint in (n,n+1):
                delta=np.int64(0) if endpoint==n else units[i,axis+2]
                difference=values[endpoint]-values[i]-delta
                if abs(float(difference)*quantum)<tolerance:
                    distance[count]=abs(float(difference)*quantum);ea[count]=i;eb[count]=endpoint;ed[count]=delta;count+=1
        # Stable lexicographic insertion sort; typical planar contact sets are small.
        for k in range(1,count):
            d0=distance[k];a0=ea[k];b0=eb[k];delta0=ed[k];j=k-1
            while j>=0 and _before(distance,ea,eb,ed,j,d0,a0,b0,delta0):
                distance[j+1]=distance[j];ea[j+1]=ea[j];eb[j+1]=eb[j];ed[j+1]=ed[j];j-=1
            distance[j+1]=d0;ea[j+1]=a0;eb[j+1]=b0;ed[j+1]=delta0
        for k in range(count):
            ri,di=_find(parent,offset,ea[k]);rj,dj=_find(parent,offset,eb[k]);difference=ed[k]
            if ri==rj:continue
            delta=difference+di-dj
            if anchored[ri] and anchored[rj] and anchors[rj]-anchors[ri]!=delta:continue
            parent[rj]=ri;offset[rj]=delta
            if anchored[rj]:
                anchored[ri]=True;anchors[ri]=anchors[rj]-delta;anchored[rj]=False
        roots=np.empty(n+2,np.int64);deltas=np.empty(n+2,np.int64);temporary=np.empty(n+2,np.int64)
        for i in range(n+2):roots[i],deltas[i]=_find(parent,offset,i)
        # Iterate roots in first-appearance order, as Python's component dictionary.
        completed=np.zeros(n+2,np.bool_)
        for first in range(n+2):
            root=roots[first]
            if completed[root]:continue
            completed[root]=True;members=0;required=np.int64(0);has_real=False
            for i in range(n+2):
                if roots[i]!=root:continue
                temporary[members]=values[i]-deltas[i];members+=1
                if i<n:
                    if not has_real:required=-deltas[i];has_real=True
                    else:required=max(required,-deltas[i])
            if anchored[root]:base=anchors[root]
            else:
                ordered=np.sort(temporary[:members])
                if members%2:median=float(ordered[members//2])
                else:median=(float(ordered[members//2-1])+float(ordered[members//2]))*.5
                base=np.int64(np.rint(median))
                if has_real:base=max(base,required)
            for i in range(n):
                if roots[i]==root:units[i,axis]=base+deltas[i]
    out=units.astype(np.float64)*quantum
    for i in range(n):
        if locked[i]:out[i,2:]=specs[i,2:]
        if preplaced[i]:out[i]=specs[i]
    return out


def canonicalize(pos,p,tolerance=1e-6):
    if not AVAILABLE:raise RuntimeError('Native canonicalization needs numba; no silent slow fallback')
    pos=np.ascontiguousarray(pos,dtype=np.float64);locked=p.fixed|p.preplaced
    public=[*p.specs[locked,2:].reshape(-1),*p.specs[p.preplaced,:2].reshape(-1)]
    exponent=max([36,*[float(v).as_integer_ratio()[1].bit_length()-1 for v in public]])
    quantum=2.**(-exponent)
    largest=max(float(np.abs(pos).max()),max([0.,*[abs(v) for v in public]]))/quantum
    # Bound every path sum and offset before converting arbitrary ints to int64.
    if not np.isfinite(largest) or largest*(4*p.n+8)>=2.**61:
        return python_canonicalize(pos,p,tolerance)
    return _kernel(pos,np.ascontiguousarray(p.specs,dtype=np.float64),
        np.ascontiguousarray(locked,dtype=np.bool_),np.ascontiguousarray(p.preplaced,dtype=np.bool_),quantum,tolerance)


def warmup():
    """Compile once before timed requests; caller records this initialization."""
    if not AVAILABLE:raise RuntimeError('numba is not installed')
    pos=np.array([[0.,0.,2.,2.],[2.,0.,2.,2.]])
    _kernel(pos,pos.copy(),np.array([False,False]),np.array([False,False]),2.**-36,1e-4)
