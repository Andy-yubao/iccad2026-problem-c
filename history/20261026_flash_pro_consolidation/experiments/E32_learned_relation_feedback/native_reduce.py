"""Exact positive-size DAG transitive reduction, <=128 nodes, no fastmath."""
import numpy as np
from numba import njit

@njit(cache=True)
def kernel(edges,n):
    out=np.empty_like(edges);count=0
    for axis in range(2):
        adjacency=np.zeros((n,n),np.bool_);degree=np.zeros(n,np.int64)
        for k in range(len(edges)):
            if edges[k,2]==axis:
                i,j=edges[k,0],edges[k,1]
                adjacency[i,j]=True;degree[j]+=1
        order=np.empty(n,np.int64);valid=True
        for at in range(n):
            selected=-1
            for i in range(n):
                if degree[i]==0:selected=i;break
            if selected<0:valid=False;break
            order[at]=selected;degree[selected]=-1
            for j in range(n):
                if adjacency[selected,j]:degree[j]-=1
        if not valid:
            for k in range(len(edges)):
                if edges[k,2]==axis:out[count]=edges[k];count+=1
            continue
        lower=np.zeros(n,np.uint64);upper=np.zeros(n,np.uint64)
        for at in range(n-1,-1,-1):
            i=order[at];lo=np.uint64(0);hi=np.uint64(0)
            for rank in range(n):
                j=order[rank]
                if not adjacency[i,j]:continue
                bit=np.uint64(1)<<np.uint64(j%64)
                covered=(lo&bit)!=0 if j<64 else (hi&bit)!=0
                if not covered:
                    out[count,0]=i;out[count,1]=j;out[count,2]=axis;count+=1
                if j<64:lo|=bit
                else:hi|=bit
                lo|=lower[j];hi|=upper[j]
            lower[i]=lo;upper[i]=hi
    return out[:count]

def reduce_edges(edges,n):
    if n>128:
        from .fast_geometry import reduce_edges as fallback
        return fallback(edges,n)
    return kernel(np.asarray(edges,dtype=np.int64).reshape(-1,3),n)

def warmup():
    kernel(np.array([[0,1,0],[1,2,0],[0,2,0]],dtype=np.int64),3)
