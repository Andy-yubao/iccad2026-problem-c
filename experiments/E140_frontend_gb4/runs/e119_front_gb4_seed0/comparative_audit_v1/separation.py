"""Independent copy of frozen relation selection; no solve or repair calls."""
import numpy as np
def topology(pos,c,mode=0,array=False,group_order=False):
    n=len(pos);i,j=np.tril_indices(n,-1);center=pos[:,:2]+pos[:,2:]/2
    delta=center[i]-center[j];extent=(pos[i,2:]+pos[j,2:])/2
    axis=np.argmax((np.abs(delta)-extent)/np.maximum(extent,1e-12),axis=1)
    a,b=c[i,4].astype(int),c[j,4].astype(int)
    if mode:
        axis=np.where((a&b&3)!=0,1,np.where((a&b&12)!=0,0,axis))
    sign=delta[np.arange(len(i)),axis]>=0
    if mode:
        lower=np.where(axis==0,1,8);upper=np.where(axis==0,2,4)
        first=((a&lower)!=0)|((b&upper)!=0);second=((b&lower)!=0)|((a&upper)!=0)
        sign=np.where(first,False,np.where(second,True,sign))
    if group_order:
        macro=center.copy();groups=c[:,3].astype(int)
        for g in np.unique(groups):
            if g>0:
                ids=np.flatnonzero(groups==g);macro[ids]=center[ids].mean(0)
        separate=(groups[i]!=groups[j])|((groups[i]==0)&(groups[j]==0))
        macro_sign=(macro[i]-macro[j])[np.arange(len(i)),axis]>=0
        if mode:
            macro_sign=np.where(first,False,np.where(second,True,macro_sign))
        sign=np.where(separate,macro_sign,sign)
    result=np.column_stack((np.where(sign,j,i),np.where(sign,i,j),axis))
    return result if array else [tuple(row) for row in result.tolist()]
