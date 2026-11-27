"""Twofold Net2Wider mapping, including per-head Q/K attention scaling.

LayerNorm statistics are preserved by uniform duplication. Dropout masks are
independent after widening; equivalence is therefore an eval-mode statement.
"""
import torch

def widen(state,model):
    result={};old_width=state['input.0.bias'].numel();new_width=model.width
    if new_width!=2*old_width:raise ValueError('Only exact uniform 2x widening supported')
    hidden=torch.arange(old_width).repeat(2)
    head=torch.cat([torch.arange(i*old_width//4,(i+1)*old_width//4).repeat(2) for i in range(4)])
    ff=torch.arange(3*old_width).repeat(2)
    local=torch.cat([hidden+i*old_width for i in range(3)])
    qkv=torch.cat([head+i*old_width for i in range(3)])
    for key,dest in model.state_dict().items():
        source=state[key]
        if source.shape==dest.shape:result[key]=source.clone();continue
        if '.qkv.' in key:
            out=source.index_select(0,qkv)
            if source.ndim==2:out=out.index_select(1,hidden)/2
            factors=torch.ones(3*new_width,dtype=out.dtype);factors[:2*new_width]=2**(-.25)
            out=out*factors.reshape((-1,)+(1,)*(source.ndim-1))
        elif source.ndim==1:
            out=source.index_select(0,ff if '.ff.0.' in key else hidden)
        else:
            row=ff if '.ff.0.' in key else hidden
            # The final head retains three output channels.
            if key=='head.2.weight':row=torch.arange(3)
            out=source.index_select(0,row)
            if key=='input.0.weight' or key=='state_embed.weight':pass
            else:
                col=local if '.local.' in key else (head if '.output.' in key else (ff if '.ff.2.' in key else hidden))
                out=out.index_select(1,col)/2
        if out.shape!=dest.shape:raise ValueError('Widening shape mismatch: '+key)
        result[key]=out
    return result
