"""Read-only hooks for visualization; caller owns the model and forward pass."""
from contextlib import contextmanager


@contextmanager
def observe_readouts(model):
    """Observe complete head modules, including a shared or independent head list."""
    result={'states':[], 'hidden':[]}
    handles=[]
    def hook(module,args,output):
        result['states'].append(output.detach().clone())
        result['hidden'].append(args[0].detach().clone())
    for head in [model.head,*list(getattr(model,'readout_heads',[]))]:
        handles.append(head.register_forward_hook(hook))
    try:
        yield result
    finally:
        for handle in handles:handle.remove()
