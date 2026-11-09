"""Guarded second CUDA graph: capturable AdamW, identical module step limit."""
import torch

class CapturedUpdate:
    def __init__(self,optimizer,parameters,projection):
        self.optimizer=optimizer;self.projection=projection
        # Initialize state without a warmup optimizer step. Existing state is
        # preserved; only scalar step counters move from CPU to the GPU.
        for group in optimizer.param_groups:
            group['capturable']=True;group['foreach']=True
            for p in group['params']:
                state=optimizer.state[p]
                if not state:
                    state['step']=torch.zeros((),dtype=torch.float32,device=p.device)
                    state['exp_avg']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    state['exp_avg_sq']=torch.zeros_like(p,memory_format=torch.preserve_format)
                else:state['step']=state['step'].to(p.device)
        # Four predeclared LR values are captured separately on demand. Python
        # LR floats otherwise become constants in a graph, so a LR drop cannot
        # silently keep executing the old learning rate.
        self.graphs={};self.finite={};self.parameters=parameters
        self.select()

    def select(self):
        key=tuple(float(g['lr']) for g in self.optimizer.param_groups)
        if key not in self.graphs:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
            counter=self.projection.steps if self.projection else None
            with torch.cuda.graph(graph,stream=stream):
                if self.projection:self.projection.before()
                self.optimizer.step()
                if self.projection:self.projection.after()
                finite=torch.isfinite(torch.stack(torch._foreach_norm(self.parameters))).all()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            if self.projection:
                self.projection.steps=counter;self.projection.last_scales.fill_(1.)
            self.graphs[key]=graph;self.finite[key]=finite
        self.key=key

    def replay(self):
        self.select();self.graphs[self.key].replay()
        if self.projection:self.projection.steps+=1
        if not bool(self.finite[self.key]):raise RuntimeError('Nonfinite parameters after captured AdamW')
