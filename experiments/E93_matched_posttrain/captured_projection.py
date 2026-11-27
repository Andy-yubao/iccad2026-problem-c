"""Replay the original per-module bound operators without Python dispatch gaps."""
import torch
from .step_control import GraphStepControl


class CapturedProjection(GraphStepControl):
    """Same bound arithmetic and counters; AdamW remains outside CUDA graphs."""
    def __init__(self,model,max_relative_step=.005):
        super().__init__(model,max_relative_step)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        counter=self.clipped.clone();steps=self.steps
        rng=torch.cuda.get_rng_state_all()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            GraphStepControl.before(self)
            GraphStepControl.after(self)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.before_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.before_graph,stream=stream):
            GraphStepControl.before(self)
        torch.cuda.synchronize()
        self.after_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.after_graph,stream=stream):
            GraphStepControl.after(self)
        torch.cuda.synchronize()
        self.steps=steps
        assert torch.equal(self.clipped,counter)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        assert all(torch.equal(x,y) for x,y in zip(rng,torch.cuda.get_rng_state_all()))

    def before(self):self.before_graph.replay()

    def after(self):
        self.after_graph.replay()
        self.steps+=1
