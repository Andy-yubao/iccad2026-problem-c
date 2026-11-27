"""Isolated prototype, not imported by the active registered training code."""
import torch
from experiments.E91_e74_32k_scratch.step_control import GraphStepControl

class CapturedProjection(GraphStepControl):
    """Capture the exact old before/after operators; keep Adam outside the graph."""
    def __init__(self,model,max_relative_step=.005):
        super().__init__(model,max_relative_step)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        counter=self.clipped.clone();steps=self.steps
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            GraphStepControl.before(self)
            GraphStepControl.after(self)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.before_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.before_graph,stream=stream):GraphStepControl.before(self)
        torch.cuda.synchronize()
        self.after_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.after_graph,stream=stream):GraphStepControl.after(self)
        torch.cuda.synchronize()
        self.steps=steps
        assert torch.equal(self.clipped,counter)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    def before(self):self.before_graph.replay()
    def after(self):
        self.after_graph.replay();self.steps+=1
