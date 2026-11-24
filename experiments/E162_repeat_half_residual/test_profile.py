"""CPU-only checks of layout residual scaling and sequential weight reuse."""
import unittest
import torch
from torch import nn
from .model import Model,Refinement
from .public import PUBLIC

class Initial(nn.Module):
    def forward(self,g):return torch.tensor([[[[1.,1.,0.]]]])

class Step(nn.Module):
    def __init__(self,delta,nonlinear=False):
        super().__init__();self.delta=delta;self.nonlinear=nonlinear;self.calls=[]
    def forward(self,g,state,residual_scale=1.):
        self.calls.append((state.clone(),residual_scale))
        delta=torch.tensor(self.delta).expand_as(state).clone()
        if self.nonlinear:delta[...,0]=state[...,0].square()
        return state+residual_scale*delta

class KnownHead(nn.Module):
    def forward(self,h):return torch.tensor([2.,4.,.2]).expand(*h.shape[:-1],3)

class ProfileChecks(unittest.TestCase):
    def model(self,stages):
        result=Model.__new__(Model);nn.Module.__init__(result)
        result.active_stage=3;result.initial_generator=Initial();result.stages=nn.ModuleList(stages)
        return result.eval()

    def test_each_stage_is_reused_twice_and_all_real_exits_returned(self):
        stages=[Step([2.,4.,.2]),Step([4.,6.,.4]),Step([6.,8.,.6])]
        model=self.model(stages);out=model({key:None for key in PUBLIC})
        self.assertEqual(out.shape[0],7)
        self.assertTrue(torch.allclose(out[-1],torch.tensor([[[19.,28.,1.8]]])))
        for stage in stages:self.assertEqual([scale for _,scale in stage.calls],[.5,1.])
        self.assertTrue(torch.equal(stages[1].calls[0][0],out[2]))

    def test_second_pass_uses_the_half_updated_layout(self):
        first=Step([0.,0.,0.],nonlinear=True)
        out=self.model([first,Step([0.,0.,0.]),Step([0.,0.,0.])])({key:None for key in PUBLIC})
        self.assertAlmostEqual(first.calls[1][0][0,0,0].item(),1.5)
        self.assertAlmostEqual(out[2,0,0,0].item(),3.75)

    def test_actual_refinement_scales_xy_and_logratio_output_residual(self):
        stage=Refinement(4,0.,0.).eval();stage.blocks=nn.ModuleList([]);stage.head=KnownHead()
        g=dict(area=torch.tensor([[4.]]),valid=torch.ones(1,1,dtype=torch.bool),
            features=torch.zeros(1,1,16),role_features=torch.zeros(1,1,16),
            port_features=torch.zeros(1,1,4),port_valid=torch.ones(1,1,dtype=torch.bool),
            role_edges=torch.ones(1,1,2),role_hops=torch.zeros(1,1,2,dtype=torch.int16),
            role_group=torch.zeros(1,1,2))
        half=stage(g,torch.tensor([[[1.,1.,0.]]]),residual_scale=.5)
        self.assertTrue(torch.allclose(half,torch.tensor([[[3.,5.,.1]]])))

if __name__=='__main__':unittest.main()
