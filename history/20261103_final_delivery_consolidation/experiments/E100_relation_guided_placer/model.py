"""Simultaneous four-direction relations control four differentiable geometry layers."""
import math
import torch
from torch import nn
from .base_model import GeometryModel as Encoder, decode


class GeometryModel(Encoder):
    def __init__(self, width=96, layers=12, rectangle=True, ffn_dropout=.05,
                 geometry_layers=4, relation_mode='guided'):
        super().__init__(width, layers, rectangle, ffn_dropout=ffn_dropout)
        if relation_mode not in ('guided', 'auxiliary', 'direct'):
            raise ValueError('Unknown registered relation mode')
        self.geometry_layers = geometry_layers
        self.relation_mode = relation_mode
        self.relation_embed = nn.Linear(width, 24)
        self.relation_head = nn.Sequential(nn.Linear(57, 48), nn.SiLU(), nn.Linear(48, 4))

    def forward(self, g):
        allowed = {'features','adjacency','mib','group','mask','known','area',
                   'specs','constraints','valid','scale'}
        if set(g)-allowed:
            raise ValueError('Only public graph inputs may enter the network')
        valid = g['valid']
        h = self.input(g['features'])*valid[...,None]
        state = self.head(h)
        for layer in self.layers:
            h = layer(h+self.state_embed(state*g['mask']), g, state)
            state = self.head(h)
        initial = state
        p = decode(initial, g)
        center = p[...,:2]+p[...,2:]/2
        n = valid.shape[1]
        z = self.relation_embed(h)
        local = g['area'].clamp_min(1e-12).sqrt()
        pair_scale = ((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
        delta = (center[:,None,:,:]-center[:,:,None,:])/pair_scale[...,None]
        wh = p[...,2:]/local[...,None]
        pair = torch.cat((z[:,:,None,:].expand(-1,-1,n,-1),
                          z[:,None,:,:].expand(-1,n,-1,-1), delta,
                          wh[:,:,None,:].expand(-1,-1,n,-1),
                          wh[:,None,:,:].expand(-1,n,-1,-1),
                          g['adjacency'][...,None],(g['mib']>0)[...,None],
                          (g['group']>0)[...,None]), -1)
        raw = self.relation_head(pair)
        # Reversing i,j reverses left/right and below/above exactly.
        reverse = raw.transpose(1,2)
        reverse = torch.stack((reverse[...,1],reverse[...,0],reverse[...,3],reverse[...,2]),-1)
        logits = .5*(raw+reverse)
        probabilities = logits.softmax(-1)
        confidence = (1+(probabilities*probabilities.clamp_min(1e-8).log()).sum(-1)/math.log(4)).clamp(0,1)
        if self.relation_mode == 'guided':
            for _ in range(self.geometry_layers):
                state = self.geometry_step(state,g,probabilities,confidence)
        return {'state':state, 'initial_state':initial, 'relation_logits':logits,
                'relation_confidence':confidence}

    @staticmethod
    def geometry_step(state,g,probabilities,confidence):
        p = decode(state,g)
        wh = p[...,2:]
        center = p[...,:2]+wh/2
        valid = g['valid']
        n = valid.shape[1]
        ids = torch.arange(n,device=state.device)
        pairs = valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
        local = g['area'].clamp_min(1e-12).sqrt()
        movable = g['mask'][...,0]*valid
        mobility = movable/local.clamp_min(1e-6)
        share = mobility[:,:,None]/(mobility[:,:,None]+mobility[:,None,:]).clamp_min(1e-6)
        delta = center[:,None,:,:]-center[:,:,None,:]
        half = (wh[:,:,None,:]+wh[:,None,:,:])/2
        violation = torch.stack((half[...,0]-delta[...,0],half[...,0]+delta[...,0],
                                 half[...,1]-delta[...,1],half[...,1]+delta[...,1]),-1).clamp_min(0)
        # Keep uncertainty weak; sharpen confident choices without train/eval switching.
        sharp = probabilities.square().square()
        sharp = sharp/sharp.sum(-1,keepdim=True).clamp_min(1e-8)
        weights = sharp*confidence[...,None]*pairs[...,None]
        strength = weights*violation*share[...,None]
        motion = torch.stack((strength[...,1]-strength[...,0],strength[...,3]-strength[...,2]),-1).sum(2)
        active = (violation>0).to(state.dtype)
        denom = (weights*active).sum((2,3)).clamp_min(1)
        separation = motion/denom[...,None]

        # Public compressed pin total/centroid and block adjacency; no full pins enter.
        degree = g['features'][...,13].expm1().clamp_min(0)
        pin_weight = g['features'][...,14].expm1().clamp_min(0)
        connection_total = degree+pin_weight
        destination = ((g['adjacency']@center)*degree[...,None]
                       +g['features'][...,15:17]*pin_weight[...,None])/connection_total[...,None].clamp_min(1e-6)
        attraction = (destination-center)*(connection_total>0)[...,None]

        lo = p[...,:2]
        hi = lo+wh
        bblo = lo.masked_fill(~valid[...,None],float('inf')).amin(1)
        bbhi = hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
        flags = g['constraints'][...,4]
        left,right,top,bottom = [(flags&bit)!=0 for bit in (1,2,4,8)]
        boundary = torch.stack(((bblo[:,None,0]-lo[...,0])*left+(bbhi[:,None,0]-hi[...,0])*right,
                                (bblo[:,None,1]-lo[...,1])*bottom+(bbhi[:,None,1]-hi[...,1])*top),-1)
        request = .5*separation+.02*attraction+.05*boundary
        limit = .25*local[...,None]
        update = limit*torch.tanh(request/limit.clamp_min(1e-6))*movable[...,None]
        xy = (center+update).maximum(wh/2)
        xy = torch.where(movable[...,None].bool(),xy,g['known'][...,:2])
        return torch.cat((xy,state[...,2:3]),-1)*valid[...,None]
