"""Original E57 forward architecture; only observational health scalars added."""
from .model import GeometryModel as OriginalGeometryModel,decode

class GeometryModel(OriginalGeometryModel):
    def __init__(self,*args,graph_prenorm=False,**kwargs):
        if graph_prenorm:raise ValueError('E61 explicitly forbids graph-input prenormalization')
        super().__init__(*args,**kwargs)

    def forward(self,g):
        if set(g)-{'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        h=self.input(g['features'])*g['valid'][...,None]
        state=self.head(h)
        for layer in self.layers:
            h=layer(h+self.state_embed(state*g['mask']),g,state)
            state=self.head(h)
        self.last_hidden_max=h.abs().amax()
        self.last_output_max=state.abs().amax()
        return state
