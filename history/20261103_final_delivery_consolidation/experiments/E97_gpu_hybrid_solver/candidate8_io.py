"""Loss-side full public connectivity; NN receives compressed LEGACY graph only."""
from .candidate7 import Packer as TeacherPacker
from .train_io.data import LEGACY

def network_inputs(g):return {k:g[k] for k in LEGACY}

class Packer(TeacherPacker):
    def pack(self,rows):
        _,d,t,u,count=super().pack(rows)
        return self.public.cpu,d,t,u,count
