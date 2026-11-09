"""Batch staging API preserving E74's original capture and optimizer path."""
class ResidentRows:
    def __init__(self,rows,device,max_nodes=None):
        self.rows,self.device,self.max_nodes=rows,device,max_nodes or 120
        self.indices={id(r):i for i,r in enumerate(rows)}
        self.bytes=0  # No whole-dataset VRAM allocation; staging allocated lazily.
    def batch(self,indices,out=None):return self.rows.batch(indices,self.device,fixed=True,out=out)
    def row_indices(self,rows):return [self.indices[id(r)] for r in rows]
