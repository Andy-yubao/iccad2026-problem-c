"""Geometry diagnostics of saved drafts and outputs."""
import numpy as np

def geometry(draft, final):
    draft, final = np.asarray(draft), np.asarray(final)
    ds = (draft[:, :2]+draft[:, 2:]).max(0)-draft[:, :2].min(0)
    fs = (final[:, :2]+final[:, 2:]).max(0)-final[:, :2].min(0)
    dc, fc = draft[:, :2]+draft[:, 2:]/2, final[:, :2]+final[:, 2:]/2
    dc, fc = dc-dc.mean(0), fc-fc.mean(0)
    slope = (dc*fc).sum(0)/np.maximum((dc*dc).sum(0), 1e-12)
    return dict(area_ratio=float(fs.prod()/ds.prod()), width_ratio=float(fs[0]/ds[0]),
                height_ratio=float(fs[1]/ds[1]), relative_distortion=float(np.sqrt(((fc-dc)**2).sum()/max((dc*dc).sum(), 1e-12))),
                affine_x=float(slope[0]), affine_y=float(slope[1]),
                affine_residual=float(np.sqrt(((fc-dc*slope)**2).sum()/max((dc*dc).sum(), 1e-12))))

