"""Common readout geometry for native S2 input and its two repeated calls."""
import numpy as np


def stage_geometry(layout, areas, flags):
    lo, hi = layout[:, :2], layout[:, :2] + layout[:, 2:]
    intersections = (np.minimum(hi[:, None], hi[None, :])
                     - np.maximum(lo[:, None], lo[None, :]))
    overlap = np.triu(intersections.clip(min=0).prod(-1), k=1).sum() / areas.sum()
    tolerance = 1e-6
    contacts = np.zeros((len(layout), len(layout)), dtype=bool)
    for axis in (0, 1):
        close = np.abs(hi[:, None, axis] - lo[None, :, axis]) <= tolerance
        contacts |= (close | close.T) & (intersections[..., 1 - axis] > tolerance)
    groups = flags[:, 3]
    contacts &= np.triu((groups[:, None] == groups[None, :]) & (groups[:, None] > 0), k=1)
    distances = np.column_stack((lo[:, 0] - lo[:, 0].min(), hi[:, 0].max() - hi[:, 0],
                                 hi[:, 1].max() - hi[:, 1], lo[:, 1] - lo[:, 1].min()))
    boundary = (distances <= tolerance) & ((flags[:, 4, None] & (1, 2, 4, 8)) != 0)
    return float(overlap), contacts, boundary


def compare_geometries(l8, l10, l12, areas, flags):
    areas, flags = np.asarray(areas), np.asarray(flags, dtype=int)
    overlap8, entry_contacts, entry_boundary = stage_geometry(l8, areas, flags)
    overlap10, _, _ = stage_geometry(l10, areas, flags)
    overlap12, contacts, boundary = stage_geometry(l12, areas, flags)
    center_delta = l12[:, :2] + l12[:, 2:] / 2 - l8[:, :2] - l8[:, 2:] / 2
    position_change = np.linalg.norm(center_delta, axis=1) / np.sqrt(areas)
    ratio_change = np.abs(np.log(l12[:, 2] / l12[:, 3]) - np.log(l8[:, 2] / l8[:, 3]))
    return dict(L8_overlap=overlap8, L10_overlap=overlap10, L12_overlap=overlap12,
                mean_position_change=float(position_change.mean()),
                p95_position_change=float(np.quantile(position_change, .95)),
                mean_ratio_change=float(ratio_change.mean()),
                entry_group_contacts=int(entry_contacts.sum()),
                retained_group_contacts=int((contacts & entry_contacts).sum()),
                entry_boundaries=int(entry_boundary.sum()),
                retained_boundaries=int((boundary & entry_boundary).sum()))
