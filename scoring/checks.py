"""Independent cross-check of official hard constraints, not a training loss."""
import math

def independent_hard_checks(positions, areas, constraints, reference):
    overlaps, area_bad, shape_bad, placement_bad = [], [], [], []
    invalid = []
    for i, (x, y, w, h) in enumerate(positions):
        if not all(math.isfinite(v) for v in [x, y, w, h]) or w <= 0 or h <= 0:
            invalid.append(i)
        fixed, preplaced = bool(constraints[i, 0]), bool(constraints[i, 1])
        if fixed or preplaced:
            if any(abs(positions[i][j] - reference[i][j]) > 1e-4 for j in [2, 3]):
                shape_bad.append(i)
            if preplaced and any(abs(positions[i][j] - reference[i][j]) > 1e-4 for j in [0, 1]):
                placement_bad.append(i)
        elif abs(w * h - float(areas[i])) / float(areas[i]) > .01:
            area_bad.append(i)
        for j in range(i):
            x2, y2, w2, h2 = positions[j]
            if min(x + w, x2 + w2) - max(x, x2) > 1e-6 and min(y + h, y2 + h2) - max(y, y2) > 1e-6:
                overlaps.append([j, i])
    return {'overlap_pairs': overlaps, 'area_violation_blocks': area_bad,
            'fixed_or_preplaced_shape_violations': shape_bad, 'preplaced_location_violations': placement_bad,
            'invalid_geometry_blocks': invalid,
            'dimension_or_location_violation_count': len(set(shape_bad) | set(placement_bad)),
            'is_feasible': not any([overlaps, area_bad, shape_bad, placement_bad, invalid])}
