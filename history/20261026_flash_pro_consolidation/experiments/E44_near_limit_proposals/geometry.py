"""Fixed-dimension LP with complete separation and retractable soft structure.

HiGHS is called directly so its CPU thread limit is explicit. Bounding box
witnesses tie every modeled extremum to a real rectangle. No E25 repair code.
"""
import time
import numpy as np
import highspy
from scipy.sparse import coo_matrix
from .problem import hard_check


def realized_decisions(pos, decisions):
    """Report actual positive-length connections; tiny positive cracks fail."""
    z = np.asarray(pos)
    lo, hi = z[:, :2].min(0), (z[:, :2] + z[:, 2:]).max(0)
    contacts, boundary, failed = 0, 0, []
    for d in decisions:
        if d.kind == 'contact':
            t = 1 - d.axis
            gap = z[d.j, d.axis] - (z[d.i, d.axis] + z[d.i, d.axis + 2])
            overlap = min(z[d.i, t] + z[d.i, t + 2], z[d.j, t] + z[d.j, t + 2])
            overlap -= max(z[d.i, t], z[d.j, t])
            # An ULP-scale overlap can result from canonical edge arithmetic;
            # record it, but never count an actual positive gap as connected.
            valid = -1e-7 <= gap <= 0 and overlap > 0
            contacts += bool(valid)
        else:
            a, upper = {1: (0, False), 2: (0, True), 4: (1, True), 8: (1, False)}[d.bit]
            edge = z[d.i, a] + (z[d.i, a + 2] if upper else 0)
            valid = abs(edge - (hi[a] if upper else lo[a])) < 1e-6
            boundary += bool(valid)
        if not valid:
            failed.append(d.__dict__)
    return {'realized_contacts': contacts, 'realized_boundary_bits': boundary,
            'unrealized': failed, 'all_realized': not failed}


def canonical_edges(p, pos, decisions):
    """Share contact edge expressions in physical units, without coarse rounding.

    Only O(ULP) discrepancies are adjusted. An immutable anchor is never moved.
    Any unresolved equalities or hard failures cause the caller to reject output.
    """
    z = pos.copy()
    contacts = [d for d in decisions if d.kind == 'contact']
    limit = 1e-8 * max(1., p.scale)
    adjustments = 0
    for _ in range(3):
        changed = False
        for d in contacts:
            i, j, a = d.i, d.j, d.axis
            edge = z[i, a] + z[i, a + 2]
            gap = z[j, a] - edge
            if gap == 0 or abs(gap) > limit:
                continue
            if not p.preplaced[j]:
                z[j, a] = edge
            elif not p.preplaced[i]:
                z[i, a] = z[j, a] - z[i, a + 2]
                # Subtraction followed by addition can lose one ULP. Restore
                # closed-edge intersection; this is not a geometric repair step.
                if z[i, a] + z[i, a + 2] < z[j, a]:
                    z[i, a] = np.nextafter(z[i, a], np.inf)
            else:
                continue
            adjustments += 1
            changed = True
        if not changed:
            break
    z[p.preplaced] = p.specs[p.preplaced]
    return z, adjustments


def solve_fixed(p, seed, wh, structure, decisions, *, contact_mode='exact',
                seconds=.1, threads=4):
    started = time.perf_counter()
    structure.validate()
    if structure.n != p.n or contact_mode not in ('exact', 'soft'):
        raise ValueError('Invalid fixed-geometry contract')
    if not 1 <= threads <= 4:
        raise ValueError('Stage A uses at most four CPU execution threads')
    for d in decisions:
        if d not in structure.decisions:
            raise ValueError('Decision not registered in this structure')
        if d.kind == 'boundary' and not p.constraints[d.i, 4] & d.bit:
            raise ValueError('Boundary attachment absent from public specification')
        if d.kind == 'contact' and (p.constraints[d.i, 3] == 0 or
                p.constraints[d.i, 3] != p.constraints[d.j, 3]):
            raise ValueError('Contact is not in a public contact group')
    z, dims = seed / p.scale, np.asarray(wh) / p.scale
    costs, lower, upper, rows, rowlo, rowhi = [], [], [], [], [], []
    inf = highspy.kHighsInf
    def var(cost=0., lo=0., hi=inf):
        index = len(costs)
        costs.append(float(cost)); lower.append(float(lo)); upper.append(float(hi))
        return index
    def row(terms, lo=-inf, hi=inf):
        rows.append(terms); rowlo.append(float(lo)); rowhi.append(float(hi))
    def equal(terms, rhs):
        row(terms, rhs, rhs)
    xy = np.array([[var(), var()] for _ in range(p.n)])
    extent = (z[:, :2] + z[:, 2:]).max(0) - z[:, :2].min(0)
    # First-order bbox AREA, not a perimeter mislabeled as area. Actual bbox
    # product is recomputed by the acceptance objective after every solve.
    gradient = .5 * extent[::-1]
    lo = np.array([var(-gradient[a]) for a in (0, 1)])
    hi = np.array([var(gradient[a]) for a in (0, 1)])
    for i in range(p.n):
        for a in (0, 1):
            row({lo[a]: 1, xy[i, a]: -1}, hi=0)
            row({xy[i, a]: 1, hi[a]: -1}, hi=-dims[i, a])
            if p.preplaced[i]:
                lower[xy[i, a]] = upper[xy[i, a]] = p.specs[i, a] / p.scale
            else:
                d = var(.001 / p.n)
                row({xy[i, a]: 1, d: -1}, hi=z[i, a])
                row({xy[i, a]: -1, d: -1}, hi=-z[i, a])
    if structure.witnesses:
        for k, witness in enumerate(structure.witnesses):
            a, is_upper = k % 2, k >= 2
            equal({xy[witness, a]: 1, (hi if is_upper else lo)[a]: -1},
                  -dims[witness, a] if is_upper else 0.)
    for i, j, a in structure.separations:
        row({xy[i, a]: 1, xy[j, a]: -1}, hi=-dims[i, a])
    nsoft = max(1, len(structure.decisions))
    for d in decisions:
        if d.kind == 'boundary':
            a, is_upper = {1: (0, False), 2: (0, True), 4: (1, True), 8: (1, False)}[d.bit]
            terms = {xy[d.i, a]: 1, (hi if is_upper else lo)[a]: -1}
            rhs = -dims[d.i, a] if is_upper else 0.
            equal(terms, rhs)
        else:
            i, j, a, t = d.i, d.j, d.axis, 1 - d.axis
            terms = {xy[j, a]: 1, xy[i, a]: -1}
            shared = .01 * min(dims[i, t], dims[j, t])
            if contact_mode == 'exact':
                equal(terms, dims[i, a])
                row({xy[i, t]: 1, xy[j, t]: -1}, hi=dims[j, t] - shared)
                row({xy[j, t]: 1, xy[i, t]: -1}, hi=dims[i, t] - shared)
            else:
                # Same geometry kernel and decision list; only contact is soft.
                slack = var(2. / nsoft)
                row({**terms, slack: -1}, hi=dims[i, a])
                row({xy[i, t]: 1, xy[j, t]: -1, slack: -1}, hi=dims[j, t] - shared)
                row({xy[j, t]: 1, xy[i, t]: -1, slack: -1}, hi=dims[i, t] - shared)
    norm = max(float(p.b2b[:, 2].sum() + p.p2b[:, 2].sum()), 1e-12)
    for i, j, w in p.b2b:
        i, j = int(i), int(j)
        if i == j:
            continue
        for a in (0, 1):
            delta = (dims[i, a] - dims[j, a]) / 2
            d = var(.5 * w / norm)
            row({xy[i, a]: 1, xy[j, a]: -1, d: -1}, hi=-delta)
            row({xy[i, a]: -1, xy[j, a]: 1, d: -1}, hi=delta)
    for pin, i, w in p.p2b:
        pin, i = int(pin), int(i)
        for a in (0, 1):
            target = p.pins[pin, a] / p.scale - dims[i, a] / 2
            d = var(.5 * w / norm)
            row({xy[i, a]: 1, d: -1}, hi=target)
            row({xy[i, a]: -1, d: -1}, hi=-target)
    ii, jj, values = [], [], []
    for r, terms in enumerate(rows):
        for k, value in terms.items():
            ii.append(r); jj.append(k); values.append(value)
    matrix = coo_matrix((values, (ii, jj)), shape=(len(rows), len(costs))).tocsr()
    stats = {'variables': len(costs), 'constraints': len(rows), 'threads': threads,
             'pair_count': len(structure.separations), 'decision_count': len(decisions),
             'contact_mode': contact_mode, 'actual_bbox_witnesses': list(structure.witnesses)}
    remaining = seconds - (time.perf_counter() - started)
    if remaining <= 0:
        return None, {**stats, 'status': 'assembly_budget_exhausted',
                      'seconds': time.perf_counter() - started}
    solver = highspy.Highs()
    for key, value in (('output_flag', False), ('threads', threads), ('parallel', 'off'),
                       ('random_seed', 0), ('primal_feasibility_tolerance', 1e-9),
                       ('dual_feasibility_tolerance', 1e-9), ('time_limit', remaining)):
        if solver.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected option ' + key)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(costs), len(rows)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = costs, lower, upper
    lp.row_lower_, lp.row_upper_ = rowlo, rowhi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
    if solver.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError('HiGHS rejected fixed-geometry LP')
    status = solver.run()
    model_status = solver.getModelStatus()
    stats.update(status=solver.modelStatusToString(model_status), api_status=str(status),
                 native_seconds=solver.getRunTime(),
                 threads_option=solver.getOptionValue('threads')[1],
                 simplex_iterations=solver.getInfo().simplex_iteration_count)
    solution = solver.getSolution()
    if not solution.value_valid:
        return None, {**stats, 'seconds': time.perf_counter() - started}
    # An interrupted LP can expose a primal vector without establishing feasibility.
    # Accept it only after independent residual, hard, and selected-structure checks.
    values = np.asarray(solution.col_value)
    activity = np.asarray(matrix @ values)
    violation = max(0., float(np.max(np.asarray(rowlo) - activity)),
                    float(np.max(activity - np.asarray(rowhi))),
                    float(np.max(np.asarray(lower) - values)),
                    float(np.max(values - np.asarray(upper))))
    stats['max_normalized_constraint_violation'] = violation
    if violation > 1e-8 or model_status not in (highspy.HighsModelStatus.kOptimal,
                                               highspy.HighsModelStatus.kTimeLimit):
        return None, {**stats, 'seconds': time.perf_counter() - started}
    out = np.column_stack((values[xy] * p.scale, wh))
    out[p.preplaced] = p.specs[p.preplaced]
    numerical = (out[:, :2] < 0) & (out[:, :2] > -1e-10)
    out[:, :2][numerical] = 0.
    if contact_mode == 'exact':
        out, adjustments = canonical_edges(p, out, decisions)
    else:
        adjustments = 0
    check = hard_check(p, out)
    realized = realized_decisions(out, decisions)
    # Soft contact control still enforces every selected boundary equality.
    selected = decisions if contact_mode == 'exact' else [d for d in decisions if d.kind == 'boundary']
    structure_valid = realized_decisions(out, selected)['all_realized']
    residual = max((out[i, a] + out[i, a + 2] - out[j, a]
                    for i, j, a in structure.separations), default=0.)
    stats.update(hard_valid=check['is_feasible'], structure_valid=structure_valid,
                 actual_separation_residual=float(residual), canonical_adjustments=adjustments,
                 realized=realized, seconds=time.perf_counter() - started)
    if not check['is_feasible'] or not structure_valid or residual > 1e-6:
        stats['rejection'] = 'output_certification_failed'
        return None, stats
    return out, stats
