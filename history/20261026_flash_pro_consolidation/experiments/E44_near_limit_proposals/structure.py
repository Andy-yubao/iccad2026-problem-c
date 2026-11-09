"""Complete separation topology, group contact forests and actual-bbox decisions."""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Decision:
    kind: str
    i: int
    j: int = -1
    axis: int = -1
    bit: int = 0
    confidence: float = 1.


@dataclass
class Structure:
    n: int
    separations: list
    decisions: list = field(default_factory=list)
    witnesses: tuple = ()  # loX,loY,hiX,hiY block identities, NOT guessed coordinates
    origin: str = 'public_geometry'
    permutations: tuple = ()

    def validate(self):
        pairs = set()
        for i, j, a in self.separations:
            if not (0 <= i < self.n and 0 <= j < self.n and i != j and a in (0, 1)):
                raise ValueError('Invalid separation edge')
            pair = tuple(sorted((i, j)))
            if pair in pairs:
                raise ValueError('Duplicate pair separation')
            pairs.add(pair)
        if len(pairs) != self.n * (self.n - 1) // 2:
            raise ValueError('Incomplete pair separation coverage')
        if self.witnesses and (len(self.witnesses) != 4 or
                any(not 0 <= v < self.n for v in self.witnesses)):
            raise ValueError('Malformed actual bbox witnesses')
        mapping = {frozenset((i, j)): (i, j, a) for i, j, a in self.separations}
        for d in self.decisions:
            if not np.isfinite(d.confidence) or not 0 <= d.confidence <= 1:
                raise ValueError('Invalid structure confidence')
            if not 0 <= d.i < self.n:
                raise ValueError('Invalid decision block')
            if d.kind == 'contact':
                if mapping.get(frozenset((d.i, d.j))) != (d.i, d.j, d.axis):
                    raise ValueError('Contact contradicts complete pair separation')
            elif d.kind != 'boundary' or d.bit not in (1, 2, 4, 8):
                raise ValueError('Unknown structural decision')
        for axis in (0, 1):
            outgoing = [[] for _ in range(self.n)]
            degrees = [0] * self.n
            for i, j, a in self.separations:
                if a == axis:
                    outgoing[i].append(j)
                    degrees[j] += 1
            queue = [i for i in range(self.n) if not degrees[i]]
            for i in queue:
                for j in outgoing[i]:
                    degrees[j] -= 1
                    if not degrees[j]:
                        queue.append(j)
            if len(queue) != self.n:
                raise ValueError('Cyclic separation topology')
        return self


def geometry_topology(pos):
    """One direction per pair, from the largest relative separating margin.

    A legal seed supplies a feasible certificate for these fixed dimensions.
    With reference-aided geometry the certificate need not survive new areas.
    """
    z = np.asarray(pos, float)
    center, wh = z[:, :2] + z[:, 2:] / 2, z[:, 2:]
    edges = []
    for i in range(len(z)):
        for j in range(i):
            delta = center[i] - center[j]
            extent = (wh[i] + wh[j]) / 2
            axis = int(np.argmax((np.abs(delta) - extent) / extent))
            before, after = (j, i) if delta[axis] >= 0 else (i, j)
            edges.append((before, after, axis))
    return edges


def sequence_pair(first, second):
    """Two full permutations induce complete acyclic XY separation constraints."""
    first, second = list(map(int, first)), list(map(int, second))
    n = len(first)
    if sorted(first) != list(range(n)) or sorted(second) != list(range(n)):
        raise ValueError('Sequence pair must contain each block exactly once')
    rank = {v: k for k, v in enumerate(second)}
    edges = []
    for k, i in enumerate(first):
        for j in first[k + 1:]:
            edges.append((i, j, 0) if rank[i] < rank[j] else (j, i, 1))
    return edges


def propose(p, guidance, *, use_sequence_pair=False, diagnostic_reference=False):
    """Heuristic decoder; a future learned decoder must return this contract.

    The only label-enabled entry is an explicitly flagged research diagnostic.
    The deployable solver never calls that branch or receives free labels.
    """
    z = np.asarray(guidance, float)
    if z.shape != (p.n, 4) or not np.isfinite(z).all() or (z[:, 2:] <= 0).any():
        raise ValueError('Invalid structure guidance')
    permutations = ()
    if use_sequence_pair:
        centers = z[:, :2] + z[:, 2:] / 2
        permutations = (tuple(np.lexsort((np.arange(p.n), centers[:, 0] + centers[:, 1]))),
                        tuple(np.lexsort((np.arange(p.n), centers[:, 0] - centers[:, 1]))))
        edges = sequence_pair(*permutations)
    else:
        edges = geometry_topology(z)
    decisions = []
    lo, hi = z[:, :2].min(0), (z[:, :2] + z[:, 2:]).max(0)
    for i, code in enumerate(p.constraints[:, 4]):
        distances = {1: z[i, 0] - lo[0], 2: hi[0] - z[i, 0] - z[i, 2],
                     4: hi[1] - z[i, 1] - z[i, 3], 8: z[i, 1] - lo[1]}
        for bit, distance in distances.items():
            if code & bit:
                decisions.append(Decision('boundary', i, bit=bit,
                                          confidence=1 / (1 + max(0., distance) / p.scale)))
    for group in sorted(set(p.constraints[:, 3]) - {0}):
        members = set(np.flatnonzero(p.constraints[:, 3] == group).tolist())
        parent = {i: i for i in members}
        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        ranked = []
        for i, j, a in edges:
            if i not in members or j not in members:
                continue
            t = 1 - a
            gap = abs(z[j, a] - z[i, a] - z[i, a + 2])
            transverse = max(0., z[i, t] - z[j, t] - z[j, t + 2],
                             z[j, t] - z[i, t] - z[i, t + 2])
            ranked.append(((gap + transverse) / p.scale, i, j, a))
        for cost, i, j, a in sorted(ranked):
            if find(i) != find(j):
                parent[find(i)] = find(j)
                decisions.append(Decision('contact', i, j, a, confidence=1 / (1 + cost)))
    witnesses = tuple(map(int, (*z[:, :2].argmin(0), *(z[:, :2] + z[:, 2:]).argmax(0))))
    return Structure(p.n, edges, decisions, witnesses,
                     'reference_structure_diagnostic' if diagnostic_reference else 'public_geometry',
                     permutations).validate()
