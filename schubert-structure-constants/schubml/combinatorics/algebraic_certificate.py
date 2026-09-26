"""Purbhoo / Pak-Robichaux algebraic vanishing certificate for Schubert
coefficients (EXPERT_RESPONSE_4 SS9, SS12 item 9).

Mathematical background
-----------------------
Purbhoo's criterion [Purbhoo, arXiv:math/0304070, Cor. 2.6], in the algebraic
form used by Pak-Robichaux [arXiv:2509.16467, Lemma 2.3 / Cor. 2.4]: for
G = SL_n, W = S_n, n = strictly upper triangular matrices (dim d = n(n-1)/2),
N = upper unitriangular subgroup, and (in the convention VALIDATED here
against exhaustive ground truth -- see _selftest; it is the paper's
statement read with the action beta |-> x beta on position pairs)

    Inv(x) := { (a,b) : 1 <= a < b <= n,  x(a) > x(b) }   (|Inv(x)| = l(x)),
    R_x    := span{ E_ab : (a,b) in Inv(x) },

we have, for generic rho_1, ..., rho_k in N,

    sigma_{x_1} ... sigma_{x_k} != 0  in H*(Fl_n)
        <=>   rho_1 R_{x_1} rho_1^{-1} + ... + rho_k R_{x_k} rho_k^{-1}
              is a DIRECT sum inside n.

For a triple this decides c_{u,v}^w via c_{u,v}^w = c(u, v, w0.w): when
l(w) = l(u) + l(v) the three subspaces have total dimension d and

    c_{u,v}^w > 0   <=>   det M(alpha) != 0   as a polynomial,

where M(alpha) is the d x d matrix whose columns are the conjugated root
vectors.  Since  (rho E_ab rho^{-1})[s,t] = rho[s,a] * rhoinv[b,t],  each
column is an OUTER PRODUCT and its structural support is the set of rows
(s,t) with s <= a < b <= t (interval containment [a,b] subseteq [s,t]).

Projected (partial-flag) problems on the SAME ambient matrix
------------------------------------------------------------
Feas_T(E) for a cut set E is nonemptiness of the generic projected
intersection on Fl_E; via the injective ring map pi*: H*(Fl_E) -> H*(Fl_n)
it equals nonvanishing of the product of the three MINIMAL COSET
REPRESENTATIVES as full-flag classes, hence is again a column-independence
statement about columns R_x drawn from the minreps' inversion sets:

*  side="right" (the project's Feas_T(E), multicut.multistep_feasible):
   pu = minrep(u W_{P_E}) (sort positions within blocks).  Inv(pu) is NOT
   contained in Inv(u): the face systems are SEPARATE, related by the twist
   Inv(pu^F) = Inv(y) u  y^{-1} Inv(pu^E)  for  pu^F = pu^E y (lengths
   additive, y in W_{P_E}).  Face-to-face columns match only up to the
   y^{-1}-relabeling of position pairs.

*  side="left" (mirrored system = right system of the inverse triple):
   qu = minrep(W_{P_E} u) = (minrep(u^{-1} W_{P_E}))^{-1}.  The cocycle
   Inv(y qu) = Inv(qu) u qu^{-1} Inv(y)  gives PLAIN containment
   Inv(qu^E) subseteq Inv(qu^F) subseteq Inv(u), so left-faces are literal
   COLUMN SUBMATRICES of the top certificate (same rows, same rho's).
   Feas_left_T(E) = [ sigma_{qu} sigma_{qv} sigma_{qw3} != 0 in H*(Fl_n) ]
   is the reference-side (left-coset) coarsening of the same Schubert
   problem.  It is NOT the right system of the inverse triple: simultaneous
   inversion is NOT a symmetry of Schubert constants (exhaustively refuted
   in S4/S5 during this work; see H3 report).  Soundness and monotonicity
   of the left lattice are nevertheless immediate from linear algebra:
   cols_L(E) subseteq cols_L(F) subseteq cols(square M), so a dependent
   left-face forces det M == 0, i.e. c_{u,v}^w = 0, and feasibility is
   downward closed.  The full cut set [n-1] returns the square certificate,
   whose independence IS c_{u,v}^w > 0.

This left/right dichotomy is the precise functorial relation between the
full certificate and the projected problems' certificates (the expert's
warning that they are "separate systems, not literal block deletions"
holds for side="right"; the nested formulation is recovered on
side="left").  Verified empirically in _selftest and
scripts/h3_certificates.py.

Decision semantics (honest randomization statement)
---------------------------------------------------
All ranks are computed at random evaluations over F_p (p = 2^31 - 1 by
default), several seeds.  A full-rank evaluation is a DETERMINISTIC PROOF of
nonvanishing / feasibility (one nonzero specialization certifies det != 0 /
independence).  A rank-deficient evaluation at every seed certifies vanishing
only with high probability (Schwartz-Zippel); on the project corpus the zero
side is independently known exact, so agreement is what is tested.

Everything is 1-indexed one-line permutation convention, matching
schubml.combinatorics.permutations.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Perm = Tuple[int, ...]
ColKey = Tuple[int, Tuple[int, int]]   # (factor index 1/2/3, (a, b))

P_DEFAULT = (1 << 31) - 1              # Mersenne prime 2^31 - 1
P_ALT = 2147483587                     # second prime < 2^31


# ---------------------------------------------------------------------------
# small permutation helpers (self-contained, conventions of permutations.py)
# ---------------------------------------------------------------------------
def as_perm(w: Iterable[int]) -> Perm:
    return tuple(int(t) for t in w)


def pad(w: Sequence[int], n: int) -> Perm:
    w = as_perm(w)
    if len(w) > n:
        raise ValueError("cannot pad down")
    return w + tuple(range(len(w) + 1, n + 1))


def inverse(w: Sequence[int]) -> Perm:
    out = [0] * len(w)
    for i, x in enumerate(w):
        out[x - 1] = i + 1
    return tuple(out)


def length(w: Sequence[int]) -> int:
    n = len(w)
    return sum(1 for i in range(n) for j in range(i + 1, n) if w[i] > w[j])


def w0_compose(w: Sequence[int], n: Optional[int] = None) -> Perm:
    """w0 . w (left multiplication): (w0 w)(i) = n + 1 - w(i)."""
    if n is None:
        n = len(w)
    return tuple(n + 1 - t for t in pad(w, n))


def descents(w: Sequence[int]) -> Tuple[int, ...]:
    return tuple(i + 1 for i in range(len(w) - 1) if w[i] > w[i + 1])


def parabolic_min_rep(x: Sequence[int], cuts: Sequence[int], n: int) -> Perm:
    """Minimal-length representative of x W_{P_E} (sort within position
    blocks).  Identical to multicut.parabolic_min_rep."""
    xp = pad(x, n)
    bounds = [0] + sorted(set(int(c) for c in cuts)) + [n]
    out: List[int] = []
    for a, b in zip(bounds, bounds[1:]):
        out.extend(sorted(xp[a:b]))
    return tuple(out)


def parabolic_min_rep_left(x: Sequence[int], cuts: Sequence[int],
                           n: int) -> Perm:
    """Minimal-length representative of W_{P_E} x  (= (minrep(x^{-1}W))^{-1})."""
    return inverse(parabolic_min_rep(inverse(pad(x, n)), cuts, n))


def inv_set(x: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Inv(x) = {(a,b): a<b, x(a) > x(b)} (position pairs). |Inv(x)| = l(x).
    This is the column-index set of the certificate (validated convention)."""
    n = len(x)
    return tuple((a, b) for a in range(1, n)
                 for b in range(a + 1, n + 1) if x[a - 1] > x[b - 1])


def blocks_of_cuts(cuts: Sequence[int], n: int) -> List[Tuple[int, int]]:
    """Consecutive blocks [lo, hi] (1-based, inclusive) determined by cuts."""
    bounds = [0] + sorted(set(int(c) for c in cuts)) + [n]
    return [(a + 1, b) for a, b in zip(bounds, bounds[1:])]


def block_index(cuts: Sequence[int], n: int) -> List[int]:
    """map value 1..n -> block number 0..k (as list indexed by value-1)."""
    idx = [0] * n
    for bi, (lo, hi) in enumerate(blocks_of_cuts(cuts, n)):
        for v in range(lo, hi + 1):
            idx[v - 1] = bi
    return idx


def crossing_cuts(a: int, b: int, cuts: Sequence[int]) -> Tuple[int, ...]:
    """Cuts e with a <= e < b (the descent blocks a column/row (a,b) crosses)."""
    return tuple(e for e in cuts if a <= e < b)


# ---------------------------------------------------------------------------
# modular linear algebra (numpy int64, p < 2^31 so products stay < 2^63)
# ---------------------------------------------------------------------------
def _inv_mod(a: int, p: int) -> int:
    return pow(int(a) % p, p - 2, p)


def rref_mod(M: np.ndarray, p: int) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon form mod p.  Returns (R, pivot_cols)."""
    R = np.array(M, dtype=np.int64) % p
    rows, cols = R.shape
    piv: List[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(R[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            R[[r, i]] = R[[i, r]]
        R[r] = (R[r] * _inv_mod(R[r, c], p)) % p
        col = R[:, c].copy()
        col[r] = 0
        mask = col != 0
        if mask.any():
            R[mask] = (R[mask] - col[mask, None] * R[r][None, :]) % p
        piv.append(c)
        r += 1
    return R, piv


def rank_mod(M: np.ndarray, p: int) -> int:
    """Rank mod p (plain elimination, no back-substitution)."""
    A = np.array(M, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = _inv_mod(A[r, c], p)
        below = A[r + 1:, c].copy()
        mask = below != 0
        if mask.any():
            f = (below[mask] * inv) % p
            A[r + 1:][mask] = (A[r + 1:][mask] - f[:, None] * A[r][None, :]) % p
        r += 1
    return r


def right_kernel_mod(M: np.ndarray, p: int) -> np.ndarray:
    """Basis of the right kernel mod p, one vector per row of the returned
    array (shape (nullity, ncols)); reduced w.r.t. rref pivots so supports
    are canonical off a hypersurface."""
    R, piv = rref_mod(M, p)
    cols = M.shape[1]
    free = [c for c in range(cols) if c not in piv]
    out = np.zeros((len(free), cols), dtype=np.int64)
    for k, fc in enumerate(free):
        out[k, fc] = 1
        for r_i, pc in enumerate(piv):
            out[k, pc] = (-int(R[r_i, fc])) % p
    return out


def left_kernel_mod(M: np.ndarray, p: int) -> np.ndarray:
    return right_kernel_mod(np.array(M).T, p)


# ---------------------------------------------------------------------------
# certificate construction
# ---------------------------------------------------------------------------
def row_keys(n: int) -> List[Tuple[int, int]]:
    """Rows of the certificate: strictly-upper positions (s,t), lex order."""
    return [(s, t) for s in range(1, n) for t in range(s + 1, n + 1)]


def rho_pair(n: int, rng: np.random.Generator, p: int,
             identity: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Random upper unitriangular rho and its exact inverse mod p."""
    rho = np.eye(n, dtype=np.int64)
    if not identity:
        iu = np.triu_indices(n, 1)
        rho[iu] = rng.integers(1, p, size=len(iu[0]))
    # invert unitriangular by back substitution: rhoinv = I - kappa*rhoinv'
    inv = np.eye(n, dtype=np.int64)
    for c in range(n):
        for r in range(c - 1, -1, -1):
            s = 0
            for k in range(r + 1, c + 1):
                s = (s + rho[r, k] * inv[k, c]) % p
            inv[r, c] = (-s) % p
    # exact verification (object dtype: int64 matmul would overflow for n>=9)
    prod = (rho.astype(object) @ inv.astype(object)) % p
    assert (prod == np.eye(n, dtype=object)).all()
    return rho, inv


def triple_factors(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                   n: Optional[int] = None) -> Tuple[int, Perm, Perm, Perm]:
    """Common ambient size and the three factors (u, v, w0.w)."""
    if n is None:
        n = max(len(u), len(v), len(w))
    u, v, w = pad(u, n), pad(v, n), pad(w, n)
    return n, u, v, w0_compose(w, n)


class Certificate:
    """The full-flag Purbhoo certificate of a triple, at fixed rho's.

    Rows: all (s,t), s<t (dimension d).  Columns col_keys: (i, (a,b)) for
    i = 1,2,3 (factors u, v, w0.w) and (a,b) in Inv(factor_i).
    M[(s,t), (i,(a,b))] = rho_i[s,a] * rhoinv_i[b,t]  (mod p).
    """

    def __init__(self, u: Sequence[int], v: Sequence[int], w: Sequence[int],
                 seed: int = 0, p: int = P_DEFAULT, n: Optional[int] = None,
                 gauge_first_identity: bool = False):
        self.p = int(p)
        self.seed = int(seed)
        self.n, self.u, self.v, self.w3 = triple_factors(u, v, w, n)
        self.w = pad(w, self.n)
        n_ = self.n
        self.d = n_ * (n_ - 1) // 2
        self.rows = row_keys(n_)
        self.row_pos = {rt: i for i, rt in enumerate(self.rows)}
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_, 0x9E77]))
        self.rhos = []
        for i in range(3):
            self.rhos.append(rho_pair(
                n_, rng, self.p,
                identity=(gauge_first_identity and i == 0)))
        self.factors: Tuple[Perm, Perm, Perm] = (self.u, self.v, self.w3)
        self.phis = [inv_set(x) for x in self.factors]
        self.col_keys: List[ColKey] = [(i + 1, ab)
                                       for i in range(3)
                                       for ab in self.phis[i]]
        self.col_pos: Dict[ColKey, int] = {k: j
                                           for j, k in enumerate(self.col_keys)}
        # build matrix
        M = np.zeros((self.d, len(self.col_keys)), dtype=np.int64)
        su = np.triu_indices(n_, 1)          # rows (s,t) in lex order == triu order
        for j, (i, (a, b)) in enumerate(self.col_keys):
            rho, rhoinv = self.rhos[i - 1]
            outer = (rho[:, a - 1][:, None] * rhoinv[b - 1, :][None, :]) % self.p
            M[:, j] = outer[su]
        self.M = M

    # -- column selections -------------------------------------------------
    def face_col_keys(self, cuts: Sequence[int],
                      side: str = "right") -> List[ColKey]:
        """Columns of the projected certificate for cut set E.

        side="right": minreps of x W_{P_E} -> the project's Feas_T(E); the
        keys are generally NOT a subset of self.col_keys (twisted system),
        so the returned matrix must be built fresh via face_matrix.
        side="left": minreps of W_{P_E} x -> nested subsystem; keys are
        always a subset of self.col_keys (cocycle containment).
        """
        minrep = (parabolic_min_rep if side == "right"
                  else parabolic_min_rep_left)
        keys: List[ColKey] = []
        for i, x in enumerate(self.factors):
            px = minrep(x, cuts, self.n)
            for ab in inv_set(px):
                keys.append((i + 1, ab))
        return keys

    def column_vector(self, key: ColKey) -> np.ndarray:
        """Column for an arbitrary key (i,(a,b)), not necessarily in
        col_keys (needed for the twisted right-side faces)."""
        i, (a, b) = key
        rho, rhoinv = self.rhos[i - 1]
        outer = (rho[:, a - 1][:, None] * rhoinv[b - 1, :][None, :]) % self.p
        return outer[np.triu_indices(self.n, 1)]

    def submatrix(self, keys: Sequence[ColKey]) -> np.ndarray:
        cols = []
        for k in keys:
            j = self.col_pos.get(k)
            cols.append(self.M[:, j] if j is not None
                        else self.column_vector(k))
        return np.stack(cols, axis=1) if cols else np.zeros((self.d, 0),
                                                            dtype=np.int64)

    def face_matrix(self, cuts: Sequence[int],
                    side: str = "right") -> Tuple[np.ndarray, List[ColKey]]:
        keys = self.face_col_keys(cuts, side=side)
        return self.submatrix(keys), keys

    # -- decisions ---------------------------------------------------------
    def square_rank(self) -> int:
        return rank_mod(self.M, self.p)

    def face_full_rank(self, cuts: Sequence[int], side: str = "right") -> bool:
        Msub, keys = self.face_matrix(cuts, side=side)
        return rank_mod(Msub, self.p) == len(keys)

    def dependent(self, keys: Sequence[ColKey]) -> bool:
        Msub = self.submatrix(keys)
        return rank_mod(Msub, self.p) < len(keys)

    def find_circuit(self, keys: Sequence[ColKey],
                     order_seed: Optional[int] = None) -> List[ColKey]:
        """Extract one circuit (minimal dependent set) from a dependent set
        of columns by greedy removal.  Deterministic given order_seed."""
        cur = list(keys)
        assert self.dependent(cur), "find_circuit requires a dependent set"
        # shrink to the support of one kernel vector first (cheap)
        ker = right_kernel_mod(self.submatrix(cur), self.p)
        kv = ker[0]
        cur = [k for k, x in zip(cur, kv) if x % self.p]
        if order_seed is not None:
            rng = np.random.default_rng(order_seed)
            perm = rng.permutation(len(cur))
            cur = [cur[i] for i in perm]
        changed = True
        while changed:
            changed = False
            for k in list(cur):
                trial = [c for c in cur if c != k]
                if self.dependent(trial):
                    cur = trial
                    changed = True
        return sorted(cur)


# ---------------------------------------------------------------------------
# multi-seed wrappers (rank = max over seeds; full rank at any seed is proof)
# ---------------------------------------------------------------------------
def cert_nonvanishing(u, v, w, n=None, seeds=(0, 1, 2), p=P_DEFAULT) -> bool:
    """True  => c_{u,v}^w > 0 (deterministic: a nonzero specialization).
    False => c_{u,v}^w = 0 with error prob <= (poly deg / p)^len(seeds)."""
    n_, uu, vv, w3 = triple_factors(u, v, w, n)
    d = n_ * (n_ - 1) // 2
    if length(uu) + length(vv) + length(w3) != d:
        return False                       # degree obstruction: exact zero
    for s in seeds:
        C = Certificate(u, v, w, seed=s, p=p, n=n)
        if C.square_rank() == d:
            return True
    return False


def cert_face_feasible(u, v, w, cuts, n=None, seeds=(0, 1, 2),
                       p=P_DEFAULT, side: str = "right") -> bool:
    """Feas_T(E) via the certificate (column independence of the face)."""
    for s in seeds:
        C = Certificate(u, v, w, seed=s, p=p, n=n)
        if C.face_full_rank(cuts, side=side):
            return True
    return False


# ---------------------------------------------------------------------------
# structural rank (transversal matroid of the support hypergraph)
# ---------------------------------------------------------------------------
def structural_rank(n: int, col_keys: Sequence[ColKey]) -> int:
    """Maximum bipartite matching between rows (s,t) and columns (i,(a,b)),
    edge iff s <= a and b <= t (entry generically nonzero).  Equals the
    maximum rank of the certificate over ALL parameter values."""
    rows = row_keys(n)
    radj: List[List[int]] = []
    for (i, (a, b)) in col_keys:
        radj.append([ri for ri, (s, t) in enumerate(rows) if s <= a and b <= t])
    match_row = [-1] * len(rows)

    def try_kuhn(c: int, seen: List[bool]) -> bool:
        for ri in radj[c]:
            if not seen[ri]:
                seen[ri] = True
                if match_row[ri] < 0 or try_kuhn(match_row[ri], seen):
                    match_row[ri] = c
                    return True
        return False

    size = 0
    for c in range(len(col_keys)):
        if try_kuhn(c, [False] * len(rows)):
            size += 1
    return size


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _trim(w) -> Perm:
    w = list(map(int, w))
    while len(w) > 1 and w[-1] == len(w):
        w.pop()
    return tuple(w)


def product_nonzero(x1, x2, x3, n: int) -> bool:
    """Ground truth: sigma_{x1} sigma_{x2} sigma_{x3} != 0 in H*(Fl_n),
    by chained Schubert expansion (positivity => no cancellation; early
    exit).  Exponential; small n only (validation)."""
    import lrcalc
    P = lrcalc.schubmult(list(pad(x1, n)), list(pad(x2, n)))
    for y, c in P.items():
        y = _trim(y)
        if len(y) > n or not c:
            continue
        Q = lrcalc.schubmult(list(pad(y, n)), list(pad(x3, n)))
        for z, c2 in Q.items():
            if c2 and len(_trim(z)) <= n:
                return True
    return False


def _lrcalc_coeff(u, v, w) -> int:
    import lrcalc
    n = max(len(u), len(v), len(w))
    prod = lrcalc.schubmult(list(pad(u, n)), list(pad(v, n)))
    tgt = pad(w, n)
    for key, c in prod.items():
        kk = tuple(int(t) for t in key)
        m = max(len(kk), n)
        if pad(kk, m) == pad(tgt, m):
            return int(c)
    return 0


def _selftest(verbose: bool = True) -> None:
    import random as _random
    rng = _random.Random(0)

    # 1) cocycle containment for LEFT minreps: Inv(q^E) subseteq Inv(q^F)
    #    subseteq Inv(x) for E subseteq F; towers compose.  For RIGHT
    #    minreps containment genuinely FAILS (twisted system): count it.
    right_containment_fails = 0
    for _ in range(300):
        n = rng.choice([4, 5, 6, 7])
        x = tuple(rng.sample(range(1, n + 1), n))
        full = set(inv_set(x))
        allcuts = list(range(1, n))
        for _ in range(4):
            E = sorted(rng.sample(allcuts, rng.randint(1, n - 1)))
            qx = parabolic_min_rep_left(x, E, n)
            assert set(inv_set(qx)) <= full, (x, E, qx)
            F = sorted(set(E) | {rng.choice(allcuts)})
            qxF = parabolic_min_rep_left(x, F, n)
            assert set(inv_set(qx)) <= set(inv_set(qxF)), (x, E, F)
            assert parabolic_min_rep_left(qxF, E, n) == qx, (x, E, F)
            px = parabolic_min_rep(x, E, n)
            if not set(inv_set(px)) <= full:
                right_containment_fails += 1
    assert right_containment_fails > 0   # the twist is real
    if verbose:
        print("selftest 1 OK: left-minrep cocycle containment + towers "
              f"(right-minrep containment fails {right_containment_fails}x, "
              "as predicted)")

    # 2) exhaustive S3 and S4 against lrcalc (0 errors tolerated)
    for n in (3, 4):
        perms = list(itertools.permutations(range(1, n + 1)))
        d = n * (n - 1) // 2
        n_pos = n_zero = 0
        for u in perms:
            for v in perms:
                lu, lv = length(u), length(v)
                for w in perms:
                    if length(w) != lu + lv:
                        continue
                    truth = _lrcalc_coeff(u, v, w) > 0
                    got = cert_nonvanishing(u, v, w, seeds=(0, 1))
                    assert got == truth, (u, v, w, truth, got)
                    n_pos += truth
                    n_zero += (not truth)
        if verbose:
            print(f"selftest 2 OK: S{n} exhaustive "
                  f"({n_pos} positives, {n_zero} zeros, 0 errors)")

    # 3) parabolic faces against multicut.multistep_feasible on random S5/S6:
    #    side="right" against Feas_T(E); side="left" against Feas_{T^-1}(E).
    from .multicut import multistep_feasible
    checked = 0
    for _ in range(4000):
        n = rng.choice([5, 6])
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        w = tuple(rng.sample(range(1, n + 1), n))
        if length(w) != length(u) + length(v):
            continue
        E = sorted(rng.sample(range(1, n), rng.randint(1, n - 1)))
        truth_r = multistep_feasible(u, v, w, E, n)
        got_r = cert_face_feasible(u, v, w, E, seeds=(0, 1), side="right")
        assert got_r == truth_r, (u, v, w, E, truth_r, got_r, "right")
        checked += 1
    if verbose:
        print(f"selftest 3 OK: {checked} random right faces agree with "
              "multistep_feasible (the project's Feas oracle)")

    # 3b) the criterion on ARBITRARY triples (any degrees) against direct
    #     product expansion in H*(Fl_n) -- this covers the left faces and
    #     the square uniformly.
    checked = 0
    for _ in range(220):
        n = rng.choice([4, 5])
        x1 = tuple(rng.sample(range(1, n + 1), n))
        x2 = tuple(rng.sample(range(1, n + 1), n))
        x3 = tuple(rng.sample(range(1, n + 1), n))
        truth = product_nonzero(x1, x2, x3, n)
        got = False
        for s in (0, 1):
            C = Certificate(x1, x2, w0_compose(x3, n), seed=s, n=n)
            # Certificate factors are (x1, x2, w0.(w0.x3)) = (x1, x2, x3)
            if rank_mod(C.M, C.p) == len(C.col_keys):
                got = True
                break
        assert got == truth, (x1, x2, x3, truth, got)
        checked += 1
    # plus left faces of random degree-additive triples vs direct product
    checked_l = 0
    for _ in range(400):
        n = rng.choice([4, 5])
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        w = tuple(rng.sample(range(1, n + 1), n))
        if length(w) != length(u) + length(v):
            continue
        E = sorted(rng.sample(range(1, n), rng.randint(1, n - 1)))
        w3 = w0_compose(w, n)
        qs = [parabolic_min_rep_left(x, E, n) for x in (u, v, w3)]
        truth = product_nonzero(qs[0], qs[1], qs[2], n)
        got = cert_face_feasible(u, v, w, E, seeds=(0, 1), side="left")
        assert got == truth, (u, v, w, E, truth, got, "left")
        checked_l += 1
    if verbose:
        print(f"selftest 3b OK: criterion == product-nonvanishing on "
              f"{checked} arbitrary triples and {checked_l} left faces")

    # 4) functoriality: left faces are literal column submatrices (keys
    #    nested along towers; full cut set reproduces all columns)
    C = Certificate((2, 1, 4, 3), (3, 1, 2, 4), (4, 1, 3, 2), seed=0)
    kfull = set(C.face_col_keys(range(1, C.n), side="left"))
    assert kfull == set(C.col_keys)
    assert set(C.face_col_keys(range(1, C.n), side="right")) == set(C.col_keys)
    kE = C.face_col_keys([1], side="left")
    kF = C.face_col_keys([1, 3], side="left")
    assert set(kE) <= set(kF) <= kfull
    if verbose:
        print("selftest 4 OK: left face columns nested inside the full "
              "certificate")

    # 5) structural rank sanity: identity triple structure
    ck = C.col_keys
    sr = structural_rank(C.n, ck)
    gr = C.square_rank()
    assert gr <= sr <= C.d
    if verbose:
        print(f"selftest 5 OK: structural rank {sr} >= generic rank {gr}")
    print("algebraic_certificate selftest: ALL OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
