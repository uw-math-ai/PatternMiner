"""Functorial left-certificate face diagram and its poset cohomology.

K1 machinery for EXPERT_RESPONSE_4 SS7.1 and EXPERT_RESPONSE_5 SS3.3-3.5,
SS11 item 11.

Construction
------------
Fix a degree-additive triple T = (u, v, w) in S_n with factors
(x_1, x_2, x_3) = (u, v, w0.w) and the Purbhoo certificate matrix
M = M_T over F_p (schubml.combinatorics.algebraic_certificate.Certificate):
d = n(n-1)/2 rows (s,t), d columns (i,(a,b)), (a,b) in Inv(x_i).

The LEFT (nested / reference-flag) lattice: let

    UL = Des(u^{-1}) u Des(v^{-1}) u Des((w0.w)^{-1}),   m = |UL|.

For E subseteq UL the left face system has columns
cols(E) = {(i, ab) : ab in Inv(minrep(W_{P_E} x_i))}, and the cocycle
identity gives PLAIN NESTING  cols(E) subseteq cols(F) for E subseteq F,
with cols(UL) = all d columns (algebraic_certificate docstring; selftest 1
there).  So every face matrix M_E is a literal column submatrix of M.

Functorial coefficient systems on the face poset
------------------------------------------------
For E subseteq F both natural maps go UP the poset (covariant; there is no
canonical restriction in the other direction):

  * kernels:    ker(M_E) -> ker(M_F), extension by zero on the new columns
                (a relation among cols(E) is one among cols(F));
  * cokernels:  coker(M_E) = F_p^d / im(M_E)  ->>  coker(M_F),
                the further quotient (im(M_E) subseteq im(M_F)).

Both assignments are cellular sheaves on the simplicial complex

    bdry Delta^{m-1}  =  { nonempty proper E subsetneq UL }  ~=  S^{m-2},

whose cells are the faces themselves (dimension |E| - 1).  Dually, the
annihilator system  E |-> ann(im M_E) = {y : y^T M_E = 0}  has arrows
DOWN the poset (inclusion), i.e. it is an inverse system; its inverse
limit over the proper faces is

    lim ann  =  ann( sum_{E coatom} im M_E ),

the annihilator of the span of all proper-face columns.  This is the
derived-inverse-limit object of expert-4 SS7.1 in its linearized form.

Cohomology
----------
The (augmented) cellular cochain complex of a system Fc:

    0 -> Fc(emptyset) -> (+)_{|E|=1} Fc(E) -> ... -> (+)_{|E|=m-1} Fc(E) -> 0

with simplicial signs; for E = F - {f}, sign(E,F) = (-1)^{index of f in F}.
For the coker sheaf we take Fc(emptyset) = F_p^d (the initial object of the
diagram), so the complex computes REDUCED cohomology of the coefficient
system on S^{m-2}; the constant system F_p gives Htilde^q = 0 except
Htilde^{m-2} = F_p (selftest).

Basis conventions.  coker(M_E) is represented through a full-row-rank
matrix Y_E (a left-kernel basis of M_E, shape (d - r_E) x d): the class of
x in F_p^d is the vector Y_E x.  The induced projection
coker(E) -> coker(F) is then the unique T_{E,F} with T_{E,F} Y_E = Y_F
(exists because ann(im M_F) subseteq ann(im M_E); solved exactly mod p).
ker(M_E) is represented by a right-kernel basis K_E in the local column
order of the face; the induced injection is zero-extension re-expressed in
K_F (solved exactly mod p).  Composition-compatibility (the diagram
commutes) is verified numerically by `verify_functoriality`.

Main theorem being tested (the LES identity).  From the cellwise-exact
sequence of sheaves 0 -> im -> const F^d -> coker -> 0 and the top-degree
part of the long exact sequence,

    Htilde^{m-2}(bdry Delta; coker)  ~=  F_p^d / sum_{E coatom} im(M_E),

with the canonical surjection onto O_T = coker(M_UL) = F_p^d / im(M)
induced by the inclusion of the coatom span into the full column span.  So

    Htilde^{m-2} ~= O_T   <=>   the proper faces already span the full
                                 generic column space,

and the identification is realized by the left-kernel covector y of the
full matrix (y^T M = 0): y annihilates every face's columns, hence defines
a compatible functional on every coker(E) -- a global section of the dual
system -- and pairing a top cocycle (xi_E)_{coatoms} with y through the
orientation signs eps_E computes the image in O_T ~= F_p.

Decision semantics: everything is computed at exact specializations over
F_p (default p = 2^31 - 1, alternate 2147483587), >= 2 seeds; a nonzero
minor at any specialization is a deterministic certificate for a generic
rank lower bound; rank-deficiency verdicts are Schwartz-Zippel-whp and are
reported with the per-spec stability flags.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .algebraic_certificate import (
    Certificate, ColKey, P_ALT, P_DEFAULT, descents, inverse, length, pad,
    parabolic_min_rep_left, rank_mod, right_kernel_mod, rref_mod,
)

Face = Tuple[int, ...]          # sorted tuple of cuts, subset of UL


# ---------------------------------------------------------------------------
# exact modular linear algebra helpers
# ---------------------------------------------------------------------------
def matmul_mod(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """(A @ B) mod p without int64 overflow (p < 2^31): accumulate the
    inner dimension in chunks of 2, reducing after each chunk."""
    A = np.asarray(A, dtype=np.int64) % p
    B = np.asarray(B, dtype=np.int64) % p
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for k0 in range(0, A.shape[1], 2):
        out = (out + A[:, k0:k0 + 2] @ B[k0:k0 + 2, :]) % p
    return out


def solve_mod(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """Solve A X = B (mod p) for A (dxr) of full column rank r, assuming
    every column of B lies in colspace(A).  Returns X (r x s).  Raises
    ValueError if the system is inconsistent or A is column-deficient."""
    A = np.asarray(A, dtype=np.int64) % p
    B = np.asarray(B, dtype=np.int64) % p
    d, r = A.shape
    s = B.shape[1]
    R, piv = rref_mod(np.concatenate([A, B], axis=1), p)
    apiv = [c for c in piv if c < r]
    if len(apiv) != r:
        raise ValueError("solve_mod: A not full column rank")
    if any(c >= r for c in piv):
        raise ValueError("solve_mod: inconsistent system (B not in im A)")
    X = np.zeros((r, s), dtype=np.int64)
    for row_i, c in enumerate(apiv):
        X[c, :] = R[row_i, r:]
    return X


def left_kernel_basis(M: np.ndarray, p: int) -> np.ndarray:
    """Rows form a basis of {y : y^T M = 0} (shape (nrows - rank, nrows))."""
    return right_kernel_mod(np.asarray(M, dtype=np.int64).T, p)


# ---------------------------------------------------------------------------
# the face diagram
# ---------------------------------------------------------------------------
class FaceDiagram:
    """Left/nested face diagram of one triple at one exact specialization
    (seed, p).  Faces are subsets of UL; the proper poset is bdry
    Delta^{m-1} ~= S^{m-2}."""

    def __init__(self, u: Sequence[int], v: Sequence[int], w: Sequence[int],
                 seed: int = 0, p: int = P_DEFAULT, n: Optional[int] = None):
        self.cert = Certificate(u, v, w, seed=seed, p=p, n=n)
        C = self.cert
        self.p, self.seed = C.p, seed
        self.n, self.d = C.n, C.d
        self.UL: Tuple[int, ...] = tuple(sorted(
            set(descents(inverse(C.u))) | set(descents(inverse(C.v)))
            | set(descents(inverse(C.w3)))))
        self.m = len(self.UL)
        full = C.face_col_keys(self.UL, side="left")
        assert set(full) == set(C.col_keys) and len(full) == self.d, \
            "UL does not reproduce the full certificate"
        # caches
        self._keys: Dict[Face, List[ColKey]] = {}
        self._idx: Dict[Face, List[int]] = {}
        self._rank: Dict[Face, int] = {}
        self._Y: Dict[Face, np.ndarray] = {}
        self._K: Dict[Face, np.ndarray] = {}
        self._rank_full: Optional[int] = None

    # -- combinatorics ------------------------------------------------------
    def norm(self, E: Iterable[int]) -> Face:
        E = tuple(sorted(set(int(e) for e in E)))
        assert set(E) <= set(self.UL), f"face {E} not inside UL {self.UL}"
        return E

    def proper_faces(self) -> List[Face]:
        """All nonempty proper subsets of UL (cells of bdry Delta^{m-1})."""
        out: List[Face] = []
        for q in range(1, self.m):
            out.extend(itertools.combinations(self.UL, q))
        return out

    def cells(self, q: int) -> List[Face]:
        """Cells of dimension q (|E| = q+1); q = -1 is the empty face."""
        if q == -1:
            return [()]
        return list(itertools.combinations(self.UL, q + 1))

    def face_keys(self, E: Iterable[int]) -> List[ColKey]:
        E = self.norm(E)
        if E not in self._keys:
            keys = self.cert.face_col_keys(E, side="left") if E else []
            # deterministic order: global column order of the certificate
            keys = sorted(keys, key=lambda k: self.cert.col_pos[k])
            self._keys[E] = keys
            self._idx[E] = [self.cert.col_pos[k] for k in keys]
        return self._keys[E]

    def face_matrix(self, E: Iterable[int]) -> np.ndarray:
        self.face_keys(E)
        idx = self._idx[self.norm(E)]
        return self.cert.M[:, idx]

    # -- per-face linear algebra -------------------------------------------
    def face_rank(self, E: Iterable[int]) -> int:
        E = self.norm(E)
        if E not in self._rank:
            self._rank[E] = rank_mod(self.face_matrix(E), self.p) \
                if self.face_keys(E) else 0
        return self._rank[E]

    def face_nullity(self, E: Iterable[int]) -> int:
        E = self.norm(E)
        return len(self.face_keys(E)) - self.face_rank(E)

    def right_kernel(self, E: Iterable[int]) -> np.ndarray:
        """Right-kernel basis of M_E, rows = kernel vectors, columns in the
        face's local (global-order) column order."""
        E = self.norm(E)
        if E not in self._K:
            Msub = self.face_matrix(E)
            if Msub.shape[1] == 0:
                self._K[E] = np.zeros((0, 0), dtype=np.int64)
            else:
                self._K[E] = right_kernel_mod(Msub, self.p)
        return self._K[E]

    def coker_Y(self, E: Iterable[int]) -> np.ndarray:
        """Y_E: full-row-rank ((d - r_E) x d) with Y_E M_E = 0; the class of
        x in coker(E) is Y_E x.  E = () gives Y = I_d (coker = F^d)."""
        E = self.norm(E)
        if E not in self._Y:
            if not E:
                self._Y[E] = np.eye(self.d, dtype=np.int64)
            else:
                self._Y[E] = left_kernel_basis(self.face_matrix(E), self.p)
        return self._Y[E]

    # -- the functorial maps ------------------------------------------------
    def coker_map(self, E: Iterable[int], F: Iterable[int]) -> np.ndarray:
        """Matrix of the projection coker(E) ->> coker(F) for E subseteq F,
        in the Y-bases: the unique T with T Y_E = Y_F."""
        E, F = self.norm(E), self.norm(F)
        assert set(E) <= set(F), "coker_map needs E subseteq F"
        YE, YF = self.coker_Y(E), self.coker_Y(F)
        # T YE = YF  <=>  YE^T T^T = YF^T
        return solve_mod(YE.T, YF.T, self.p).T

    def ker_map(self, E: Iterable[int], F: Iterable[int]) -> np.ndarray:
        """Matrix of the injection ker(E) -> ker(F) (zero-extension) in the
        K-bases: S with (rows of K_E, extended) = S @ K_F."""
        E, F = self.norm(E), self.norm(F)
        assert set(E) <= set(F), "ker_map needs E subseteq F"
        KE, KF = self.right_kernel(E), self.right_kernel(F)
        keysE, keysF = self.face_keys(E), self.face_keys(F)
        posF = {k: j for j, k in enumerate(keysF)}
        emb = np.zeros((KE.shape[0], len(keysF)), dtype=np.int64)
        for jE, k in enumerate(keysE):
            emb[:, posF[k]] = KE[:, jE]
        if KE.shape[0] == 0:
            return np.zeros((0, KF.shape[0]), dtype=np.int64)
        # emb^T = KF^T @ S^T
        return solve_mod(KF.T, emb.T, self.p).T

    # -- global objects -----------------------------------------------------
    def full_rank(self) -> int:
        if self._rank_full is None:
            self._rank_full = rank_mod(self.cert.M, self.p)
        return self._rank_full

    def corank(self) -> int:
        return self.d - self.full_rank()

    def left_kernel_full(self) -> np.ndarray:
        """Basis of ann(im M) = O_T^* (rows), shape (corank, d)."""
        return left_kernel_basis(self.cert.M, self.p)

    def coatom_union(self) -> Tuple[List[int], int, List[ColKey]]:
        """(sorted union of coatom column indices, its rank, top-only keys).
        rank(union) = d - dim Htilde^{m-2}(coker) by the LES identity; the
        annihilator of the union span is lim^0 of the dual system."""
        union: Set[int] = set()
        for e in self.UL:
            E = tuple(c for c in self.UL if c != e)
            self.face_keys(E)
            union |= set(self._idx[E])
        idx = sorted(union)
        rk = rank_mod(self.cert.M[:, idx], self.p) if idx else 0
        top_only = [self.cert.col_keys[j] for j in range(self.d)
                    if j not in union]
        return idx, rk, top_only

    def lim0_annihilator(self) -> np.ndarray:
        """Basis (rows) of lim^0 of the annihilator inverse system
        = ann(span of all proper-face columns)."""
        idx, _, _ = self.coatom_union()
        return left_kernel_basis(self.cert.M[:, idx], self.p)

    # -- cochain complex ----------------------------------------------------
    def _stalk_dim(self, system: str, E: Face) -> int:
        if system == "const":
            return 1
        if system == "coker":
            return self.d - self.face_rank(E) if E else self.d
        if system == "ker":
            return self.face_nullity(E) if E else 0
        raise ValueError(system)

    def _map_block(self, system: str, E: Face, F: Face) -> np.ndarray:
        if system == "const":
            return np.ones((1, 1), dtype=np.int64)
        if system == "coker":
            return self.coker_map(E, F)
        if system == "ker":
            # ker_map is in row-vector convention (extended K_E = S K_F);
            # the block acting on column coordinate vectors is S^T
            return self.ker_map(E, F).T
        raise ValueError(system)

    def delta(self, system: str, q: int) -> np.ndarray:
        """Coboundary C^q -> C^{q+1} of the augmented complex
        (q = -1 .. m-3).  Rows indexed by (q+1)-cells, cols by q-cells,
        blocks sign(E,F) * map(E -> F)."""
        src, tgt = self.cells(q), self.cells(q + 1)
        sdims = [self._stalk_dim(system, E) for E in src]
        tdims = [self._stalk_dim(system, F) for F in tgt]
        soff = np.concatenate([[0], np.cumsum(sdims)])
        toff = np.concatenate([[0], np.cumsum(tdims)])
        D = np.zeros((int(toff[-1]), int(soff[-1])), dtype=np.int64)
        spos = {E: i for i, E in enumerate(src)}
        for it, F in enumerate(tgt):
            if tdims[it] == 0:
                continue
            for pos_f in range(len(F)):
                E = F[:pos_f] + F[pos_f + 1:]
                isrc = spos[E]
                if sdims[isrc] == 0:
                    continue
                sign = 1 if pos_f % 2 == 0 else self.p - 1
                blk = (sign * self._map_block(system, E, F)) % self.p
                D[toff[it]:toff[it + 1], soff[isrc]:soff[isrc + 1]] = blk
        return D

    def cohomology_dims(self, system: str,
                        degrees: Optional[Sequence[int]] = None
                        ) -> Dict[int, int]:
        """Reduced cohomology dims Htilde^q of the system on S^{m-2}.
        degrees: which q to report (default all -1..m-2).  Computes only
        the needed coboundaries."""
        if degrees is None:
            degrees = list(range(-1, self.m - 1))
        degrees = sorted(set(degrees))
        need_delta = set()
        for q in degrees:
            need_delta.add(q - 1)
            need_delta.add(q)
        ranks: Dict[int, int] = {}
        for q in sorted(need_delta):
            if q < -1 or q > self.m - 3:
                ranks[q] = 0
            else:
                D = self.delta(system, q)
                ranks[q] = rank_mod(D, self.p) if D.size else 0
        out = {}
        for q in degrees:
            cdim = sum(self._stalk_dim(system, E) for E in self.cells(q)) \
                if -1 <= q <= self.m - 2 else 0
            out[q] = cdim - ranks[q] - ranks[q - 1]
        return out

    # -- top-degree detail: generator + natural pairing with O_T ------------
    def orientation_signs(self) -> Dict[Face, int]:
        """eps_E = (-1)^i for the coatom UL - {e_i} (fundamental cocycle of
        bdry Delta^{m-1})."""
        out = {}
        for i, e in enumerate(self.UL):
            E = tuple(c for c in self.UL if c != e)
            out[E] = 1 if i % 2 == 0 else -1
        return out

    def top_cohomology(self) -> dict:
        """Htilde^{m-2}(coker): dimension, a basis of representatives, the
        natural functional(s) induced by the left kernel of the full matrix
        (one per corank dimension), and the verdict whether the pairing
        identifies Htilde^{m-2} with O_T."""
        p = self.p
        coat = self.cells(self.m - 2)
        cdims = [self._stalk_dim("coker", E) for E in coat]
        off = np.concatenate([[0], np.cumsum(cdims)])
        Ctop = int(off[-1])
        D = self.delta("coker", self.m - 3)     # C^{m-3} -> C^{m-2}
        rkD = rank_mod(D, p) if D.size else 0
        dimH = Ctop - rkD
        # basis of a complement of im D in C^top: non-pivot rows of rref(D^T)
        # -> use left kernel of D (rows spanning ann(im D)) as the DUAL of H
        HdualB = left_kernel_basis(D, p) if D.size else \
            np.eye(Ctop, dtype=np.int64)
        assert HdualB.shape[0] == dimH
        # natural functional(s): y in ann(im M) -> Lambda_y on C^top
        Y = self.left_kernel_full()             # (corank, d)
        eps = self.orientation_signs()
        lam = np.zeros((Y.shape[0], Ctop), dtype=np.int64)
        for i, E in enumerate(coat):
            YE = self.coker_Y(E)
            # coefficients c with c^T Y_E = y^T for each row y of Y
            Cc = solve_mod(YE.T, Y.T, p)        # (d - r_E, corank)
            s = eps[E] % p
            lam[:, off[i]:off[i + 1]] = (s * Cc.T) % p
        # Lambda must kill coboundaries (naturality check)
        chk = matmul_mod(lam, D, p) if D.size else np.zeros((Y.shape[0], 0),
                                                            dtype=np.int64)
        lambda_kills_coboundaries = bool((chk % p == 0).all())
        # rank of the pairing Lambda restricted to H (via the dual basis):
        # pair lam with a complement basis: solve H-representatives
        # complement basis: columns of C^top dual to HdualB -- use rref
        # pivots of D to find B = im D; representatives = coords not pivoted.
        if D.size:
            # pivot coordinates of a row-echelon basis of colspace(D)
            _, pivt = rref_mod(D.T, p)
            im_pivot_rows = pivt
        else:
            im_pivot_rows = []
        rep_rows = [i for i in range(Ctop) if i not in im_pivot_rows][:dimH]
        reps = np.zeros((dimH, Ctop), dtype=np.int64)
        for k, i in enumerate(rep_rows):
            reps[k, i] = 1
        pairing = matmul_mod(lam, reps.T, p)     # (corank, dimH)
        pairing_rank = rank_mod(pairing, p)
        return dict(
            dim=int(dimH), corank=int(self.corank()),
            lambda_kills_coboundaries=lambda_kills_coboundaries,
            pairing_rank=int(pairing_rank),
            natural_iso=bool(dimH == self.corank()
                             and pairing_rank == dimH
                             and lambda_kills_coboundaries),
            coatom_stalk_dims={str(E): int(dd)
                               for E, dd in zip(coat, cdims)})

    # -- functoriality verification ----------------------------------------
    def verify_functoriality(self, n_towers: int = 40,
                             rng: Optional[np.random.Generator] = None
                             ) -> dict:
        """Random towers E subseteq F subseteq G subseteq UL: verify
        (a) key nesting, (b) minrep tower identity per factor,
        (c) coker maps compose exactly, (d) ker maps compose exactly,
        (e) coker projections are surjective / ker maps injective."""
        if rng is None:
            rng = np.random.default_rng(12345)
        p = self.p
        ok = dict(nesting=0, tower=0, coker_comp=0, ker_comp=0,
                  surj=0, inj=0)
        for _ in range(n_towers):
            # random chain by three nested subsets (allow equalities)
            picks = [set(), set(), set()]
            for c in self.UL:
                r = rng.integers(0, 4)
                if r >= 1:
                    picks[2].add(c)
                if r >= 2:
                    picks[1].add(c)
                if r >= 3:
                    picks[0].add(c)
            E, F, G = (self.norm(picks[0]), self.norm(picks[1]),
                       self.norm(picks[2]))
            kE, kF, kG = (set(self.face_keys(E)), set(self.face_keys(F)),
                          set(self.face_keys(G)))
            assert kE <= kF <= kG
            ok["nesting"] += 1
            for x in self.cert.factors:
                qF = parabolic_min_rep_left(x, F, self.n)
                qEF = parabolic_min_rep_left(qF, E, self.n)
                assert qEF == parabolic_min_rep_left(x, E, self.n)
            ok["tower"] += 1
            TEF = self.coker_map(E, F)
            TFG = self.coker_map(F, G)
            TEG = self.coker_map(E, G)
            assert (matmul_mod(TFG, TEF, p) == TEG % p).all()
            ok["coker_comp"] += 1
            SEF = self.ker_map(E, F)
            SFG = self.ker_map(F, G)
            SEG = self.ker_map(E, G)
            if SEF.shape[0]:
                assert (matmul_mod(SEF, SFG, p) == SEG % p).all()
            ok["ker_comp"] += 1
            # coker maps surjective (T has full row rank), ker maps injective
            if TEF.size:
                assert rank_mod(TEF, p) == TEF.shape[0]
            ok["surj"] += 1
            if SEF.shape[0]:
                assert rank_mod(SEF, p) == SEF.shape[0]
            ok["inj"] += 1
        ok["n_towers"] = n_towers
        return ok


# ---------------------------------------------------------------------------
# selftest on exhaustively checkable small cases
# ---------------------------------------------------------------------------
def _lrcalc_coeff(u, v, w) -> int:
    import lrcalc
    n = max(len(u), len(v), len(w))
    prod = lrcalc.schubmult(list(pad(u, n)), list(pad(v, n)))
    tgt = pad(w, n)
    for key, c in prod.items():
        kk = tuple(int(t) for t in key)
        mx = max(len(kk), n)
        if pad(kk, mx) == pad(tgt, mx):
            return int(c)
    return 0


def _selftest(verbose: bool = True) -> None:
    import random as _random
    rng = _random.Random(7)
    nprng = np.random.default_rng(7)

    # collect small degree-additive triples with known coefficient
    cases = []
    for _ in range(4000):
        n = rng.choice([4, 5])
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        w = tuple(rng.sample(range(1, n + 1), n))
        if length(w) != length(u) + length(v):
            continue
        cases.append((u, v, w, _lrcalc_coeff(u, v, w)))
        if len(cases) >= 60:
            break

    n_zero = n_pos = 0
    n_iso = n_noniso = 0
    checked_les = 0
    for (u, v, w, coeff) in cases:
        for spec in ((0, P_DEFAULT), (1, P_ALT)):
            FD = FaceDiagram(u, v, w, seed=spec[0], p=spec[1])
            if FD.m < 2:
                continue
            # 1) constant system: reduced sphere cohomology
            dims = FD.cohomology_dims("const")
            expect = {q: (1 if q == FD.m - 2 else 0)
                      for q in range(-1, FD.m - 1)}
            assert dims == expect, (u, v, w, dims)
            # 2) functoriality on random towers
            FD.verify_functoriality(n_towers=6, rng=nprng)
            # 3) LES identity: dim Htilde^{m-2}(coker) = d - rank(union)
            top = FD.cohomology_dims("coker", degrees=[FD.m - 2])[FD.m - 2]
            _, rk_union, _ = FD.coatom_union()
            assert top == FD.d - rk_union, (u, v, w, top, FD.d, rk_union)
            checked_les += 1
            # 4) corank vs coefficient (certificate correctness recheck)
            if coeff > 0:
                assert FD.corank() == 0
            else:
                assert FD.corank() >= 1
            # 5) top detail: pairing and natural surjection onto O_T
            det = FD.top_cohomology()
            assert det["dim"] == top
            assert det["lambda_kills_coboundaries"]
            assert det["pairing_rank"] == min(det["dim"], FD.corank())
            # dim Htilde >= corank always (surjection onto O_T)
            assert det["dim"] >= FD.corank()
            if det["natural_iso"]:
                n_iso += 1
            else:
                n_noniso += 1
            # 6) full-degree run on the coker and ker sheaves (small m)
            if FD.m <= 5:
                FD.cohomology_dims("coker")
                FD.cohomology_dims("ker")
        n_zero += (coeff == 0)
        n_pos += (coeff > 0)
    if verbose:
        print(f"selftest OK: {len(cases)} S4/S5 triples "
              f"({n_zero} zeros, {n_pos} positive), 2 specs each; "
              f"constant-sheaf reduced cohomology == S^(m-2); "
              f"functoriality towers exact; LES identity verified "
              f"{checked_les}x; natural-iso verdicts {n_iso} iso / "
              f"{n_noniso} non-iso")

    # 7) a corank-1 zero end-to-end: Htilde^{m-2} ~= O_T naturally
    found = None
    for (u, v, w, coeff) in cases:
        if coeff:
            continue
        FD = FaceDiagram(u, v, w, seed=0, p=P_DEFAULT)
        if FD.corank() == 1 and FD.m >= 3:
            found = (u, v, w, FD)
            break
    if found is not None:
        u, v, w, FD = found
        det = FD.top_cohomology()
        lim0 = FD.lim0_annihilator()
        assert lim0.shape[0] == det["dim"]
        # y is a global section of the annihilator system: check on every
        # proper face
        y = FD.left_kernel_full()
        for E in FD.proper_faces():
            ME = FD.face_matrix(E)
            if ME.size:
                assert (matmul_mod(y, ME, FD.p) == 0).all()
        if verbose:
            print(f"selftest 7 OK: corank-1 zero {u},{v},{w}: "
                  f"dim Htilde^(m-2) = {det['dim']}, natural_iso = "
                  f"{det['natural_iso']}, lim0 dim = {lim0.shape[0]}, "
                  f"y is a global annihilator section on all "
                  f"{len(FD.proper_faces())} proper faces")
    print("face_diagram selftest: ALL OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
