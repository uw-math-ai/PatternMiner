# P16: statements, proofs, and open conjecture

These are scoped exports of the research arguments supporting [the results summary](results.md). The proofs underwent internal adversarial review in the originating research campaign. The publication replay checks finite instances and arithmetic; it is not a formal proof check, external peer review, or a novelty certification.

## Conventions and dependencies

Permutations in code are tuples containing `0,...,n-1`. Right multiplication by a simple reflection swaps adjacent positions. Length is inversion count. Off-order KL polynomials are zero. For odd positive gap `d`, `mu(x,w)` denotes the coefficient at degree `(d-1)/2`; it is zero for other gaps. A common position/value is cancellable (called *naked* in the retained notes) when its strict-prefix rank defect is zero; a *canonical* pair has no such position. The support graph uses the position transpositions of a saturated Bruhat chain.

| Research label | Included argument or established input |
| --- | --- |
| T016–T017 | [Coefficient difference at length thirteen](#coefficient-difference-at-length-thirteen), restated below with the current source attribution |
| T020 | Warrington's published leading-coefficient spectrum through S₁₁; its maximum is 28 |
| T029 | [Leading recurrence](#leading-recurrence), derived below |
| T031 | Common-entry cancellation: Warrington Fact 8; the support proof supplies preservation of the full labelled interval |
| T039 | [Support factorization](#support-factorization) |
| T041 | [Reduction to twelve positions](#reduction-to-twelve-positions) |
| T042 | [Common descents and cancellation](#common-descents-and-cancellation) |
| T044 | [Genus heredity](#genus-heredity) |
| T045 | [Corrected descent gain](#corrected-descent-gain) |

The retained notes mention other historical experiment/conjecture IDs to delimit their scope. Their results are not additional premises of the proofs here. In particular, this package does not rely on the omitted basin search, deletion experiment, or proposed bounds Q009–Q011.

## Coefficient difference at length thirteen

For two isomorphic type-A intervals `I,I'` of length thirteen, there is an integer `Delta` such that

```math
\widetilde R_I(z)-\widetilde R_{I'}(z)=\Delta z^9,
\qquad
P_I(q)-P_{I'}(q)=\Delta(q^2-9q^3+36q^4-84q^5+126q^6).
```

**Proof.** Esposito–Marietti–Stella, Corollary 1.3, gives invariance of `R-tilde` modulo `z^8` and full type-A invariance through length twelve. Monicity and parity leave degrees nine and eleven as possible discrepancies at length thirteen. Patimo's invariant coefficient of `q` in `P`, together with KL inversion, gives

```math
[z^{d-2}]\widetilde R_{x,w}=d+[q]P_{x,w}
-\#\{y\in[x,w]:\ell(y)=\ell(w)-1\}.
```

Thus degree eleven also agrees. Every proper subinterval has length at most twelve. Subtract the two KL inversion identities, using invariance on those proper subintervals, and put `D=P_I-P_I'`:

```math
q^{13}D(q^{-1})-D(q)=\Delta q^2(q-1)^9.
```

The degree bound `deg D <= 6` determines its coefficients from the degrees at most six on the right, giving the displayed formula. Positivity then forces the positively oriented member to have `mu >= 126 Delta` when `Delta > 0`. This is a conditional restriction; it supplies no counterexample and does not prove `Delta=0`.

## Leading recurrence

For a common right descent `s` and odd gap `d=2r+1 >= 3`, the usual recurrence is

```math
P_{x,w}=P_{xs,ws}+qP_{x,ws}
-\sum_{\substack{x\le z\lt ws\\zs\lt z}}\mu(z,ws)
 q^{(\ell(w)-\ell(z))/2}P_{x,z}.
```

Only terms with nonzero `mu(z,ws)` occur. Extracting degree `r` and using the KL degree bounds gives

```math
\mu(x,w)-\mu(xs,ws)=[q^{r-1}]P_{x,ws}
-\sum_{\substack{x\lt z\le ws,\ zs\lt z\\\ell(z)-\ell(x)\text{ odd}}}
\mu(x,z)\mu(z,ws).
```

The first term is `A_s`, the correction sum is `C_s`, and their difference is `B_s` in the descent-gain proof. The correction parity follows from the nonzero leading multiplier, and only the highest permitted coefficient of `P_{x,z}` can contribute. Negative gains are possible, as the executable controls demonstrate; the telescope does not assume gain nonnegativity.

## Open conjecture

For every canonical connected pair in `S_12` with gap thirteen, three cycles in `x^{-1}w` (including fixed points), and positive leading coefficient, every common right descent has `B_s <= 4`. T045 shows that this would prove length-thirteen invariance. **This inequality is unproved.** The coefficient-eight example concerns `A_s` in a broader facet class and does not refute this source-sensitive conjecture.

## Established references

- [Esposito–Marietti–Stella, arXiv:2509.16433v2](https://arxiv.org/html/2509.16433v2), Corollary 1.3: the type-A length-twelve theorem is established background. The first arXiv version stated a weaker consequence; use version 2.
- [Patimo, A combinatorial formula for the coefficient of q in Kazhdan–Lusztig polynomials](https://doi.org/10.1093/imrn/rnz255), Corollary 6.2: invariance of that coefficient.
- [Warrington, Equivalence classes for the mu-coefficient](https://arxiv.org/html/1010.3961), Theorem 1 and Facts 5, 6, 8: the spectrum through S₁₁, extremality, leading-coefficient symmetry, and cancellation.
- [Tsukerman–Williams, Bruhat interval polytopes](https://dmtcs.episciences.org/2507/pdf), Definition 4.5 and Corollary 4.8: chain-independent support partition.
- [Dyer, Hecke algebras and shellings of Bruhat intervals](https://www.numdam.org/item/CM_1993__89_1_91_0.pdf): the increasing-path formula used in factorization.

## Internal review and finite checks

The source audit of T039 also reviews T041's exact S₁₂ reduction. T044 and T045 have separate internal proof reviews. Their historical source hashes are retained in `research/SOURCE_MAP.json`; the current proof excerpts and executable controls are included below and under `research/src/`. The new publication replay recomputes the selected finite controls from the exported code, including a two-engine check of the coefficient-eight example. Those controls do not search all S₁₂ intervals or prove the open inequality.

## Support factorization

Identifier: T039. Date: 2026-09-22. Status: proved; independent audit A-T039-20260922 passed. This is a deduction from established support-partition, reflection-order and cancellation results; publication novelty is not claimed.

Let `u<v` lie in `S_n`, and write `d=ell(v)-ell(u)`. Choose a saturated Bruhat chain from `u` to `v` and draw its position-transposition graph: a step swapping positions `a,b` contributes the edge `{a,b}`. Let `B_1,...,B_k` be its nonsingleton connected components and retain singleton components when describing the full interval. By Tsukerman–Williams, Corollary 4.8, the partition is independent of the chosen saturated chain. For a component `B`, list its positions increasingly and flatten the values of `u` and `v` on those positions to permutations `u_B,v_B`.

**Theorem.** Restriction and flattening give an explicit product isomorphism of reflection-labelled directed Bruhat graphs, and hence of ranked intervals,

    [u,v] = product_B [u_B,v_B].

Moreover `P_{u,v}(q)=product_B P_{u_B,v_B}(q)`, and the same product identity holds for `R` and `tilde R`. This assertion concerns the specified embedded support decomposition; it does not infer KL multiplicativity from an arbitrary abstract poset product.

If `d` is odd and `mu(u,v)>0`, precisely one component is nonsingleton. Therefore the canonical naked-point-free core of `[u,v]` lies in `S_m` with

    m <= d+1.

Consequently every normalized pure-transport path starting at a canonical positive-mu odd-gap-`d` interval has at most

    binom(d+1,2)-d = d(d-1)/2

steps. In particular the horizons for the fixed gaps `7,9,11` are at most `21,36,55`, uniformly over ambient symmetric groups and all choices of pure descent.

### Proof of the embedded product assertion

Every saturated chain uses transpositions inside the same partition blocks, by chain-independence of the support partition. In particular every vertex of the interval preserves the set of values on each block. Project one saturated chain to a block `B`, deleting the steps on other blocks. An ambient cover exchanging values `p<q` at positions `a<b` has no intermediate position with value strictly between `p,q`; the same is true after restriction to `B`, so it projects to a cover. Thus the projected chain is saturated from `u_B` to `v_B`. If its length is `d_B`, then `d_B=ell(v_B)-ell(u_B)` and `sum_B d_B=d`.

For surjectivity, choose arbitrary saturated chains in each flattened component interval and interleave their steps in any order preserving each component's order. Lift these transpositions to their original position blocks and original value sets. Each lifted step swaps an increasing pair of values in increasing positions, so it increases ambient inversion length by a positive odd integer, at least one. There are exactly `sum_B d_B=d` steps between the fixed endpoints `u,v`, whose ambient inversion lengths differ by exactly `d`. Every step must therefore increase length by exactly one. All mixed component states lie in `[u,v]`, and their ranks are the sums of their component ranks.

Conversely, extend any `z in [u,v]` to a saturated chain. Its block projections are prefixes of saturated component chains, showing `z_B in [u_B,v_B]`. Restriction is injective because the positions and value set of each block are fixed. This proves the interval bijection. The same lifting argument for comparable component tuples, completed to full chains, proves the product order. A reflection joining two interval vertices must act inside one block, since a swap across blocks changes their fixed value sets. For a reflection inside one block, its upward orientation is exactly the orientation after flattening, since this depends only on the comparison of the two exchanged values. Hence the full directed reflection-labelled Bruhat graph is a product, not only its cover graph.

### Why the polynomials factor

Fix any reflection order on the ambient type-A reflections. Its restriction to the reflections inside a block, after increasing relabelling of positions, is a reflection order for that component. In the product graph, a globally increasing Bruhat path is uniquely obtained by taking one increasing path in each component and merging their distinct reflection labels in ambient increasing order. Conversely the component projections of every increasing path are increasing. Path lengths add, so Dyer's increasing-path formula gives multiplicativity of `tilde R`. The usual change of variables gives the `R` identity.

This graph factorization also applies to every subinterval. Products of the component KL polynomials satisfy the KL degree bounds and the defining inversion relation with these product `R` polynomials: the convolution sum separates over the Cartesian product. Uniqueness in the KL characterization proves the claimed `P` identity. This proof avoids the unresolved combinatorial-invariance conjecture.

### Why nonzero leading mass forces one active block

All nonsingleton components have positive gap. Indeed, the projected saturated chain contains an edge in each such component. Suppose there are `k>=2` positive-gap components. The KL degree bound gives

    deg P_{u,v} <= sum_B floor((d_B-1)/2) <= floor((d-k)/2) < (d-1)/2

when `d` is odd. Thus the coefficient `mu(u,v)` vanishes, contrary to the hypothesis. There is exactly one active component. This extends T028's product-primality argument to the explicitly proved interlaced support decomposition.

### From support to the actual canonical core

A singleton position is untouched by every cover along a saturated chain and has the same value `j` at both endpoints. In the repository's zero-based convention, its nakedness test is `D(i,j)=#{a<i:v(a)>=j}-#{a<i:u(a)>=j}`. Each upward cover adds one to `D(k,l)` on its rank rectangle. Such a rectangle cannot contain `(i,j)` when position `i` is untouched: otherwise the fixed point with value `j` would lie strictly inside the cover rectangle, contradicting the cover criterion. Therefore `D(i,j)=0`, and this point is naked and cancellable.

Apply the preceding factorization to the canonical core itself. It has no naked points, so it has no singleton support components. Positive odd leading mass forces its entire position graph to be connected. Its saturated chain has `d` edges; a connected graph on `m` vertices needs at least `m-1` edges, proving `m<=d+1`. Cancellation preserves gap and KL polynomials by T031/Warrington Fact 8. We do not claim that arbitrary deletion preserves the original adjacent-descent set or its individual selected atom masses.

For later use, naked cancellation preserves the full reflection-labelled interval and hence `tilde R` as well as `P`. Indeed, put `A=#{a<i:u(a)>=j}=#{a<i:v(a)>=j}` at a naked common point `(i,j)`. At prefixes `i,i+1` and thresholds `j,j+1`, the endpoint rank counts are respectively `A,A,A+1,A`. Every intermediate permutation has the same four counts by Bruhat rank inequalities, which force its value at position `i` to be exactly `j`. After deleting this fixed position and value, all remaining rank comparisons are equivalent to the original comparisons. The inversion contribution involving the deleted point is the constant `2A+j-i`, so all relative lengths are preserved. Reflection edges and their labels are carried to their increasing relabellings. Dyer's path formula consequently preserves `tilde R` (and `R`); alternatively the singleton factor in the product decomposition is trivial. This supplies the additional preservation needed in T041.

For comparison, without positive leading mass, the same argument gives `m<=2d`, since every remaining vertex must be touched by one of the `d` cover transpositions. This coarse bound is sharp for a product of `d` disjoint adjacent covers. The positive-mu bound is attained at gap three by the canonical pair `1324<3412`, whose KL polynomial is `1+q` and whose core has four positions.

### Uniform transport and the remaining question

A pure step replaces `(x,w)` by `(xs,ws)` for a shared right descent with `x not<=ws`; it preserves the gap and polynomial. Normalize after each step using canonical naked-point cancellation. A pure step decreases the lower endpoint's inversion length by one, and deleting an entry and flattening cannot increase its inversion length. Thus every path length is at most the starting lower length `ell(x)`. For a canonical positive-mu core, `ell(x)=ell(w)-d<=binom(m,2)-d<=d(d-1)/2`, proving the uniform bound.

This bound applies only to paths made of the specified pure descent followed by normalization. Arbitrarily interspersing star operations, dualities or decompressions is outside its scope and can destroy the decreasing potential.

This closes the fixed-gap normalized-horizon finiteness question recorded after C014 and E038. It does not show that a terminal vertex has positive selected correction loss or positive atom mass. Q009 and Q010 remain open, but their canonical fixed-gap domains lie in the finite groups `S_m`, `m<=8,10,12` for gaps `7,9,11`, respectively. The theorem concerns those canonical domains directly; it does not transport their descent-labelled hypotheses through arbitrary embeddings or assert an efficient exhaustive search.

### Attribution and reference scope

Tsukerman and Williams, *Bruhat interval polytopes*, Definition 4.5 and Corollary 4.8 supply chain-independence of the position-support partition: https://dmtcs.episciences.org/2507/pdf . Dyer, *Hecke algebras and shellings of Bruhat intervals*, Compositio Mathematica 89 (1993), supplies the increasing-path description of `tilde R`: https://www.numdam.org/item/CM_1993__89_1_91_0.pdf . Björner–Brenti, *Combinatorics of Coxeter Groups*, Theorem 2.8.1 attributes to Dyer the older theorem that a length-`d` finite-Coxeter interval has a realization in a reflection subgroup of rank at most `d`: https://sites.math.washington.edu/~billey/classes/reflection.groups/references/EntireBook.pdf . Thus finite-rank representability is established background. Here the explicit flattening/product argument and naked-point identification are used to control the actual canonical type-A embedding and its normalized descent process. Whether the combined corollary already appears verbatim is unresolved; no novelty claim is made.

## Reduction to twelve positions

Identifier: T041. Date: 2026-09-22. Status: proved; independent audit A-T039-20260922 passed. This is a conditional obstruction theorem, not a constructed counterexample or a proof of combinatorial invariance.

**Theorem.** Suppose two isomorphic type-A Bruhat intervals of length thirteen have different Kazhdan–Lusztig polynomials. Orient them so that the scalar discrepancy `Delta` of T016–T017 is positive. Cancel every naked common point from the positive-discrepancy member, obtaining `[x,w]`. Then

    x,w in S_12;   ell(w)-ell(x)=13;
    mu(x,w) >= 126 Delta >= 126;
    [z^9] tilde R_{x,w}(z) >= Delta >= 1;
    x^{-1}w has exactly three cycles, counting fixed points.

Every saturated chain has connected position support, and its thirteen transpositions form a minimal-length transitive factorization of `x^{-1}w`: eleven steps join permutation cycles and two split them. Every nine-edge increasing Bruhat path is a shortest transposition factorization and has a forest as its position graph.

Thus an arbitrary-ambient-rank failure at the first unresolved length forces a canonical twelve-position, three-cycle witness. The theorem does not assert that such a witness exists or that every interval with these conditions violates combinatorial invariance. Nor does it bound the other member's core rank by twelve.

### A general short-path rank inequality

Let `[u,v]` be a canonical core of odd gap `d` with `mu(u,v)>0`, and let `m` be its number of positions. T039 makes the position-support graph of every saturated chain connected. Let `rho=u^{-1}v` have `c` cycles, including fixed points. Write the transpositions of a saturated chain as `t_1,...,t_d`, so `rho=t_1...t_d`.

In a partial product, each permutation cycle lies inside a connected component of the graph of transpositions used so far. Whenever a new transposition joins two graph components, it therefore joins two permutation cycles. Connecting all `m` vertices requires at least `m-1` such cycle-joining steps. If `J` and `S` are the total numbers of cycle joins and cycle splits, then

    J+S=d,    c=m-J+S,    J=(d+m-c)/2 >= m-1.

Hence `c<=d-m+2`, and the absolute transposition length satisfies

    ell_T(rho)=m-c >= 2m-d-2.

If `[z^{d-2r}] tilde R_{u,v}(z)>0`, Dyer's nonnegative increasing-path formula supplies a directed Bruhat path with `d-2r` transpositions. Thus `ell_T(rho)<=d-2r`, and combining the two inequalities gives

    m <= d-r+1.                                      (*).

This statement requires nonzero odd leading mass to obtain connected support. A disconnected product interval cannot be treated as connected merely because it has a short path. When `r=0`, (*) recovers T039's `m<=d+1`; for the four-short coefficient, `r=2` gives `m<=d-1`.

### Apply the inequality to a hypothetical discrepancy

T016 gives `tilde R_I-tilde R_{I'}=Delta z^9`, while T017 gives `P_I-P_{I'}=Delta(q^2-9q^3+36q^4-84q^5+126q^6)`. Since `Delta>0`, coefficient nonnegativity gives both `[z^9]tilde R_I>=Delta` and `mu(I)>=126 Delta`. Canonical cancellation preserves the gap, `P` and `tilde R`; the latter preservation is proved explicitly in T039 by fixing the deleted point throughout the interval and preserving the reflection-labelled graph. These inequalities therefore hold for the canonical core.

Apply (*) with `d=13,r=2` to obtain `m<=12`. The exhaustive coefficient sets through `S_11` quoted and source-audited in T020 have maximum 28, so the inequality `mu>=126` forces `m>=12`. Therefore `m=12` exactly.

The saturated-chain inequality now gives `c<=13-12+2=3`. A nine-transposition path gives `12-c=ell_T(rho)<=9`, hence `c>=3`. Thus `c=3`. Substituting into the join/split equations yields `J=11,S=2`; equality holds in the minimum connected-factorization length `d=m+c-2`. A shortest transposition factorization of length `m-c=9` has only cycle joins. Its edges therefore join previously disjoint graph components, giving a forest whose components are the cycles of the final permutation.

### Why this improves the earlier frontier

T020 supplied only an ambient lower bound of twelve from published coefficient magnitudes. T039 supplies the actual canonical support bound, and (*) uses the four-short-path coefficient to meet that lower bound exactly. The resulting finite-rank obstruction is independent of the E035 basin and E055 certificate counts. It is derived symbolically, with exact small-domain checks used to test conventions rather than infer the theorem.

A sufficient next target would be an upper bound below 126 for the leading coefficient on all canonical gap-thirteen `S_12` intervals with the stated three-cycle/short-path conditions. No such bound is proved here. These intervals have `d-(m-1)=2`, which suggests focusing on the two excess steps beyond a spanning tree, but a geometric description of those two steps remains necessary. This reduction does not justify calling every arbitrary `S_12` large-mu interval relevant: the published examples of coefficients 158 and 163 have gaps 25 and 27, not thirteen.

### Dependencies and priority

T039 provides the embedded factorization, connected-support conclusion and cancellation preservation. T016–T017 provide the discrepancy fingerprint, using established low-degree and length-twelve invariance; the contemporary length-twelve statement is explicit in Esposito–Marietti–Stella, arXiv:2509.16433v2. T020 cites McLarnan–Warrington and Warrington's exhaustive through-`S_11` coefficient sets. The short-path interpretation is Dyer's increasing-path formula. The join/split argument above is elementary permutation-factorization counting. This combined reduction is recorded as the present mathematical deduction; no external novelty assessment is claimed.

## Common descents and cancellation

Identifier: T042. Date: 2026-09-22. Status: proved; independent audit A-T042-20260922-review.md passed. This is a structural refinement of the transport reduction, not a bound on the leading coefficient.

Write `C(x,w)` for the canonical naked-point compression of a comparable pair. Use zero-based positions and `D_{x,w}(k,j)=#{a<k:w(a)>=j}-#{a<k:x(a)>=j}`. A common point `(k,j)` is naked when `x(k)=w(k)=j` and `D_{x,w}(k,j)=0`. Let `s_i` swap adjacent positions `i,i+1`, and assume it is a right descent of both `x` and `w`.

**Theorem.** The simultaneous swap `(x,w)->(xs_i,ws_i)` carries naked points bijectively to naked points, moving the points at positions `i,i+1` with the swap and fixing every other point. Let `C(x,w)=(a,b)`.

- If either swapped position is naked, the descent is necessarily pure (`x not<=ws_i`) and `C(xs_i,ws_i)=(a,b)`.
- If neither swapped position is naked, their surviving images are adjacent positions `j,j+1` of the core and `C(xs_i,ws_i)=(as_j,bs_j)`. This is a common descent of the core, and `x<=ws_i` if and only if `a<=bs_j`. Thus the original descent is pure precisely when the compressed descent is pure.

In particular a canonical pair stays canonical under every simultaneous common descent. Normalizing after each pure step is redundant on canonical input. Every genuine pure path keeps the canonical ambient rank and the cycle type of its relative permutation fixed. In the connected-support setting, the quantity `(d-m-c+2)/2`, where `m` is core rank and `c` the number of relative-permutation cycles, is therefore constant along the path.

### Proof of naked-point invariance

The positions of common points are simply permuted by `s_i`; no new common point is created. For any position outside `{i,i+1}`, the endpoint prefixes at that position have the same multisets before and after swapping, so its nakedness test is unchanged.

Suppose the upper of the two shared positions is common, `x(i)=w(i)=A`. Both other adjacent entries are smaller than `A` because the descent is shared. After the swap the common point has position `i+1`, and

    D_{xs_i,ws_i}(i+1,A)=D_{x,w}(i,A).

Suppose instead `x(i+1)=w(i+1)=B`. Both entries at position `i` exceed `B`. The common point moves to position `i`, and

    D_{xs_i,ws_i}(i,B)=D_{x,w}(i,B)=D_{x,w}(i+1,B).

This proves the bijection, including the case where both positions are common. By T031, deleting all these naked points produces the canonical core independently of deletion order.

### Why a moved naked point forces pure transport

If `x(i)=w(i)=A` is naked, then `D_{x,w}(i+1,A)=0`: both original prefixes acquire one entry at least `A`. Swapping only the upper endpoint to form the facet `ws_i` removes that entry, giving

    D_{x,ws_i}(i+1,A)=-1.

If instead the naked point is `x(i+1)=w(i+1)=B`, the prefix before that position contains no occurrence of `B`, so nakedness also gives `D_{x,w}(i+1,B+1)=0`. Swapping only `w` again removes one qualifying entry, yielding `D_{x,ws_i}(i+1,B+1)=-1`. Either negative rank defect proves `x not<=ws_i`. This argument uses the actual Bruhat criterion, not only polynomial preservation.

Deleting the moved naked point(s) erases the local swap: the order of all surviving entries and their value set remain unchanged. Hence the compressed pair is identical.

### When both positions survive

All deleted positions are outside the adjacent pair, so deleting them carries the simultaneous swap to the indicated adjacent swap on the compressed pair. Increasing flattening preserves the descent comparisons.

For endpoint comparability, deleting a common entry preserves and reflects Bruhat order even when that entry is not naked: in the rank-defect matrix its position gives a repeated row and its value gives a repeated column; all other entries become the smaller pair's rank defects. Apply this observation to the common entries deleted from `(x,ws_i)`. They are still common because none lies in the swapped positions. The resulting pair is exactly `(a,bs_j)`, proving the comparability equivalence and hence purity equivalence.

Finally `(xs_i)^{-1}(ws_i)=s_i(x^{-1}w)s_i`, so simultaneous common descent conjugates the relative permutation. On canonical input no point is removed, preserving its cycle type, rank and gap. The stated genus expression is therefore unchanged. QED.

## Genus heredity

Identifier: T044. Date: 2026-09-22. Status: proved; independent audit A-T044-20260922-review.md passed (commit `[withheld for anonymous review]`). This theorem concerns support and relative cycles, not a bound on KL coefficients.

For a finite transposition word `t_1...t_L` on `m` vertices, let `k` be the number of components of its edge graph, including isolated vertices, and let `c` be the number of cycles of its permutation product, including fixed points. Define its total genus by

    G=(L-m-c+2k)/2.

**Word lemma.** The total genus is a nonnegative integer. It does not decrease when another transposition is appended. It is unchanged by reversing the word. Consequently every contiguous subword has genus at most that of the whole word.

**Proof.** The empty word has `L=0` and `k=c=m`, hence genus zero. On appending a transposition, there are three cases. If its endpoints lie in different current graph components, they also lie in different permutation cycles; `L` increases by one and both `k,c` decrease by one, so `G` is unchanged. If the endpoints lie in one graph component but different permutation cycles, `k` is unchanged and `c` decreases by one, so `G` increases by one. If they lie in one permutation cycle, `k` is unchanged and `c` increases by one, so `G` is unchanged. Thus `G` counts precisely the joins of permutation cycles whose endpoints were already connected in the edge graph. Reversal inverts the permutation product and preserves its cycle count and edge graph. To retain a contiguous subword, first remove a suffix, which cannot increase genus, then reverse and remove the remaining unwanted prefix as a suffix. QED.

For a Bruhat interval `[x,w]` in `S_m`, define `G(x,w)` as the sum of the genera of its support components, giving singleton equality factors genus zero. T039's support theorem and a saturated chain show that this equals the word expression above, with `L=ell(w)-ell(x)` and `c=cycles(x^{-1}w)`.

**Heredity theorem.** If `[a,b]` is a subinterval of `[x,w]`, then

    0 <= G(a,b) <= G(x,w).

In particular every support component of every subinterval of a genus-zero interval has genus zero. The original source need not be connected or canonical for this assertion.

**Proof.** Extend a saturated chain from `a` to `b` to one from `x` to `w`. The transposition word of the shorter chain is a contiguous subword of the longer chain. Apply the word lemma and identify word genus with the support-component sum. Each component genus is nonnegative. QED.

### Endpoint-cover dichotomy

Suppose `[x,w]` is connected and genus zero on `m` vertices, with `c` relative cycles. Delete its first or last cover along any saturated chain, retaining a subinterval of gap one less. With all `m` vertices still counted, the resulting support has exactly one of the following forms:

- one component, with `c-1` relative cycles;
- two components, with `c+1` relative cycles in total.

Singleton equality components are allowed here. Indeed, the new relative permutation differs by multiplication by a transposition, so its cycle count is `c'=c+1` or `c-1`. Heredity gives `0=(d-1-m-c'+2k')/2`; subtracting `0=(d-m-c+2)/2` yields `2k'=3+c'-c`. Thus `k'=1` or `2` as stated. All nonsingleton components are canonical genus-zero intervals by T039.

### Consequence for the twelve-position frontier

For T041's `(m,d,c)=(12,13,3)` witness, every endpoint-cover facet is either connected with two relative cycles and complexity one, or has two genus-zero components with four relative cycles in total. A connected proper subinterval that still uses all twelve positions has gap at most twelve, and genus zero gives `c'=d'-10>=1`. Therefore its only possibilities are gap twelve with two relative cycles, or gap eleven with one relative cycle (complexity zero). Smaller gaps necessarily have smaller connected support.

This is an induction-domain theorem: all proper support factors stay inside genus zero. It does not close the KL recurrence. A shared-descent recurrence additionally contains the same-gap pair `(xs,ws)` and the highest permitted coefficient of the even-gap facet `(x,ws)`; bounding these terms against the selected correction is still necessary. T042 preserves genus under common descents but does not supply that inequality. C017 prohibits substituting a naive polynomial domination under common-point deletion.

### Attribution and limits

The proof is elementary transposition cycle counting together with the existing support-factorization theorem T039. No learned mechanism, new coefficient computation, or generic torus-complexity bound is claimed. The torus interpretation of `d-m+1` is the published Donten-Bury–Escobar–Portakal formula. External novelty of this elementary hereditary formulation has not been assessed. Q011 and the coefficient bound required for the length-thirteen target remain open.

## Corrected descent gain

Identifier: T045. Date: 2026-09-22. Status: proved; independent audit A-T045-20260922-review.md passed (commit `[withheld for anonymous review]`). The descent-gain bound that would exclude this obstruction is not proved.

Let `[x,w]` be a canonical connected interval in `S_m` with odd gap `d>=3` and `mu(x,w)>0`. For a shared right descent `s`, put

    A_s(x,w)=[q^((d-3)/2)]P_{x,ws},
    C_s(x,w)=sum_{x<z<=ws, zs<z, ell(z)-ell(x) odd} mu(x,z)mu(z,ws),
    B_s(x,w)=A_s(x,w)-C_s(x,w).

Off-order polynomials and empty sums are zero. The standard leading recurrence is

    mu(x,w)=mu(xs,ws)+B_s(x,w).                         (1)

**Theorem.** After possibly replacing the pair by `(w_0w,w_0x)`, there is a sequence of at most

    L_max=floor((binom(m,2)-d)/2)

shared right descents, stopped at the first zero leading coefficient, for which

    mu(x,w)=sum_{j=0}^{L-1} B_{s_j}(x_j,w_j),  1<=L<=L_max.    (2)

Every source pair in this sum remains canonical and connected, has the same rank, gap and relative cycle count, and hence the same factorization genus. Consequently some source pair satisfies

    B_{s_j}(x_j,w_j)>=ceil(mu(x,w)/L_max).              (3)

The zero endpoint of the sequence is not asserted connected or genus zero componentwise.

**Proof.** The usual long-element symmetry preserves the leading coefficient: `mu(x,w)=mu(w_0w,w_0x)`. These two orientations have lower lengths summing to `binom(m,2)-d`, so one has lower length at most `L_max`. Reversing a saturated chain and multiplying its vertices on the left by `w_0` preserves its position-transposition graph. It therefore preserves support connectedness and canonicality. The relative permutation is inverted, so its cycle count is unchanged.

At any current pair with positive leading coefficient and gap at least three, every right descent of the upper endpoint is a descent of the lower endpoint by the standard extremality lemma. Select one. The simultaneous descent lowers both endpoint lengths by one, retains their gap, and remains comparable by lifting. T042 preserves canonicality and relative cycle count under every common descent. If the transported coefficient stays positive, T039 forces its canonical support to be connected, so its genus is unchanged. If it vanishes, stop. The process must stop by the time the lower length reaches zero, since a lower identity has no right descent and cannot form a non-cover positive-leading pair. Thus it takes at most `L_max` steps. Telescope (1), whose final leading coefficient is zero, to obtain (2); averaging gives (3). QED.

### Exact frontier consequence

T041 confines a hypothetical positive length-thirteen discrepancy to a canonical connected `S_12` pair of genus zero with `mu>=126 Delta` for an integer `Delta>=1`. Here `L_max=floor((66-13)/2)=26`. Therefore a shared-descent path of at most twenty-six steps reaches a positive-leading genus-zero source with

    B_s>=ceil(126 Delta/26)>=5,  hence A_s>=C_s+5.     (4)

In particular this is an on-order, nonpure descent facet. By T044, that even-gap facet is either a connected complexity-one interval or a two-component genus-zero product. The coefficients in its correction lie in proper genus-zero support factors. The twenty-six-step conclusion allows common descents with nonzero correction; it is different from T039's pure-descent horizon.

A sufficient remaining lemma is therefore the **local bound** `B_s<=4` for every common right descent of a canonical connected genus-zero `S_12` pair of gap thirteen with positive leading coefficient. If proved, (2) would give `mu<=104`, contradicting T041. The stronger statement `A_s<=4` would also suffice but is not established. Neither statement is inferred from low torus complexity, T043's deletion theorem, or the finite controls. This reformulation reduces the unresolved estimate to a one-step correction inequality without claiming to prove that inequality.

### Sources and limitations

The leading recurrence is fixed in T029. The long-element leading symmetry and extremality appear explicitly as Facts 6 and 5, respectively, in Warrington, *Equivalence classes for the mu-coefficient of Kazhdan–Lusztig polynomials in S_n*, [primary source](https://arxiv.org/html/1010.3961). T039, T042 and T044 supply the support and genus statements. Full KL polynomials are not claimed invariant under long-element order reversal. This deduction does not establish Q011 or combinatorial invariance at length thirteen; external novelty remains unassessed.
