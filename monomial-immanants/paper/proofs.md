# P18: width-two positivity proof

This document collects the proof and its supporting lemmas for [the results summary](results.md). The originating internal proof review accepted the argument after clarifying the tableau-descent indexing. The fresh exact replay checks its finite instances and conventions. External mathematical review and publication novelty remain unresolved.

## Statement and normalization

Let `D=mu/nu` be a nonempty skew shape of size `N`, represented with `n` rows, and let `theta` partition `N`. Let `phi^lambda` be the class function of `S_n` with Frobenius characteristic `m_lambda`. The claim is

```math
\theta_2\le2\quad\Longrightarrow\quad
[s_\theta]\mathrm{Imm}_{\phi^\lambda}(H_D)\ge0
\qquad\text{for every }\lambda\vdash n.
```

A missing second part is zero. There is no rank bound or boundary-spacing restriction. Empty rows can be removed for the pointwise proof and restored by the known induction rule. The single-column coefficient partition is covered by the established hook theorem; the integrated proof treats `(A,2^b,1^c)` with `A>=2`.

The zero-based endpoints are `b_i=mu_i+n-1-i`, `c_j=nu_j+n-1-j`; a permutation selects `alpha_j(w)=b_{w(j)}-c_j`. A negative content entry has Kostka weight zero. Write `hat h(w)=1{w(j)<=h(j) for every j}` for a weakly increasing Hessenberg function with `h(j)>=j`. After removing empty rows, the stronger statement is

```math
K_{\theta,\alpha(w)}=\sum_h a_h\widehat h(w),\qquad a_h\in\mathbb Z_{\ge0}.
```

For the conjugacy class `C_rho` containing `w`, define

```math
\Gamma_D^\theta(w)=\frac{n!}{|C_\rho|}
 \sum_{v\in C_\rho}K_{\theta,\alpha(v)},\qquad
\Gamma_h(w)=\frac{n!}{|C_\rho|}\sum_{v\in C_\rho}\widehat h(v).
```

Applying this same normalization to the pointwise identity gives `Gamma_D^theta = sum_h a_h Gamma_h`. Lesnevich's immanant pairing identifies the target coefficient with the usual inner product of `Gamma_D^theta` and `phi^lambda`. Hikita's positivity theorem makes each `Gamma_h` a nonnegative integer combination of characters induced from trivial representations of Young subgroups. Their Frobenius characteristics are the complete functions `h_lambda`, dual to `m_lambda`, so this gives the stated nonnegativity. These published inputs are attributed separately from the width-two decomposition.

## Proof map

| Ingredient | Included section |
| --- | --- |
| T002: feasibility and zero/one counts on prefix fibers | [Prefix fibers](#prefix-fibers) |
| T003: upper-poset Möbius inversion | [Corner inversion](#corner-inversion) |
| T051: adjacent loss sets | [Loss geometry](#loss-geometry) |
| Bernstein and dual Cauchy calculation | [Content series](#content-series) |
| Two-state transfer recurrence | [Path factors](#path-factors) |
| T053: spare large entry and all-ones boundary | [Integrated proof](#integrated-proof) |

The zero-based tableau-descent convention is explicit in the integrated proof: descent `i` means that entry `i+2` is below entry `i+1`. The finite checker independently compares this count with full-poset inversion. Historical references to C005 denote the conjecture resolved by T053; the unresolved broader immanant problem and fiber-connectivity questions are separate.

## Established references and review

- [Lesnevich, Hook-shape immanant characters from Stanley–Stembridge characters](https://numdam.org/articles/10.5802/alco.331/), Theorem 1.4, Definition 2.8, Lemma 2.9, and the empty-row reductions: hook background and the character formulation.
- [Hikita, A proof of the Stanley–Stembridge conjecture](https://arxiv.org/abs/2410.12758): positivity for the Stanley–Stembridge summands. This is distinct from the full monomial-immanant conjecture.
- Bernstein creation, dual Cauchy, Littlewood–Richardson multiplication, tableau standardization, and finite-poset Möbius inversion are standard ingredients, not discoveries claimed here.

The full internal T053 review and its original hash are recorded in `research/SOURCE_MAP.json`. The statement and supporting arguments are exported below. The retained exact results and fresh replay use an independently written horizontal-strip Kostka recursion, full-poset subtraction, and local-corner comparisons. Model training is not part of this proof or its checker.

## Prefix fibers

Identifier: T002
Date: 2026-09-16
Status: proved in campaign; internally audited (fresh parent auditor, pass)

### Statement

Let $`\mu/\nu`$ be a skew shape with $`n`$ nonempty rows and size $`N\ge4`$.
For $`w\in S_n`$, set
$`c(w)=\mu+\delta-w(\nu+\delta)`$, with
$`\delta=(n-1,\ldots,0)`$, and interpret a Kostka number with a negative content
entry as zero.  In the equivalent column-to-row convention, let

```math
 p(w)_j=\max_{q\le j}w(q).
```

Then $`K_{(N-2,2),c(w)}`$ depends only on $`p(w)`$.

More precisely, on every feasible prefix fiber the number of zero selected
Jacobi--Trudi indices and the number of indices at least two are individually
constant.

### Proof

Use zero-based indices temporarily and put

```math
 b_i=\mu_i+n-1-i,\qquad d_j=\nu_j+n-1-j.
```

Both sequences are strictly decreasing by at least one at every step.  The
coordinates of $`c(w)`$, merely reordered, are
$`c(w)_{w(j)}=b_{w(j)}-d_j`$.

The record positions $`j`$, record values $`m=w(j)=p(w)_j`$, and their selected
content values $`b_m-d_j`$ are fixed by the prefix vector $`p(w)`$.  Consider a
nonrecord position $`j`$, put $`y=w(j)`$, and choose an earlier record position
$`\ell`$ carrying the current record $`m=p(w)_j`$.  Then $`m>y`$, and strict
decrease gives

```math
 b_y-d_j=(b_m-d_\ell)+(b_y-b_m)+(d_\ell-d_j)
 \ge (b_m-d_\ell)+(m-y)+(j-\ell)
 \ge (b_m-d_\ell)+2.                                      \tag{1}
```

If some record content is negative, every permutation in the fiber is
infeasible.  Conversely, if all record contents are nonnegative, (1) shows that
every nonrecord content is at least two, for every permutation in the fiber.
Thus feasibility is fiber-constant.  On a feasible fiber, only the fixed record
contents can be zero or one; every nonrecord coordinate contributes to both the
positive count $`r`$ and the at-least-two count $`t`$.

More explicitly, let $`q`$ be the number of records, and among their fixed
contents let $`z`$ be the number equal to zero and $`o`$ the number equal to one.
Then

```math
 r=n-z,\qquad t=n-z-o,
```

which depend only on $`p(w)`$.  Finally,

```math
 K_{(N-2,2),c}=\binom{r-1}{2}+t-1
```

for nonnegative $`c`$, proving the claim. $`\square`$

For completeness, the Kostka formula follows at once from
$`s_{(N-2,2)}=h_{N-2}h_2-h_{N-1}h_1`$.  The first product has
$`\binom r2+t`$ ways to choose a size-two subcontent and the second has $`r`$,
giving $`\binom r2+t-r=\binom{r-1}{2}+t-1`$.

## Corner inversion

Identifier: T003
Date: 2026-09-16
Status: proved; internally audited (fresh parent auditor, pass)

### Statement

Use zero-based Hessenberg functions
$`h=(h_0,\ldots,h_{n-1})`$, so $`i\le h_i\le n-1`$ and $`h`$ is weakly
increasing.  Let $`F(h)`$ be the well-defined prefix-fiber weight from T002 and
let $`a_h`$ be determined by

```math
 F(k)=\sum_{h\ge k}a_h.
```

Define the addable positions

```math
 A(h)=\{i\in\{0,\ldots,n-2\}:h_i\lt h_{i+1}\}.
```

For $`S\subseteq A(h)`$, write $`h+\mathbf1_S`$ for the Hessenberg function
obtained by increasing coordinates in $`S`$ by one.  Then

```math
 \boxed{\displaystyle
 a_h=\sum_{S\subseteq A(h)}(-1)^{|S|}F(h+\mathbf1_S).}
```

If the record contents of $`g`$ include a negative value, set $`F(g)=0`$.
Otherwise, if $`z(g)`$ and $`o(g)`$ count record contents equal to zero and one,
respectively, T002 makes this completely explicit:

```math
 F(g)=\binom{n-z(g)}2-o(g).                              \tag{1}
```

Thus C001 is equivalent to nonnegativity of the displayed finite difference for
every skew shape and every Hessenberg function.

### Proof

Associate to $`h`$ its Hessenberg diagram
$`D(h)=\{(i,j):i\lt j\le h_i\}`$.  These diagrams form the distributive lattice of
order ideals in the type-$`A`$ positive-root poset, with coordinatewise order on
functions corresponding to inclusion of diagrams.  For order ideals $`I\subseteq
J`$, the Möbius function is zero unless $`J\setminus I`$ is an antichain, in
which case it is $`(-1)^{|J\setminus I|}`$.  This follows directly by Boolean
inclusion--exclusion on the minimal elements of the complement.

The minimal cells outside $`D(h)`$ are precisely
$`(i,h_i+1)`$ with $`i\in A(h)`$.  Adding any subset produces
$`D(h+\mathbf1_S)`$, and these are exactly the upper diagrams whose difference
from $`D(h)`$ is an antichain.  Möbius inversion of
$`F(k)=\sum_{h\ge k}a_h`$ gives the formula.  Finally, (1) is the simplification
$`\binom{r-1}{2}+t-1=\binom{n-z}{2}-o`$, using
$`r=n-z`$ and $`t=n-z-o`$ from T002. $`\square`$

## Loss geometry

Excerpt from the T051 proof. Endpoints and feasible corner conventions are defined in [the integrated width-two proof](#integrated-proof). The argument below concerns the positive-entry and one counts and applies to the entire width-two family.

### One-removal sets on a legal corner cube

Write R,O for the initial positive-content and one counts. A selected zero at column j is the condition hⱼ=i with bᵢ=cⱼ; this cannot be destroyed by a legal raise. A one at column j, with bᵢ=cⱼ+1, is counted by 1[hⱼ=i]−1[hⱼ₋₁=hⱼ=i]. The preceding feasible bound is at most i, so these are exactly its record and plateau conditions. Omit the plateau term at j=0.

For Boolean legal-corner coordinates xᵢ, the statistics have the form

```math
r=R-\sum_i u_i x_i,\qquad o=O+\sum_i v_i x_i+\sum_{i\lt j}Q_{ij}x_ix_j. \tag{2}
```

Here uᵢ∈{0,1}. If uᵢ=0, no one is lost and at most one is created, so vᵢ∈{0,1}. If uᵢ=1, define Lᵢ to be the set of initially selected ones removed by this raise. Only its own position and the immediately following record can be affected, so |Lᵢ|≤2 and vᵢ=−|Lᵢ|.

A quadratic term can arise only from a plateau −xᵢ(1−xⱼ) at consecutive legal corners. Such an edge Qᵢⱼ=1 forces uᵢ=uⱼ=1: the left raise changes a record to zero, the right initial record is one, and feasibility of the right raise forces its new content to be zero. The edge counts their one shared loss, so Qᵢⱼ=|Lᵢ∩Lⱼ|=1. Distinct nonadjacent corners have disjoint loss sets. A selected one can be removed only at its own corner and the preceding corner; there is no triple overlap. Thus the graph of Q-edges is a subgraph of the path of legal corner positions. In particular it has at most two edges on three vertices. These descriptions also prove (2) directly without assuming the signs to be shown.

## Content series

Date: 2026-09-22. Status: supporting symbolic lemma; reviewed in the passed full T053 audit. No novelty claim.

Let θ=(A,2ᵇ,1ᶜ) with A≥2 and b,c≥0, let N=A+2b+c, and let α be any nonnegative content of total N. Write r for the number of positive entries and o for the number equal to one. Set U=(1+x)(1+y) and V=1+x+y. Then

```math
K_{\theta,\alpha}=[x^{b+c}y^b](1-y/x)U^{r-o-1}V^o. \tag{1}
```

The expression is a formal power series when r=o; its negative exponent is intentional and does not require a content entry at least two. Consequently the Kostka weight for every coefficient shape with second part at most two is constant on feasible prefix fibers, by the zero-and-one invariance in T002. This proves the constancy half of C005, not the nonnegativity of its canonical Möbius coefficients.

### Uniform proof, including the all-ones boundary

Let β=(2ᵇ,1ᶜ), so β′=(b+c,b). The standard first-row Bernstein identity is

```math
s_{(A,\beta)}=\sum_{j\ge0}(-1)^j h_{A+j}e_j^\perp s_\beta.
```

Here e_j^⊥ is adjoint to multiplication by e_j for the Hall inner product. Dual Cauchy and its skew version give

```math
\sum_{\gamma:\,\gamma_1\le2}(e_j^\perp s_\gamma)(X)s_{\gamma'}(x,y)=h_j(x,y)\prod_i(1+X_i x)(1+X_i y).
```

One can see this directly by applying e_j^⊥ to the X alphabet in the dual Cauchy identity: its eigenfactor is ω(e_j)(x,y)=h_j(x,y). For a monomial X^δ from the right-hand product, taking the coefficient of X^α after multiplication by the appropriate complete function contributes one exactly when δ≤α coordinatewise. The degree of that complete function is N−|γ|+j, so no extra multiplicity is introduced. Truncating independently at each α_i gives a factor 1 for α_i=0, V for α_i=1, and U for α_i≥2. Thus the coefficient associated with γ is obtained from

```math
\left(\sum_{j\ge0}(-1)^j h_j(x,y)\right)U^{r-o}V^o=U^{r-o-1}V^o.
```

All calculations can be truncated at degree |β|, so the formal inverse is harmless. The coefficient of s_(b+c,b) in a symmetric two-variable series G is `[x^(b+c)y^b](1−y/x)G`, proving (1). The Bernstein identity is used only for the valid partition (A,β), with A≥2; coefficients corresponding to an invalid first row are not given a positivity interpretation.

For r=o=N, (1) equals the number of standard tableaux of θ. The hook-length formula gives the explicit positive boundary value

```math
[x^{b+c}y^b](1-y/x)\frac{V^N}{U}=\frac{N!(c+1)}{(A+b+c)(A+b-1)(A-2)!(b+c+1)!b!}.
```

This also covers b=0. In particular no anchor hypothesis is needed for (1).

## Path factors

Date: 2026-09-22. Status: supporting symbolic argument for T053; reviewed in its passed full independent audit. This note establishes an algebraic identity, not a claim of positivity without the accompanying geometry.

Put U=(1+x)(1+y), V=1+x+y, W=xy, and h_j=h_j(x,y), with h₀=1 and h₋₁=0. Consider a component consisting of s≥1 zero-creating legal corners joined by shared-one losses. Each internal edge represents one initially selected one removed by either adjacent corner. An unshared initial one can occur at either endpoint; write a,b∈{0,1} for their presence and e=a+b. For s=1 the two possible endpoint ones are distinct, even though incident to the same vertex. There are ℓ=s−1+e initial ones in this component. For a selected vertex subset S, let C(S) be the number of those ℓ ones incident to at least one selected vertex.

Define the cleared polynomial

```math
M_{s,e}=\sum_{S\subseteq[s]}(-1)^{|S|}U^{s-|S|+C(S)}V^{\ell-C(S)}.
```

Then, independently of which endpoint is used when e=1,

```math
M_{s,0}=U^{s-1}(h_s+Wh_{s-1}),\qquad M_{s,1}=U^s h_s,\qquad M_{s,2}=U^s(h_s+h_{s+1}).
```

### Transfer proof

Replace S by its complement T. A one is uncovered precisely when both incident vertices lie in T (for an internal edge) or when its incident endpoint lies in T (for an endpoint one). The defining sum becomes M_{s,e}=(-1)^s U^ℓ Z_s, where

```math
Z_s=\sum_T(-U)^{|T|}(V/U)^{\#\text{uncovered ones}}.
```

Keep the two partial sums corresponding to the final vertex being absent from or present in T. On appending a vertex the transfer matrix is

```math
\begin{pmatrix}1&1\\-U&-V\end{pmatrix}.
```

The initial column is (1,−U(V/U)^a)ᵀ, and the terminal row is (1,(V/U)^b). Its characteristic polynomial is λ²+(V−1)λ+(U−V)=(λ+x)(λ+y). Therefore (−1)^s Z_s satisfies the same recurrence as h_s: f_s=(x+y)f_{s−1}−xy f_{s−2}. Checking s=1 and s=2 proves all three displayed identities for every s. The temporary powers of U⁻¹ are formal power series with invertible constant term; every displayed final M is a polynomial.

### The full corner difference

Write R,O for the initial positive-entry and one counts and B=R−O. Let z be the number of corners that create a one without creating a zero. Any corner that changes neither statistic makes the full difference zero. The shared-loss graph of the remaining zero-creating corners is a disjoint union of paths. Let c₀,c₁,c₂ count its components by endpoint type and s be their total number of vertices. Distinct components have disjoint initial loss sets. There are O−s+c₀−c₂ initial ones outside all these loss sets, a nonnegative number denoted f.

For a corner subset, use the width-two content series U^{r−o−1}V^o. A one-creating corner contributes the difference factor 1−V/U=W/U. A zero-creating component contributes its cleared polynomial divided by U^s and with its original one factors removed. Consequently the complete alternating series is exactly

```math
W^z U^{B-1-z-c_0}V^f
\prod_{e=0}(h_s+Wh_{s-1})\prod_{e=1}h_s\prod_{e=2}(h_s+h_{s+1}),
```

where each product uses the size of its own component. The sum over all subsets is implicit in this factorization; no signs remain in these factors.

If B−1−z−c₀≥0, every displayed factor is Schur-positive in two variables: U=1+s_(1)+s_(1,1), V=1+s_(1), W=s_(1,1), and h_s=s_(s). Products preserve Schur positivity by the Littlewood–Richardson rule. Thus every width-two canonical coefficient is nonnegative in this case. The separate geometric argument in the integrated proof must ensure this exponent condition whenever B>0, and treat the initial all-ones case B=0 separately. This algebraic note does not silently assume those remaining statements.

## Integrated proof

Identifier: T053. Date: 2026-09-22. Status: proved; fresh independent internal audit passed; novelty unresolved.

For every skew shape D=μ/ν of size N≥1, with empty rows removed, and every coefficient partition θ of N satisfying θ₂≤2, the Kostka permutation weight has a pointwise nonnegative integral expansion in Hessenberg indicators. Consequently the immanant character is a nonnegative integral sum of Stanley–Stembridge characters. In particular this proves C005, including every θ=(A,2ᵇ,1ᶜ) with A≥2 and arbitrary b,c≥0, without rank, boundary-spacing, or small-index restrictions. Hooks are already known; the proof below treats the displayed family uniformly and invokes the known hook case for the remaining single-column shape.

This is an all-parameter proof using a content generating series, an exact path-factor identity, and a geometric boundary argument. It is not an extrapolation from the earlier width-two pilot. The assertion is restricted to coefficient shapes whose second part is at most two, rather than all coefficient partitions. No fiber-connectivity or publication-novelty claim is made.

### Content series and prefix reduction

Use strictly decreasing endpoints bᵢ=μᵢ+n−1−i and cⱼ=νⱼ+n−1−j, indexed from zero. In the column-to-row convention a permutation w selects αⱼ=b_(w(j))−cⱼ. Negative selections have weight zero. Let hⱼ=max_(q≤j)w(q). By [prefix-fiber constancy](#prefix-fibers), every feasible nonrecord selection is at least two; hence the positive-content count r and the one count o are constant on a feasible prefix fiber, without assumptions on either boundary.

Put U=(1+x)(1+y), V=1+x+y, and W=xy. The [uniform content-series lemma](#content-series), proved by Bernstein and dual Cauchy including the all-ones boundary, gives

```math
K_{(A,2^b,1^c),\alpha}=[x^{b+c}y^b](1-y/x)U^{r-o-1}V^o. \tag{1}
```

The right-hand side extracts the coefficient of s_(b+c,b)(x,y) in the symmetric formal series U^(r−o−1)Vᵒ. Thus it suffices to prove nonnegativity of that Schur coefficient after each canonical legal-corner alternating difference. The exponent −1 when r=o is a formal power series, not an omitted exceptional case.

[Corner inversion](#corner-inversion) supplies the Möbius formula a_h=Σ_(S⊆A(h))(−1)^|S|F(h+1_S). If h is infeasible then its entire upper interval is infeasible and a_h=0. At a feasible h, discard any corner raise that exceeds the feasible bound H: every term containing it is zero. Every vertex of the remaining legal-corner cube is feasible. Let R,O denote its initial positive and one counts, and B=R−O its initial count of entries at least two.

### Exact loss-set decomposition

Use the [loss-set geometry](#loss-geometry) extracted from T051. Each legal corner has u∈{0,1}, where u records whether the raise creates a zero. If u=0, it either changes neither statistic or creates one new one. No shared interaction involves such a corner. A corner changing neither statistic makes the entire alternating difference zero, so assume none occurs and let z count the corners creating a one but no zero.

Each u=1 corner removes a set Lᵢ of initially selected ones, of size at most two: its own one and possibly the following column's one. Two loss sets overlap only at adjacent corners and in a single one; no one belongs to three sets. Draw an edge for each shared one. The connected components are paths, including isolated vertices. In a path of s corners, the s−1 internal edges account for s−1 shared ones. There may additionally be e∈{0,1,2} unshared endpoint ones. Interior vertices cannot carry another one because they already have two shared losses. For an isolated corner, e is its number of removed ones. Let c₀ and c₂ count the components with e=0 and e=2, and let s_tot be the total number of zero-creating corners.

Different components use disjoint initial ones. Their total number of such ones is s_tot−c₀+c₂, so

```math
q=O-s_{\rm tot}+c_0-c_2\ge0. \tag{2}
```

For a component, let cov(S) count its initial ones covered by the subset S of its corners. The cleared alternating polynomial is

```math
M_{s,e}=\sum_{S\subseteq[s]}(-1)^{|S|}U^{s-|S|+\mathrm{cov}(S)}V^{s-1+e-\mathrm{cov}(S)}.
```

Writing h_j=h_j(x,y), with h₀=1, the exact path identities are

```math
M_{s,0}=U^{s-1}(h_s+Wh_{s-1}),\qquad M_{s,1}=U^s h_s,\qquad M_{s,2}=U^s(h_s+h_{s+1}). \tag{3}
```

These are finite polynomial identities for all s≥1. The [supporting path-factor proof](#path-factors) derives them by the two-state recurrence for whether the last path corner is selected; it is a symbolic identity, rather than a fit to computed paths.

Applying (3) to each component, and U−V=W to each one-creating corner, factors the full alternating content series as

```math
W^z U^{B-1-z-c_0}V^q\prod_{e=0}(h_s+Wh_{s-1})\prod_{e=1}h_s\prod_{e=2}(h_s+h_{s+1}). \tag{4}
```

If B−1−z−c₀≥0, every factor in (4) is Schur-positive in x,y: U=1+s₁+s₁₁, V=1+s₁, W=s₁₁, and h_j=s_(j). The Littlewood–Richardson product rule therefore proves nonnegativity of every coefficient extracted in (1).

### A spare large entry removes every nontrivial denominator

Every one-creating corner uses a distinct initially selected entry at least two: it cannot remove an initial one and cannot start at a zero. The first corner in an e=0 component likewise has an initially selected entry at least two. If it had its own initial one, that one would be unshared, since there is no preceding shared edge in its component; this contradicts e=0. These positions are distinct, proving B≥z+c₀.

We need the stronger inequality B≥z+c₀+1 whenever B>0. Choose the rightmost position j whose initially selected entry is at least two, using any fixed representative permutation of the prefix h. All later positions are records, since every nonrecord entry is at least two. The strictly increasing tail of record values, ending at n−1, forces w(l)=h_l=l for every l>j.

If j=n−1, it is not a corner and is already an unused large entry. If j<n−1 but is not a legal corner, the same conclusion holds. Otherwise h_j<h_(j+1)=j+1, together with h_j≥j, gives h_j=j. Put d=b_(j+1)−c_(j+1), the following initial record entry. It is in {0,1}. The raised selection is d−(c_j−c_(j+1)); its feasibility forces d=1, c_j−c_(j+1)=1, and raised selection zero. Hence j is not a one-creating corner. It is a zero-creating corner removing the following initial one.

Follow this removed one to the next corner if it is shared. Any such next corner also lies in the diagonal record tail and must remove the next initial one by the same feasibility argument. This process moves strictly rightward and terminates: the last position has no corner. Its terminal one is unshared. Thus the component containing j has e≥1, so j is not the assigned first position of an e=0 component either. The rightmost large entry is therefore distinct from all z+c₀ assigned entries. This proves the required strict bound and makes the exponent of U in (4) nonnegative whenever B>0.

### The all-ones boundary is a descent count

It remains to consider B=0. Every initially selected entry is zero or one. Since D has n nonempty rows, N≥n, while the selected content has total N=O=R≤n. Hence N=n and all n selections are one. They are all records, forcing w=id and h=id. Therefore b_i=c_i+1, or equivalently every row of D contains exactly one box.

A corner i of the identity prefix is legal exactly when c_i−c_(i+1)=1. Raising a subset S of these corners merges each maximal run of chosen adjacent boundaries into a single positive content entry equal to its block size, leaving zeros in the other positions of that block. This follows directly from the cyclic representative w(a)=a+1,…,w(b−1)=b,w(b)=a on a run {a,…,b−1}: its first selections are zero and its final selection is b_a−c_b=b−a+1. Thus, after removing zero entries, the selected content is the composition of N whose adjacencies inside blocks are precisely the chosen positions S.

Use zero-based descent indices throughout this paragraph: Des(T)={i∈{0,…,N−2}: the entry i+2 lies in a lower row than i+1}. By standardization, K_(θ,α(S)) counts standard tableaux of θ whose descent set avoids S. Indeed a block of equal letters in a semistandard tableau standardizes to consecutive labels containing no descent, and conversely any standard tableau with that avoidance property destandardizes uniquely. Inclusion–exclusion now gives

```math
a_{\mathrm{id}}=\sum_{S\subseteq A(h)}(-1)^{|S|}\#\{T\in\mathrm{SYT}(\theta):\mathrm{Des}(T)\cap S=\varnothing\}=\#\{T\in\mathrm{SYT}(\theta):A(h)\subseteq\mathrm{Des}(T)\}\ge0.
```

This boundary argument works for every coefficient shape, not just the width-two family. Together with (4) and the spare-entry argument, it completes the canonical nonnegativity proof in all cases.

### Consequence and scope

The integers a_h give the pointwise identity K_(θ,α(w))=Σ_h a_h ĥ(w). Conjugacy averaging supplies the nonnegative integral Stanley–Stembridge-character decomposition. Therefore C005 is proved; the complete symbolic argument passed fresh independent internal audit. The theorem subsumes T004, T051 and T052 and includes tails with arbitrarily many parts equal to two. It does not prove positivity for coefficient shapes with θ₂≥3, for which prefix constancy itself can fail.

The standard ingredients are Bernstein creation, dual Cauchy, Littlewood–Richardson multiplication, tableau standardization, and finite-poset Möbius inversion. The new argument in this continuation is the path factorization combined with the spare-large-entry geometry. Novelty remains unresolved pending comparison and external review; no trained model is credited with this symbolic proof.
