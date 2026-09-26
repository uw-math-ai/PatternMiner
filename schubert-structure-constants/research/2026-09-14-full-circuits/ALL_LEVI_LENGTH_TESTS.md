# Computing every Levi length restriction

The necessary inequalities are those in [LEVI_RESTRICTIONS.md](LEVI_RESTRICTIONS.md). The following computation applies them to every Levi subset in type A, including disconnected subsets. It uses the Bruhat lifting property and permutation rank matrices. No external priority claim is made for these standard operations or their combination.

## Bruhat projection by downward Demazure operators

Write \(w=xz\), with \(x\in W_J\) and \(z\) minimal in its left coset. Let \(d_i^R(h)=\min(h,hs_i)\), and choose a reduced expression \(z=s_{i_1}\cdots s_{i_k}\). For \(p\in W_J\), the product \(pz\) is length additive. Repeated use of the right-handed Bruhat lifting adjunction gives

\[
h\le pz
\quad\Longleftrightarrow\quad
d_{i_1}^R\bigl(d_{i_2}^R(\cdots d_{i_k}^R(h)\cdots)\bigr)\le p.
\]

If \(h\le w_J^0z\), the permutation on the right belongs to \(W_J\). It is therefore \(\pi_z(h)\). One may compute it by sorting \(z\) to the identity with adjacent right descents and applying the corresponding downward operators to \(h\), in the same order. This avoids enumerating \(W_J\).

## Decomposition into consecutive value intervals

For a value interval \(C=[a,b]\), let \(J_C=\{a,\ldots,b-1\}\), and write \(w=x_Cz_C\). Set

\[
a_C=\pi_{z_C}(u),\qquad b_C=\pi_{z_C}(v),\qquad
\Delta_C=\ell(a_C)+\ell(b_C)-\ell(x_C).
\]

Suppose \(u,v\le w\). If \(J\) has components corresponding to disjoint value intervals \(C\), the component of \(\pi_{z_J}(h)\) in \(W_{J_C}\) is \(\pi_{z_C}(h)\), for \(h=u,v\). To see this, use the rank-matrix criterion for Bruhat order. Sorting the values of \(w\) within one interval changes its rank matrix only at thresholds internal to that interval. The inequalities at such thresholds depend only on that component. At all other thresholds they follow from \(h\le w\). The least admissible permutation consequently decomposes componentwise.

It follows that

\[
\ell(\pi_{z_J}(u))+\ell(\pi_{z_J}(v))-\ell(x_J)
=\sum_C\Delta_C.
\]

For a candidate \(i\in D_L(v)\), the length restriction for a subset \(J\) containing \(i\) excludes \(B_i=c^w_{u,s_i v}>0\) precisely when

\[
\sum_C\Delta_C-\mathbf1_{i\in D_L(b_{C_i})}>0,
\]

where \(C_i\) is the unique component interval containing both \(i\) and \(i+1\).

## Maximization over all subsets

Let \(F(t)\) be the largest sum of interval weights \(\Delta_C\) over disjoint nontrivial value intervals in \(\{1,\ldots,t\}\). Empty selections are allowed. Then

\[
F(0)=0,\qquad
F(t)=\max\left(F(t-1),\ \max_{a<t}\{F(a-1)+\Delta_{[a,t]}\}\right).
\]

Define the analogous suffix optimum \(G(t)\). The largest violation over all Levi subsets containing \(i\) is

\[
\Gamma_i=
\max_{a\le i<b}
\left{
F(a-1)+\Delta_{[a,b]}-
\mathbf1_{i\in D_L(b_{[a,b]})}+G(b+1)
\right}.
\]

Thus \(B_i>0\) implies \(\Gamma_i\le0\). All projections and all these maxima can be computed in \(O(n^4)\) elementary comparisons using the preceding downward operators. This replaces both factorial enumeration inside a Levi subgroup and exponential enumeration of Levi subsets.

For the present degree-one task, the parabolic order formula gives \(\sum_C\Delta_C\le1\) for every subset. The full Levi group gives \(\Delta=1\) and retains every left descent of \(v\). Consequently \(\Gamma_i\in\{0,1\}\) for candidate indices. This is a necessary zero test, not a characterization of nonvanishing.

## Checks and limitation

[The implementation check](ALL_LEVI_RESTRICTIONS.json) compares 3,304 projections with direct Bruhat enumeration, using all comparable pairs in \(S_4\) and a fixed sample in \(S_5\), with every Levi subset. On the 124 exposed S10 inputs, the dynamic program agrees with explicit enumeration of every subset and exactly reproduces the earlier tests on consecutive blocks of five through eight values.

All Levi subsets together correct six of the thirty exposed errors, with no valid targets excluded. Twenty-four errors remain. [An explicit coefficient computation](ALL_LEVI_COUNTEREXAMPLE.json) verifies a positive supplied coefficient and a zero candidate coefficient that satisfies every one of these necessary length inequalities. Increasing the collection of Levi subsets alone therefore cannot make this criterion sufficient. This statement concerns these length inequalities; it does not rule out stronger conditions involving Levi structure constants or incidence geometry.
