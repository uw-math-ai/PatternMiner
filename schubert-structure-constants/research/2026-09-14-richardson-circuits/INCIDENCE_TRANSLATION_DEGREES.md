# Simple-root translations of the incidence image

Work over an algebraically closed field of characteristic zero, with complete flags in k^n. Use the conventions of Lemma `lem:ordinary-image` in the frozen manuscript. Put U equal to the upper unitriangular subgroup, N=dim U, and

\[
Z=Z^w_{u,v}=\{(g,F)\in U\times X_w:F\in X^u,\ g^{-1}F\in X^v\},
\qquad Y=\operatorname{pr}_U(Z)_{\rm red}.
\]

Assume u,v <= w, \(\ell(u)+\ell(v)=\ell(w)+1\), and \(q^w_{u,v}\ne0\). The cited lemma makes Z irreducible of dimension N-1, Y an irreducible hypersurface, and Z -> Y generically finite. Let e be its generic degree. Choose an irreducible defining polynomial f of Y in k[U]. For \(x_i(t)=I+tE_{i,i+1}\), define

\[
\delta_i^L=\deg_t f(x_i(t)g),\qquad
\delta_i^R=\deg_t f(gx_i(t)),
\]

where g is the generic point of U, so degrees are taken in k(U)[t]. Set
\(A_i=c^w_{s_i u,v}\) for i in D_L(u), and \(B_i=c^w_{u,s_i v}\) for i in D_L(v), with zero values outside these descent sets.

**Proposition.** For every simple root i,

\[
A_i=e\delta_i^L,\qquad B_i=e\delta_i^R.
\]

The point requiring justification is that lowering a left descent of an opposite Schubert index gives a birational incidence correspondence. The following rank calculation proves this directly in type A.

**Rank calculation.** Represent a flag H by its first q column spans, and write \(r_v(p,q)=\#\{a\le q:v(a)\le p\}\). The opposite cell C^v has northwest ranks exactly r_v; its closure X^v has northwest ranks at most r_v. Suppose i is a left descent of v. Then \(v^{-1}(i+1)<v^{-1}(i)\).

Left multiplication by x_i(t) adds t times row i+1 to row i. All northwest ranks with p different from i are unchanged. The rank can increase by at most one for p=i. The only places where \(r_{s_i v}(i,q)\) exceeds \(r_v(i,q)\) are

\[
v^{-1}(i+1)\le q<v^{-1}(i),
\]

where it exceeds it by one. For q before this interval, the bound on the first i+1 rows already gives the required unchanged bound. For q after the interval, the bound on the first i-1 rows, plus the one additional row, gives the required unchanged bound. Hence
\(x_i(t)X^v\subseteq X^{s_i v}\).

If H lies in C^v, choose q in the displayed interval and put r=r_v(i,q). The first i-1 and first i rows have rank r, whereas the first i+1 rows have rank r+1. Thus row i lies in the span of the preceding rows, and row i+1 does not. For every nonzero t, adding t times row i+1 raises the first i-row rank to r+1. Consequently

\[
H\in C^v,\ t\ne0\quad\Longrightarrow\quad x_i(t)H\notin X^v.
\]

In particular, a simple-root orbit through C^v meets X^v at only one point on its affine parameter line.

**Birational incidence correspondence.** Let v'=s_i v. On the dense open subset of Z where F and g^{-1}F lie in C_w and g^{-1}F lies in C^v, consider

\[
\Theta:\mathbb A^1\times Z\dashrightarrow Z^w_{u,v'},
\qquad(t,g,F)\longmapsto(gx_i(-t),F).
\]

This map is defined on that open subset and has image in the stated incidence variety by the rank calculation. If two points have the same image, their second flags differ by x_i(t-s); since both second flags lie in C^v, the last assertion of the rank calculation forces t=s. The original g and F then also agree. Thus Theta is injective on this open subset. Source and target are irreducible of the same dimension N by the ordinary-incidence lemma. In characteristic zero, Theta is therefore dominant and birational.

The open subset is nonempty: the open Richardson intersections C_w intersect C^u and C_w intersect C^v are nonempty for u,v <= w, and U acts transitively on C_w. Since Z is irreducible, it is dense. This also justifies using the indicated open cells in the preceding argument.

**Degrees.** The composite to U factors as

\[
\mathbb A^1\times Z\longrightarrow\mathbb A^1\times Y
\xrightarrow{\ \mu\ }U,\qquad \mu(t,g)=gx_i(-t).
\]

The first map has generic degree e. If delta_i^R is positive, the second map is dominant and has generic degree delta_i^R: over a generic h in U its fiber is cut out by
\(f(hx_i(t))=0\).
The change of coordinates \((h,t)\mapsto(hx_i(t),t)\) on U times A^1 preserves irreducibility. Hence this polynomial is irreducible over k(U) when it has positive t-degree, and its generic degree is exactly that degree. Characteristic zero gives separability.

The birational correspondence identifies the resulting degree with the generic degree of \(Z^w_{u,v'}\to U\), which is the ordinary coefficient B_i by the frozen incidence lemma. Thus B_i=e delta_i^R. If delta_i^R=0, the translation preserves the defining equation, the composite is not dominant, and B_i=0.

For a left descent i of u, use instead

\[
(t,g,F)\longmapsto(x_i(t)g,x_i(t)F)
\quad\hbox{in }Z^w_{s_i u,v}.
\]

The same rank calculation applies to F. The variety X_w is U-stable, so the map stays in the required incidence variety. The preceding proof gives A_i=e delta_i^L. If i is not a left descent, X^u or X^v respectively is stable under x_i(t); the corresponding incidence image is invariant and the corresponding translation degree is zero. This proves the proposition.

**Conormal consequence.** At a generic smooth g in Y, let eta be a nonzero conormal covector. Then

\[
B_i>0\ \Longleftrightarrow\ \eta(gE_{i,i+1})\ne0,
\qquad
A_i>0\ \Longleftrightarrow\ \eta(E_{i,i+1}g)\ne0.
\]

Indeed, the quotient of U by the indicated one-dimensional root subgroup has affine-line fibers. Its restriction to Y is dominant exactly when the translation degree is positive. Generic smoothness identifies dominance with injectivity of its differential on T_gY, equivalently with nonvanishing of the displayed conormal scalar. Thus the generic conormal supports agree with the full coefficient supports, not just subsets of them.

This statement is about generic points or rational functions. A zero at one finite-field sample still does not certify a zero coefficient, and bad reduction must also be considered. A screening procedure may use such samples to propose exact queries; its final training labels can still come entirely from exact Schubert products.

This note supplies a derivation within the present project. No external priority or new complexity claim is made. The frozen manuscript has not been edited.
