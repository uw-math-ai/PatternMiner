# Left descents and the Bruhat projection to a Levi subgroup

Use the frozen manuscript's conventions. Write w=xz with x in W_J and z minimal in the left coset W_J w. For h below w_J^0 z, let pi_z(h) be the least p in W_J such that h is below pz in Bruhat order.

Put d_i(h)=min(h,s_i h) and D_i(h)=max(h,s_i h). The Bruhat lifting property gives the adjunction

\[
d_i(h)\leq y\quad\Longleftrightarrow\quad h\leq D_i(y).
\]

For i in J, minimality of z implies D_i(pz)=D_i(p)z. Consequently, for every p in W_J,

\[
\begin{aligned}
\pi_z(d_i h)\leq p
&\Longleftrightarrow d_i h\leq pz\\
&\Longleftrightarrow h\leq D_i(p)z\\
&\Longleftrightarrow \pi_z(h)\leq D_i(p)\\
&\Longleftrightarrow d_i\pi_z(h)\leq p.
\end{aligned}
\]

Thus pi_z commutes with the downward Demazure operator d_i for i in J. This is a consequence of the lifting property, not a new priority claim.

Now suppose u,v are below w, their length sum is ell(w)+1, and i is a left descent of v. Put a=pi_z(u), b=pi_z(v), and B_i=c^w_{u,s_i v}. The parabolic order formula in the frozen proof implies

\[
B_i>0\quad\Longrightarrow\quad
\ell(a)+\ell(\pi_z(s_i v))\leq\ell(x).
\]

Indeed, an ordinary nonzero coefficient has parabolic order zero. A local nonzero coefficient attaining that order has output r below x and length ell(a)+ell(pi_z(s_i v)). For i in J, the preceding identity therefore gives the necessary condition

\[
B_i>0\quad\Longrightarrow\quad
\ell(a)+\ell(b)-\mathbf1_{i\in D_L(b)}\leq\ell(x).
\]

This is only a necessary condition. It uses no ordinary-product evaluation for the input triple. In type A, pi_z can be computed by comparing permutation rank matrices within W_J. The implementation tests consecutive blocks of two, three and four values, corresponding to Levi subgroups of semisimple ranks one, two and three. It records the two projected lengths, ell(x), their difference, projected left descents, the resulting degree excess, and block validity.

These quantities are explicitly supplied mathematical information. Any benefit from them must be separated from the network's learned contribution. The connection to the earlier inverse-position interventions is a hypothesis to test; those interventions did not establish that the network computes pi_z.

## When a length restriction can detect a zero

Write the nonzero degree-one coefficient as q=sum_k Q_k alpha_k, and put S={k:Q_k>0}. If S is not contained in J, then nu_J(q)=0. The parabolic order formula supplies a local nonzero coefficient of degree zero with output r below x. Therefore ell(a)+ell(b)=ell(r) is at most ell(x). Subtracting the projected descent indicator cannot violate the necessary inequality above.

Thus a Levi length restriction at i can exclude B_i only when J contains both i and the full simple-root support S. This limitation follows from the theorem; it is not a statement about how well a particular network was trained.

ERROR_ROOT_SUPPORT.json recomputes the coefficient on the 16 retained error factor pairs, respecting both factor orientations. In 50 of the 51 incorrect supplied inputs, the smallest interval containing S has semisimple rank at least four. The tested Levi blocks cannot detect those errors by their length restrictions. For the remaining input, S={6,7}, whereas the incorrectly selected candidate is 2. Every consecutive block containing both S and that candidate has semisimple rank at least six. Thus all 51 errors are covered by the blindness criterion for consecutive blocks of rank at most three. LEVI_CANDIDATE_SCOPE.json records this final check and the index conventions. All 51 errors in fact survive the numerical tests. These diagnostic exact rows are excluded from training.

## Interval support in degree one

**Corollary of the primitive leading-form theorem.** In finite type A, the simple-root support of a nonzero degree-one equivariant Schubert coefficient is an interval.

**Proof.** Let J=S be that support. Since q is homogeneous of degree one, nu_J(q)=1 and in_J(q)=q. The parabolic order theorem gives an output r below x whose nonzero Levi coefficient has degree one. Decompose J into its connected components. The Levi flag variety and its structure constants factor over those components. The degrees of the nonzero component coefficients are nonnegative and sum to one. Exactly one component coefficient has degree one; all the others are positive integers.

The primitive part P of this local coefficient consequently uses variables from only one connected component of J. By the primitive leading-form theorem, q=P F, with F a nonzero polynomial in the simple roots outside J and with nonnegative integer coefficients. Homogeneity forces F to be a positive integer. Hence the support of q lies in that one component. Since J was the full support, J is connected. In type A this means that J is an interval. This proves the assertion.

The argument uses the written general theorems in the frozen project, rather than inferring the statement from finite samples. It makes no claim of external priority. ROOT_SUPPORT_INTERVAL_CHECK.json finds no disconnected support among 5,502,579 retained exact coefficients. Among these same entries, 3,677,446 are not scalar multiples of a positive root and 505,001 have a nonunimodal coefficient sequence. Representative exact triples are retained in that report. Those counts describe this training corpus, not a distribution over all Schubert coefficients.
