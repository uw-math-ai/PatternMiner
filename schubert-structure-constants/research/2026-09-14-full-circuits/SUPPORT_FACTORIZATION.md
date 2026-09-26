# Factorization over the connected components of the root support

This is a corollary of the primitive leading-form theorem in the frozen manuscript. It does not require a claim about the internal computation of the neural model, or an external priority claim. The manuscript's general proofs still require independent mathematical review.

Let q=q^w_{u,v} be a nonzero equivariant Schubert structure constant in type A. Let S be the set of simple roots occurring in q, and let C_1,...,C_t be the connected components of S in the Dynkin diagram. Then

\[
q=c\prod_{k=1}^t P_k,
\qquad c\in\mathbb Z_{>0},\qquad
P_k\in\mathbb Z_{\geq0}[\alpha_i:i\in C_k],
\]

where each P_k is primitive and homogeneous, and its simple-root support is exactly C_k. In particular, q is homogeneous separately in the variables of each component. If q has positive degree d, then t is at most d. If d=1, its support is an interval.

**Proof.** Take J=S in the primitive leading-form theorem. Every monomial of q has J-degree d, so its initial form is q itself. Choose a minimum-degree nonzero Levi coefficient q^{r,J}_{a,b}. Its degree is d. The Levi flag variety is a product over the components C_k, and the coefficient is the product of the corresponding nonzero component coefficients q_k. These coefficients have nonnegative integer coefficients and nonnegative degrees d_k, with sum d.

Write q_k=c_k P_k with P_k primitive. Gauss's lemma implies that the product of the P_k is primitive. The leading-form theorem therefore gives

\[
q=\left(\prod_k P_k\right)F,
\qquad F\in\mathbb Z_{\geq0}[\alpha_i:i\notin S].
\]

Both q and the product have degree d. Hence F is a positive integer c. Since the support of q is S and the factors use disjoint sets of variables, the support of P_k is exactly C_k. Every C_k is nonempty, so d_k is positive. Thus t is at most the sum of the d_k, which is d. The degree-one assertion follows.

For q constant, S is empty and the displayed product is the empty product. The statement does not assert that the P_k are irreducible. It also does not give a combinatorial rule for their coefficients or the integer c.

This strengthens the interval-support observation in LEVI_RESTRICTIONS.md to a statement in every degree. The numerical degree-one check is ROOT_SUPPORT_INTERVAL_CHECK.json; no exhaustive higher-degree computation is claimed here.
