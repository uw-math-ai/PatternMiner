# A limitation of all Levi necessary length inequalities

Use the finite and stable conventions of the manuscript. The examples below show that the necessary length inequalities for **every** Levi subset do not characterize positivity of a candidate coefficient. They do not rule out stronger criteria involving Levi structure constants. The deductions use the frozen general theorems and inherit their need for independent mathematical review; no external priority claim is made.

## An example in S6

Set

\[
a=152364,\qquad b=431256,\qquad x=461253.
\]

Their lengths are \(4,5,8\). The retained calculations and the Levi reduction below give

\[
q^x_{a,b}=\alpha_1+2\alpha_2+2\alpha_3+2\alpha_4+\alpha_5,
\qquad c^x_{s_4a,b}=1,
\qquad c^x_{a,s_3b}=0.
\]

Thus a positive supplied coefficient is available, but candidate \(3\in D_L(b)\) has zero coefficient. Nevertheless it satisfies every Levi necessary length inequality.

Indeed, the simple-root support of \(q\) is the full Dynkin diagram of type \(A_5\). A proper Levi subset omits part of this support, so its parabolic order is zero and its length restriction cannot exclude the candidate, by [LEVI_RESTRICTIONS.md](LEVI_RESTRICTIONS.md). For the full Levi group the restriction is simply

\[
\ell(a)+\ell(b)-1=\ell(x).
\]

Appending fixed points gives the same obstruction in every \(S_N\), \(N\ge6\). If \(J\) contains \(\{1,\ldots,5\}\), then \(a,b,x\in W_J\) and \(z_J=e\), so the same equality holds. If \(J\) omits a root from this support, the preceding zero-order argument applies.

## Derivation from retained coefficients

The [retained matched pair](LEVI_COUNTEREXAMPLE_COMPANION.json) has common factors

\[
u=(1,7,4,2,5,3,9,8,6,10),\qquad
v=(6,1,7,5,2,3,8,4,9,10),
\]

and outputs

\[
w=(6,8,9,3,2,4,7,1,5,10),\qquad
w'=(6,8,10,2,3,4,7,1,5,9).
\]

The lengths of the factors are 11 and 13, and both output lengths are 23. With supplied index 6, the ordinary coefficients are

| Output | \(A_6\) | \(B_4\) | \(B_5\) |
|---|---:|---:|---:|
| \(w\) | 1 | 0 | 2 |
| \(w'\) | 1 | 1 | 0 |

The only candidate indices are 4 and 5. Their recorded local data agree: the relevant inverse positions at values \(4,5,6,7\), the supplied index, the descents and the lengths coincide. Both candidates satisfy every Levi necessary length inequality for both outputs. Hence even supplementing these local data with the truth values of all those inequalities does not determine a correct witness. This assertion concerns the truth values of the inequalities, not the complete projected permutations or their lengths.

The current model has additional information. For example, \((wv^{-1})(\alpha_4)=\alpha_1+\alpha_2\), whereas \((w'v^{-1})(\alpha_4)=\alpha_1\). Thus this pair does not establish insufficiency of the model's complete augmented candidate features.

[The first coefficient computation](ALL_LEVI_COUNTEREXAMPLE.json) gives

\[
q^w_{u,v}=\alpha_1+2\alpha_2+3\alpha_3+3\alpha_4
+4\alpha_5+3\alpha_6+2\alpha_7+\alpha_8.
\]

The [companion calculation](LEVI_COMPANION_VERIFICATION.json) deduces

\[
q^{w'}_{u,v}=\alpha_3+2\alpha_4+2\alpha_5+2\alpha_6+\alpha_7.
\]

For completeness, this deduction uses the first-order Cauchy identity retained in ROOT_SUPERVISION_PLAN.md. Write \(D_i=c^{s_iw'}_{u,v}\) at a left ascent of \(w'\), and zero otherwise, including the additional stable index 10. The common cover \(s_2w'=s_9w\) gives \(D_2=1\). The retained ordinary coefficients give \(\sum_i i(A_i+B_i)=10\), so the boundary identity requires \(\sum_i iD_i=10\). The possible nonzero indices of \(D\) are \(2,3,4,6,8,10\). With \(D_2=1\), only two nonnegative integral possibilities remain: \(D_4=2\), or \(D_8=1\), with the other entries zero. The former gives a negative simple-root coefficient and is impossible. The latter yields the displayed polynomial. This deduction makes no additional product query.

Now take \(J=\{3,4,5,6,7\}\) and decompose \(w'=x_Jz_J\). Under the identification of \(W_J\) with \(S_6\), the projections of \(u,v\) are \(a,b\), and \(x_J=x\). Their degree difference is one. Since the global coefficient has parabolic order one, the minimum-degree nonzero Levi output below \(x\) must be \(x\) itself. The full Levi divisibility theorem therefore applies. The global polynomial is primitive and uses only roots in \(J\), so the outside integer factor is one. This proves the stated polynomial for \(q^x_{a,b}\).

For the supplied coefficient, downward Demazure commutation gives \(\pi_{z_J}(s_6u)=s_4a\). The global ordinary coefficient is one, and the local degree at \(x\) is zero. The same parabolic order and full Levi divisibility arguments imply \(c^x_{s_4a,b}=1\).

Finally \(3\notin D_L(a)\) and \(3\in D_L(x)\). In the Cauchy identity at index 3, both the left-factor term and the cover term vanish. Therefore

\[
c^x_{a,s_3b}=2Q_3-Q_2-Q_4=4-2-2=0.
\]

This proves the claimed obstruction. It is a deduction from the retained exact data and the written general theorems, rather than a fresh independent ordinary-product evaluation in \(S_6\).
