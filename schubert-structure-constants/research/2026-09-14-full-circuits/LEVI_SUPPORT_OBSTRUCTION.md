# A limitation of proper Levi nonvanishing tests

The degree-one obstruction extends beyond necessary length inequalities. Under the frozen parabolic order theorem, a Levi wall along which the original coefficient has order zero cannot exclude a candidate by the ordinary nonvanishing test described below. The argument uses positivity and the Chevalley formula; it makes no new product query.

## Ordinary support under lowering a factor

Let \(a,b,r\) belong to a finite Weyl group and suppose \(c^r_{a,b}>0\). If \(b'\le b\) in Bruhat order, then there is \(r'\le r\) with \(c^{r'}_{a,b'}>0\).

It suffices to consider a Bruhat cover \(b'\lessdot b\). Choose an integral ample divisor class \(H\) on the flag variety. The Chevalley formula expresses \(H\sigma_{b'}\) as a positive linear combination of the classes indexed by covers of \(b'\), including \(\sigma_b\) with positive coefficient. Positivity of ordinary Schubert structure constants therefore gives

\[
[\sigma_r]\bigl(\sigma_a H\sigma_{b'}\bigr)>0.
\]

By associativity this is the coefficient of \(\sigma_r\) in \(H(\sigma_a\sigma_{b'})\). Another application of the Chevalley formula gives some \(r'\lessdot r\) with \(c^{r'}_{a,b'}>0\). Iterating along a saturated Bruhat chain proves the statement. The same proof applies to a product of flag varieties and hence to any standard Levi subgroup.

## Order-zero walls cannot exclude the candidate

Let \(q=q^w_{u,v}\ne0\) have degree one, let \(j\in D_L(v)\), and let \(J\) contain \(j\). Write \(w=x_Jz_J\) and

\[
a=\pi_{z_J}(u),\qquad b=\pi_{z_J}(v),\qquad
b'=\pi_{z_J}(s_jv).
\]

Suppose \(\nu_J(q)=0\). The frozen parabolic order theorem supplies \(r\le x_J\) with ordinary Levi coefficient \(c^{r,J}_{a,b}>0\). Downward Demazure commutation gives

\[
b'=s_j\star b\le b.
\]

The preceding support statement supplies \(r'\le r\le x_J\) with

\[
c^{r',J}_{a,b'}>0.
\]

Thus the following necessary test for a nonzero ordinary candidate coefficient always passes at an order-zero wall:

\[
\text{There exists }t\le x_J\text{ with }
c^{t,J}_{\pi_{z_J}(u),\pi_{z_J}(s_jv)}>0.
\]

This tests ordinary Levi nonvanishing, including outputs below \(x_J\); it is stronger than merely checking that the projected length sum does not exceed \(\ell(x_J)\). The assertion does not concern every possible inference from the numerical Levi coefficients.

## The explicit degree-one consequence

In degree one, \(\nu_J(q)=0\) whenever \(J\) omits a simple root occurring in \(q\). Consequently, a Levi subset that excludes a candidate by the displayed nonvanishing test must contain the full simple-root support of \(q\), as well as \(j\).

For the example in [LEVI_LENGTH_OBSTRUCTION.md](LEVI_LENGTH_OBSTRUCTION.md),

\[
a=152364,\quad b=431256,\quad x=461253,
\qquad q^x_{a,b}=\alpha_1+2\alpha_2+2\alpha_3+2\alpha_4+\alpha_5.
\]

The candidate coefficient \(c^x_{a,s_3b}\) is zero and a supplied coefficient at index 4 is positive. Nevertheless, **every proper standard Levi subgroup of \(S_6\) containing \(s_3\) passes the displayed ordinary nonvanishing test**. Every proper subset omits a root of the displayed polynomial, so the result follows from the order-zero argument. The full Levi group would require the original zero coefficient and is not covered by this assertion.

This is a deduction from the retained example, the frozen parabolic order and Demazure commutation theorems, and the ordinary Chevalley formula. No new independent evaluation of the example is claimed. It strengthens the limitation of proper Levi tests without ruling out criteria that combine other data or use the actual multiplicities in a different way.
