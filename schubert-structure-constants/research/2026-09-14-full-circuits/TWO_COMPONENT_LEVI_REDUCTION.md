# Two Levi components suffice for the degree-one length tests

Suppose \(q^w_{u,v}\ne0\) has degree one. The necessary length inequalities used here to exclude an ordinary coefficient \(c^w_{u,s_jv}\) need only be tested on Levi subsets containing \(j\) with at most two connected components.

For a component \(C\) of a Levi subset, write

\[
\delta_C=\ell(\pi_C(u))+\ell(\pi_C(v))-\ell(x_C),
\]

using the Bruhat projection and Levi part of \(w\) in [ALL_LEVI_LENGTH_TESTS.md](ALL_LEVI_LENGTH_TESTS.md). Projections for disconnected components are independent. If \(C_j\) contains \(j\), the tested inequality is

\[
\sum_{C\subset J}\delta_C-
\mathbf 1_{j\in D_L(\pi_{C_j}(v))}\le0.
\]

There is at most one component with positive \(\delta_C\), and its value is one. To see this, take the union \(K\) of all components with positive \(\delta_C\). The frozen parabolic order theorem gives

\[
\sum_{C\subset K}\delta_C
\le \nu_K(q^w_{u,v})\le1.
\]

Here the first inequality follows because every nonzero local output lies below the Levi part of \(w\); the second follows from homogeneity of degree one. Each summand is a positive integer.

If the original inequality fails, keep this unique positive component and the component containing \(j\), identifying them if they coincide. Removing all other components removes only nonpositive summands and leaves the descent indicator unchanged. The resulting inequality still fails. It has at most two components, proving the assertion. In type A these components are disjoint intervals in the Dynkin diagram.

The [retained verification](TWO_COMPONENT_LEVI_REDUCTION.json) compares the unrestricted maxima with maxima over at most two components on the same 124 exposed error and control inputs. This reduction is a consequence of the frozen parabolic order theorem. It neither makes the necessary length inequalities sufficient nor identifies a circuit in the neural model.

The same argument gives a degree bound. If \(q^w_{u,v}\ne0\) has degree \(d\ge1\), at most \(d\) mutually disconnected Levi components can have positive \(\delta_C\). For a left descent \(j\) of \(v\), the necessary degree inequality for \(q^w_{u,s_jv}\ne0\) is

\[
\sum_{C\subset J}\delta_C-
\mathbf 1_{j\in D_L(\pi_{C_j}(v))}\le d-1.
\]

Any violation is therefore witnessed by a Levi subset with at most \(d+1\) components: retain the positive components and the component containing \(j\). This is a reduction of this family of necessary degree inequalities, not a positivity criterion. The numerical check above concerns degree one only.

The degree-one bound cannot be reduced to one component. A [retained example](TWO_COMPONENT_LEVI_SHARPNESS.json) is

\[
\begin{aligned}
u&=(1,2,8,5,3,9,4,10,6,7),\\
v&=(1,2,8,6,5,4,3,7,9,10),\\
w&=(2,8,9,10,4,3,1,5,6,7).
\end{aligned}
\]

The lengths are \(12,11,22\), a supplied coefficient at index 7 is positive, and candidate 4 has coefficient zero. Every connected Levi subset containing 4 satisfies the necessary length inequality. For the disconnected subset \(J=\{4,6,7,8\}\), however, the projected data are

| Component | Projected \(u\) | Projected \(v\) | Levi part of \(w\) | \(\delta_C\) |
|---|---|---|---|---:|
| \(\{4\}\), type \(A_1\) | \(12\) | \(12\) | \(12\) | 0 |
| \(\{6,7,8\}\), type \(A_3\) | \(3412\) | \(1324\) | \(3412\) | 1 |

The projected \(v\) has no descent at candidate 4. The resulting violation is \(0+1-0=1\). The verification retains all 24 connected tests containing 4, each with nonpositive violation, and the integer projections displayed above. It uses the existing exact labels and makes no new product query. This establishes sharpness only of the degree-one component bound; sharpness of the bound in higher degree is not asserted.
