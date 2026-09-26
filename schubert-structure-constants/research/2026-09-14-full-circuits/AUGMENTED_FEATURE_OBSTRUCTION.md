# Candidate data do not determine a witness

The [retained calculation](AUGMENTED_FEATURE_OBSTRUCTION.json) checks eight pairs in four factor orbits for which the complete candidate feature vectors agree, as do all candidate-to-candidate attention relations. Their valid witness sets are disjoint. This strengthens the earlier diagnostic that checked feature vectors alone. It concerns the specified representation, not every possible local invariant.

One example has

\[
\begin{aligned}
u&=(1,3,8,2,6,4,7,5,9,10),\\
v&=(1,6,2,3,7,5,8,4,9,10),\\
w&=(1,10,5,2,6,4,8,3,7,9),\\
w'&=(2,10,5,1,6,4,7,3,8,9).
\end{aligned}
\]

The factor lengths are 9 and 8, and both output lengths are 16. The supplied index can be chosen as \(i=5\) for both outputs. Among the two candidate indices \(\{4,5\}\), the retained exact labels give

\[
\{j:c^w_{u,s_jv}>0\}=\{5\},\qquad
\{j:c^{w'}_{u,s_jv}>0\}=\{4\}.
\]

At each candidate \(j\), the inverse positions of \(j,j+1\), the rank functions at thresholds \(j,j+1\), and all six roots \((ab^{-1})(\alpha_j)\), for distinct \(a,b\in\{u,v,w\}\), agree with the corresponding data for \(w'\). The verification compares these quantities as integers. It also checks exact equality of the model's 148 candidate feature coordinates and 78 relation coordinates for each ordered pair of candidates. The relation coordinates include the coefficients of the six transformed simple roots.

Consequently, any deterministic selection rule depending only on these candidate data must return the same index for the two outputs and must fail on at least one. In particular, a transformer restricted to these candidate positions, with the recorded features and relations, cannot solve both examples, regardless of its parameters or depth. The full model can distinguish them through other positions.

Adjoining fixed points preserves the two witness sets and equality of the listed inverse-position, rank and relative-root data. Thus the same mathematical obstruction occurs in every \(S_N\), \(N\ge10\). The current implementation represents output permutations only through size seventeen.

This gives a precise necessity statement for the small subset used in the circuit analysis. It does not show that the particular first-MLP path is necessary, sufficient, or unique. The labels are inherited from retained exact computations; the present check makes no new product query and is not an independent evaluation of those coefficients.
