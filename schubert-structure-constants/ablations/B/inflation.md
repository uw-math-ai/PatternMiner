# An inflation construction for Schubert zeros

For top-degree permutations $a,b,c\in S_n$, write $F_n(a,b,c)=c_{a,b}^{w_0c}$. The rank-seven triple

$$
(A,B,C)=(1436527,2164375,4321765)
$$

has $F_7(A,B,C)=0$. It satisfies the pairwise Bruhat and interval inversion inequalities. At each simple position, at least two of the three permutations have a descent, so there is no descent-cycling move. The coefficient and interval conditions can be checked directly. [Rank-seven examples](projections.tsv)

For $p\in S_r$ and $x\in S_q$, let $p[x]\in S_{rq}$ replace each entry of $p$ by a block of $q$ consecutive values in the relative order $x$. Starting with $(A_1,B_1,C_1)=(A,B,C)$, define

$$
(A_{k+1},B_{k+1},C_{k+1})=(A[A_k],B[B_k],C[C_k]).
$$

For every $k\geq1$, this is a zero in rank $7^k$ with the same Bruhat, interval and descent properties.

**Proof.** The length formula $\ell(p[x])=q^2\ell(p)+r\ell(x)$ shows that inflating one top-degree triple by another gives a top-degree triple. Project the full flag variety to the partial flag variety that retains the dimensions $q,2q,\ldots,(r-1)q$. In each of the $r$ blocks the three restricted indices are $x,y,z$. Their lengths sum to the dimension of the corresponding flag variety, so the parabolic factorization of the top intersection has a factor $F_q(x,y,z)$ for each block. Thus a zero for the inner triple gives a zero after inflation.

At an internal position the descents are those of the inner triple. At a block boundary the comparison is determined by the outer triple. Both therefore preserve the condition on the number of descents.

Block inflation by the identity preserves Bruhat comparisons. Indeed, replacing each adjacent swap in a reduced word for the outer permutation by the $q^2$ adjacent swaps that interchange two blocks gives a reduced word. Apply the subword criterion. Taking a direct sum of copies of the inner permutation also preserves Bruhat comparisons. These two operations compose length-additively to give $p[x]$. Since complementing all values complements both the outer and the inner permutation, the three pairwise Bruhat inequalities are preserved.

For an interval meeting several blocks, let $t_i$ be the number of its positions in block $i$. The interior blocks have $t_i=q$. If $m_{ij}$ is the number of inversions across positions $i<j$ in the three outer permutations, the contribution from different blocks to the interval inversion sum minus the number of pairs is

$$
\sum_{i<j}(m_{ij}-1)t_i t_j.
$$

This expression is affine in each of the two endpoint sizes. At each corner, where both endpoint sizes belong to $\{0,q\}$, it is $q^2$ times the inversion excess on a contiguous interval of the outer triple, hence is nonnegative. It follows that the contribution is nonnegative for all endpoint sizes. The contributions within individual blocks are nonnegative by the interval inequalities for the inner triple. This proves the induction.
