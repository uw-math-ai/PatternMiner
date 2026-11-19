# Experimental details

We use one-line notation for permutations. A homogeneous triple satisfies $\ell(u)+\ell(v)=\ell(w)$, where $\ell$ counts inversions. Its Schubert structure constant $c_{u,v}^w$ is the top intersection number for $(a,b,c)=(u,v,w_0w)$, where $(w_0w)(i)=n+1-w(i)$. A hard negative in A is a triple with zero structure constant that satisfies all the necessary conditions specified below.

Each new session had a 60-minute limit, one numerical-library thread and up to 1 GiB of generated files. Each matched comparison used one fresh session per condition. The five main investigations by PatternMiner used Claude Fable 5 and included human guidance. The eight sampling blocks in A share a single pair of LLM sessions. [Original prompts](prompts/) · [Common coefficient data](inputs/)

## A. Hard-negative mining

In the earlier investigation without hard-negative mining during training, six rank-seven zeros survived descent cycling. The coefficient column in [the search data](A/no-mining-search.csv) identifies the six zeros. The matched sampling comparison below used a separate set of triples.

For each pair of distinct factors $p,q$ in the top triple, we imposed $p\leq w_0q$ in Bruhat order. For each contiguous interval $I$ of positions, we also imposed

$$
\operatorname{inv}_I(a)+\operatorname{inv}_I(b)+\operatorname{inv}_I(c)\geq \binom{|I|}{2}.
$$

Here $\operatorname{inv}_I(p)$ counts inversions with both positions in $I$. In each of eight rank-seven sampling blocks, we drew two uniform factors, discarded infeasible total lengths, and drew a uniform target of the resulting length. We selected the first 256 ordinary triples. From the first 1,024 triples satisfying the inequalities, we selected 256 uniformly and also selected the 256 with the highest positivity scores from a fixed logistic model. We chose all triples before computing their labels. The blocks used seeds 71016100 through 71016107 and examined 483,754 distinct unlabeled proposals in total. Permuting the three factors and applying the simultaneous involution $p\mapsto w_0pw_0$ identified equivalent triples. Different policies could select the same triple within a block. Each selection counted against that policy's budget. [Selected triples and coefficients](A/acquisition.csv) · [Positivity model](A/positivity-model.npz) · [Feature order](A/features.json)

Each agent received 4,096 common training examples and 1,024 common validation examples in ranks five and six, its 2,048 additional rank-seven labels, and a class-balanced depth-six decision tree trained on its data. We withheld 384 ordinary triples and 384 triples satisfying the inequalities in each of ranks seven and eight. Their symmetry orbits were disjoint from the supplied data. Both agents used descent cycling. The agent given mined triples also derived an inequality for the prefix ranks $R_p(k,t)=\#\{i\leq k:p(i)\leq t\}$. For $q\in\{1,2\}$ and thresholds $A+B+C=qn$, positivity implies $R_a(k,A)+R_b(k,B)+R_c(k,C)\geq qk$. In the held-out example $c_{13726548,53274186}^{65748132}$, the ranks at $k=3$ and thresholds $(6,1,1)$ sum to $2$. This certifies the additional zero. [Ordinary rule](A/ordinary.py) · [Rule with mining](A/screen.py) · [Rank-seven example](A/prefix-example.json)

| Held-out population | Zeros | Certified with ordinary sampling | Certified with mining | Positives incorrectly certified zero by either rule |
| --- | ---: | ---: | ---: | ---: |
| Ordinary triples, ranks seven and eight | 759 | 759 | 759 | 0 of 9 |
| Rank-seven triples satisfying the conditions | 83 | 83 | 83 | 0 of 301 |
| Rank-eight triples satisfying the conditions | 107 | 105 | 106 | 0 of 277 |

The fitted trees and the agents' mathematical rules gave different results. On the 768 held-out triples satisfying the conditions, the tree trained with mining identified 4 of 190 zeros and called 9 of 578 positives zero. The tree trained on ordinary data called every one of these triples positive. [Tree fits and scores](component-scores.json)

## B. Matched Schubert projection search

The prompts differed only by the instruction to train and use a small model. Both agents received the same data, tools, 60-minute budget and allowance of 50,000 coefficient or projection queries. Both found zeros satisfying the descent condition in Appendix C for which the smallest number of retained flag dimensions giving an empty generic intersection was four, five and six in ranks seven, eight and nine. The first certificates used 78, 784 and 2,953 queries with the training instruction and 438, 7,112 and 18,591 without it. This secondary measure was chosen after seeing interim progress and before computing the counts. [Query logs, prompts and scoring](B/matched-projection/README.md)

### Earlier comparison with the original Schubert study

We removed the instruction to train a small model and kept the training tools available. The agent chose direct Schubert computations. Its eight rank-seven zeros satisfy the symmetric Bruhat inequalities and all interval inversion inequalities. None admits a descent-cycling move. Starting with the top triple $(1436527,2164375,4321765)$, the agent used permutation inflation to extend the construction to every rank $7^k$ for $k\geq1$. [Construction and proof](B/inflation.md) · [Research instructions](prompts/B_research.md)

For each zero we computed the smallest number of retained flag dimensions for which the projected Schubert incidence conditions have empty generic intersection. Four examples need one dimension and four need two. The paper's zero $c_{1326457,2146375}^{2367145}$ needs the four dimensions $1,2,4,6$. The paper also constructs examples for every $n\geq5$ that require all $n-3$ specified dimensions. This comparison measures the strength of the projection obstruction in the two constructions. [Triples and minimal dimension sets](B/projections.tsv)

## C. Open problem and matched full run

The open-problem prompt was “Seek structural criteria, formulas, and constructions for Schubert structure constants, particularly their vanishing. Propose and prove conjectures about this problem.” Both sessions had the same 7,168 supplied labels, coefficient routine, software, and allowance of 50,000 additional label evaluations. The full run also received the PatternMiner workflow and instructions to train and interpret small models, mine hard negatives, and retrain. The open-problem session used 48,898 additional evaluations and the full session used 11,004. [Open-problem prompt](prompts/C_open_problem.txt) · [Full-run prompt](prompts/C_full.txt)

A Boolean target $w$ has a reduced word in which each simple reflection occurs at most once. The open-problem run expressed $c_{u,v}^w$ as the existence of a nonnegative integral flow on an oriented path forest. Every such coefficient is zero or one. For $w=s_1\cdots s_{n-1}$, the number of ordered pairs $(u,v)$ with nonzero coefficient is the Catalan number $C_n$. The formula agreed with all 5,374 targeted coefficient evaluations. [Proof](C/boolean-target-proof.pdf) · [Formula](C/boolean_formula.py) · [Targeted examples](C/boolean-targets.csv)

The matched full run enumerated all 8,881,334 homogeneous ordered triples in rank six. Bruhat comparisons, interval inversion inequalities, descent cycling and the elementary terminal cases left 31 components unresolved. A coefficient evaluation on one representative from each component gave 26 components with coefficient two, four with coefficient one and one with coefficient zero. The zero component has five vertices up to permutation of the three top factors. One representative is $c_{231645,231645}^{451623}=0$. The open-problem run's general decision procedure left this value unresolved. [All 31 components](C/components.json) · [Five vertices and cycling moves](C/exceptional-component.json) · [Enumeration totals](C/enumeration.json)

## D. Model interpretation

Both agents received the same 320 development pairs from 29 factor orbits of the paper's six-layer Schubert selector. Given $c_{s_i u,v}^{w}>0$, the task was to choose $j$ with $c_{u,s_jv}^{w}>0$. The two members of each pair had disjoint sets of valid answers. Both agents saw the permutations, valid moves and unmodified output summaries. One also saw the effects of replacing each individual attention head's output with the corresponding output on the other member of the pair. Each agent nominated one of 19 sets of four heads and submitted a rule that chose $j$ using the permutations alone. We scored both submissions on the same 249 withheld pairs in 22 separate factor orbits. [Development pairs](D/discovery.jsonl) · [Single-head effects](D/single-head-development.npz) · [Original intervention study](../research/2026-09-16-circuit-paths/REPRODUCE.md)

We measured the change in log-sum-exp over the donor's valid moves minus log-sum-exp over the destination's valid moves when all four heads were replaced together. We averaged the change over directions in each factor orbit and weighted orbits equally. The primary comparison used the 248 pairs from 21 orbits on which the unmodified selector chose valid moves for both inputs.

| Held-out outcome | Outputs alone | With individual-head interventions |
| --- | ---: | ---: |
| Four heads selected (layer, head) | (1,0), (1,8), (3,0), (3,8) | (1,0), (1,8), (3,3), (3,11) |
| Mean change in contrast under joint replacement | 10.46 | 16.18 |
| Rank among the 19 sets | 2 | 1 |
| Inputs with a valid move | 423/498 | 457/498 |
| Pairs with two valid moves | 174/249 | 214/249 |
| Pair success, weighting factor orbits equally | 70.8% | 80.3% |

Both mathematical rules used inequalities from Schubert Newton polytopes. The agent with intervention data also used bounded descent cycling and the number of tight inequalities to choose among surviving moves. The two rules both solved 168 pairs. The intervention arm alone solved 46 and the outputs-only arm alone solved six. The original trained selector solved 248 of 249. [Outputs-only rule](D/outputs.py) · [Rule with interventions](D/internals.py) · [All joint effects](D/joint-patches.csv)

## Checking the results

Run `python3 verify.py` from this directory using Python 3 with the `lrcalc` package. The script recomputes the hard-negative counts, the minimal projection dimensions, the Boolean formula checks, the rank-six component values, and the intervention rankings and held-out scores. Add `--replay-rules` to run all four A and D rules on the saved assessment inputs and compare each prediction with its archived value.
