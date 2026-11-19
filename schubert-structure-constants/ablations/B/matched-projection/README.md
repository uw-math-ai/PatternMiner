# Matched Schubert projection search

Two agents received the same Schubert projection problem, data, tools, 60-minute budget and allowance of 50,000 coefficient or projection queries. The prompts differed by the final sentence instructing one agent to train and use a small model. Each condition used one GPT-6.1 Sol session at extra-high reasoning. [Prompts](prompts/) · [Shared instructions and input digests](protocol.json) · [Supplied data and oracle](inputs/)

For a homogeneous zero, we take the smallest number of flag dimensions in the union of the factor descents for which the projected Schubert incidence conditions have empty generic intersection. Appendix C restricts to triples for which the descents of the target lie in this union. Both agents reached four, five and six retained dimensions in ranks seven, eight and nine.

| First certificate in the query log | Rank 7, depth 4 | Rank 8, depth 5 | Rank 9, depth 6 |
|---|---:|---:|---:|
| [Without the training instruction](optional/questions.tsv) | 438 | 7,112 | 18,591 |
| [Full workflow](full/questions.tsv) | 78 | 784 | 2,953 |
| [Additional projection-problem-only control](raw/questions.tsv) | 1,578 | 7,189 | 13,093 |

Each row of a log is one distinct query, in the order charged to the session. Counting starts at one and includes earlier queries in all ranks. A zero coefficient or an empty projection establishes vanishing. To certify depth at least d, every subset of d−1 factor descents must lie in a queried nonempty projection of the same triple. We chose this secondary measure after seeing interim progress and before computing the counts. The original score evaluated the examples nominated at the end of each session. The first certificates could occur earlier and need not have been nominated.

Run `python3 verify.py` from the ablations directory with Python 3 and lrcalc. It replays all three logs and recomputes the coefficient and minimum number of retained dimensions for each first witness. [Scoring code](../milestones.py) · [First witnesses](first-certificates.json)
