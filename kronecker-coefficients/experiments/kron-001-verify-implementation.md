# Experiment: Implement and verify Murnaghan-Nakayama characters and Kronecker coefficients

Identifier / date: kron-001 / 2026-09-13
Question or conjecture: none yet -- this is an implementation-verification
pilot, prerequisite to any conjecture work on Stanley Problem 10.
Status: proposed

## Question and proposal

Uncertainty addressed: whether an independent, from-scratch implementation
of S_n character values (Murnaghan-Nakayama rule) and the Kronecker
coefficient formula is correct, before it is trusted as ground truth for
any later model or conjecture.

Proposed procedure:
1. Implement Murnaghan-Nakayama recursively: chi^lambda(rho) via removing
   rim hooks of size rho_1, rho_2, ... from lambda, with sign (-1)^(height-1).
2. Implement conjugacy-class enumeration and sizes for S_n (partitions of n,
   class size n! / z_rho with z_rho = prod_i i^{m_i} m_i!).
3. Implement g(lambda,mu,nu) = (1/n!) * sum_rho |C_rho| * chi^lambda(rho) *
   chi^mu(rho) * chi^nu(rho).
4. Verify against known facts, not against each other circularly.

## Expected outcomes and decision

Exploratory/confirmatory mix. Predictions before execution:
- Character-table orthogonality: sum_rho |C_rho| chi^lambda(rho)
  chi^mu(rho) / n! should equal 1 if lambda=mu, else 0, for all lambda,mu
  at a given n. This must hold exactly or the character implementation is
  wrong.
- g((n),(n),(n)) = 1 for every n (trivial representation only appears once
  in itself).
- g(lambda,mu,nu) symmetric under all permutations of (lambda,mu,nu).
- At least one nontrivial hook-shape case should match a value computable
  by the classical Littlewood/Murnaghan hook formula I will state before
  looking up any external number, so the check is not circular.

If any check fails, the implementation is wrong and must be fixed before
proceeding -- this is a correctness gate, not a mathematical result.

## Resources and execution

- Resource agreement: PROJECT.md, small-n exact computation pre-cleared;
  n up to ~10, pure Python, no external package, no internet needed for
  execution (only used earlier for literature identification).
- Commands: pure Python in this container.
- Code/environment: Python 3, container default, no seeds needed (exact
  computation, deterministic).
- Data: none external; all generated in-memory.
- Exactness: fully exact integer arithmetic throughout.

## Results

Ran mn.py in this container. Exact integer/fraction
arithmetic throughout; no external data or package used.

- Check 1 (hook-length dimension vs chi^lambda(1^n)): matched for every
  partition of every n from 1 to 8 (all p(n) partitions checked, p(8)=22,
  so 1+2+3+5+7+11+15+22 = 66 partitions total). No mismatches.
- Check 2 (row orthogonality sum_rho |C_rho| chi^lam(rho) chi^mu(rho) / n!
  = delta_{lam,mu}): matched for every ordered pair (lambda, mu) of
  partitions of n, for n = 1..6. No mismatches. This is a strong
  independent check since it constrains all character values
  simultaneously, not just the identity-class values used in Check 1.
- Check 3 (g((n),(n),(n)) = 1): matched for n = 1..7.
- Check 4 (symmetry of g under permuting its three arguments): matched on
  40 randomly sampled triples of partitions of n=6 (seed 0), each checked
  against all 6 permutations.

All predictions stated before execution were confirmed; no failures to
diagnose. Not yet checked: an explicit match against a published
special-case number (e.g. Blasiak's hook-shape rule) -- deferred rather
than done from memory, to avoid citing a number I have not verified via
search. Resource use: well under a second of CPU, n capped at 8 for
exhaustive partition checks and n=6-7 for the coefficient checks, all
within the pre-cleared small-n allocation.

## Interpretation and next action

The implementation is trustworthy for exact Kronecker coefficients at
small n (validated well beyond just this problem's needs, up to n~8-10
being cheap; larger n limited mainly by the number of partitions/classes
growing, not by any bug found here). This closes the verification gate
from kron-001's proposal.

Next action (per STATE.md): design the actual research question this
ground-truth generator serves -- e.g., which sub-family of (lambda,mu,nu)
triples, or which representation of the inputs, would let a model's
errors or a learned selection expose structure a combinatorial rule would
need. This is a model-design question (models.md / model-design.md), not
yet started. No conjecture is proposed at this stage -- this experiment
was purely an implementation-correctness gate.
