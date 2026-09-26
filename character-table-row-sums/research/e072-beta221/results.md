# E072 results: beta221 construction-directed matching pilot

2026-09-21. Execution and exact consistency checks completed. **The learned
rule failed the injectivity gate.** Independent code/result review remains
pending. This is a root-audited exploratory result, not an E071-style
independently reviewed certificate or a new theorem.

## What this contributes to Problem 12

The experiment targets the unresolved cancellation in

```text
T221 = (s22+s211) h2[s21] - s1 h3[s21].
[s_nu] T221 = [s_(5,nu)] h3[s221], |nu|=10, nu1<=5.
```

All 30 ordinary outputs now have a complete executable finite object
record. There are 97 positive objects and 50 negative objects. A separate,
explicitly data-derived lexicographic lookup cancels all 50 and leaves
47 positive objects across 22 nonzero outputs; eight outputs are zero.
The complete negative domain also retains six nonordinary sources, giving
56 in total. Those six are not sent to an ordinary ambient oracle.

All 240 ordered positive LR families agree between the inherited generator
and a separately implemented brute-force permutation/word predicate.
Every ordinary signed count agrees with MN and JT extraction from the
full, untruncated degree-fifteen h3[s221]. MN/JT share power-sum arithmetic
and inherited code; these are cross-checks, not wholly independent software.
The old sixteen cube origins are recovered from the frozen survivor rule,
with original tableaux and square-origin payloads retained.

This certifies finite capacity and a finite lookup, not a uniform geometric
injection, all parameters, full f5, full tail ten, or Problem 12. Counting
capacity was never the same task as finding a natural rule.

## Learned rule and held-out assessment

The model is a 47-feature integer linear candidate scorer. It learns from
collisions of simultaneous independent image choices, not from an arbitrary
matching treated as a uniquely correct label. Three fixed seeds each receive
600 coordinate proposals. Validation selects seed31415. All output-shape
groups are isolated between training, validation and test; the same primitive
origins can occur across them. Prior symbolic exposure to the finite family
is disclosed. There is no size extrapolation or blind mathematical discovery.

An excess collision is number of sources minus number of distinct chosen
images. A three-source collision contributes two, not three pair conflicts.
Every chosen image is locally valid by construction. Injectivity is the
actual behavioral requirement.

| Population | Sources | Nonempty-source output groups | Learned excess collisions | Fully injective groups | Geometry baseline collisions | Lexicographic baseline collisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 22 | 10 | 0 | 10 | 8 | 12 |
| Validation | 11 | 4 | 1 | 3 | 1 | 7 |
| Test | 17 | 7 | 5 | 3 | 6 | 10 |

Nine further ordinary outputs have no negative sources and need no pairing.
The 100 fixed random-weight baselines have test collision min/median/max
2/4/8, versus the learner's5. These are an unselected reference distribution,
not 100 newly selected test models; their computational budgets differ from
training. **No learned advantage over random scoring is established.**

The selected nonzero weights are:

```text
negative_entry_disagreement   +1
corner_entry_2                -2
E_base1_content1              -1
F_base1_content1              +1
G_base0_content1              +1
G_base1_content1              +1
H_base1_content0              +1
```

Base0=42, base1=321; content0=22, content1=211. The disagreement feature
itself is minus the number of disagreeing entries on common skew cells.
The appended corner's proxy label1 is for scoring only, not a claim that
the source's two product steps form a single LR tableau. Ties use the
fixed JSON lexicographic candidate order and are not hidden learned rules.

Removing the disagreement feature raises validation collisions from1 to4,
but lowers test collisions from5 to4. Removing the corner-entry2 feature
raises test collisions to6. These exact ablations diagnose dependence of
this finite scorer, not a general mathematical mechanism or a neural circuit.
No activation-patching claim is made. There was no test-driven retraining.

## Retained counterexamples to the learned map

Validation fails at3331. Test failures occur at5221,433,43111,4222.
There are six total excess collisions and seven distinct colliding pairs.

For example, at output433, the two actual sources are old432/H with
corner(3,3), and old333/H with corner(1,4). The learned rule sends both to
the same gamma222/B, content211 tableau with reading word1123. This is
a literal counterexample to injectivity of the selected rule, not a
counterexample to the target multiplicity or Problem12.

At output4222, the distinct old4221/D4 and old4221/F sources share the
same new corner(4,2), and the learned rule again merges their images.
Origin sensitivity is present among the available features, but this
training run does not learn a successful separation at that test output.
No two source rows have identical complete candidate-feature matrices in
the saved domain, so this audit does not establish an information-theoretic
impossibility for the feature representation or for all linear scorers.

## Integrity, resources and limitations

Command:

```text
python -B scripts/run_e072_beta221.py
```

The sole ledger entry is e072-beta221-matching-pilot-v1, passed execution.
Body wall0.4592714160098694s; charged wall0.465038258000277s;
process peak RSS120160KiB. No GPU, model API, network, installation, Sage/GAP
table or Specht call. GPU was not useful for this tiny exact integer search.
The new artifact is16163 gzip bytes plus81 sidecar bytes. Cumulative
storage99311790bytes, leaving688210bytes; cumulative charged wall time
323.7568247571687s across76 ledger entries. The resource cap was not reset.

All127 pinned paths match after execution and during read-only post-audit.
The 120-file frozen src set is unchanged. The post-audit re-evaluated
selected scores from saved feature matrices, reconstructed collision totals,
and checked lookup coverage/inverses and survivor complements. It did not
repeat coefficient computation, and is not independent reviewer approval.

```text
8c16d0e187f3da83eade2e31f366fb3b624836d621fc8e6de92af7e9f8f9482d  artifacts/e072/result.json.gz
84f447fe7a4a02ecdeec9159c4703de0bb51d84e2a93d3551506a9c284d93c9f  scripts/run_e072_beta221.py
90d5b47241322da5a36cd15160a36c39bc31853c4b76dc59d4e48c0e89cdd2ee  experiments/E072-beta221-matching-pilot.md
```

## Next action

Study the retained H/H and D4/F collisions as geometric obstructions to
the selected priority, not as evidence of insufficient positive capacity.
Try an explicit origin/corner-sensitive local transformation with a direct
image predicate and inverse. Do not replace the problem by generic LR
classification or claim that the finite lookup was learned.

All E072 outcomes are now exposed. Any adaptation must label these outputs
development data; a further pass on them is not fresh confirmation. The
small fixed beta221 domain cannot sustain unlimited independent splits.
A broader genuinely unexposed construction family, or a mathematical proof
of a proposed rule, would be needed for a stronger next claim. Increasing
model size alone is not justified by this pilot.
