# Pure beta221 at cubic tail ten: exact reduction and a rank-four subfamily

2026-09-20 local. NEW bounded proof-only working note, requiring separate
review. Root supplied the first-excess endpoint to investigate; the
degree14-host-audit agent derived the alternative full reduction and the
GL4 complement argument below. Root was informed of the proposed rank-four
subfamily, not asked to regard it as a completed whole review.

No scientific module, tableau/LR generator, coefficient extractor or
experimental runner was executed. No existing source, input, proof,
result or ledger was changed. This note is outside E070 and any planned
E071 window. It is not a full beta221 cancellation or a full f5/tail-ten
assembly, and no novelty or external verification is claimed.

I authored the earlier full h3[s21] cancellation used below; it has the
separately pinned root whole review. The mixed s21[s21] construction was
authored by the paper agent from root's proposed algebra/map and has my
separate whole review. These roles are inherited dependencies, not an
independent review of this new note.

## 1. Exact complete branching and the full signed degree-ten formula

Put beta=(2,2,1), S=s21, w=s1 and H=sum h_k[W]. One-letter branching
has gamma1=2, gamma2 in{1,2} and gamma3 in{0,1}. Consequently

```text
s221[1+W]=s21+(s22+s211)+s221,
```

with respective degrees3,4,5. The absent degree-four first-row corner is
31, since w*s21=31+22+211. Direct-sum grading of the cubic and division
by H give, without any ordinary projection,

```text
T221=[degree10] h3[s221[1+W]]/H
    =(s22+s211)*h2[S]-w*h3[S].                    (1)
```

Indeed degree nine contributes h3[S]; degree ten uses two degree-three
factors and one degree-four factor, giving h2[S]*(s22+s211), with no
multinomial factor. Higher branching terms cannot contribute. The degree-
one inverse-H term is -w. Thus (1) is also a direct derivation, independent
of extrapolating any later excess formula.

Outer Pieri says s21[S]=S*h2[S]-h3[S]. Therefore the reviewed saturated
first-excess formula supplied by root is equivalently

```text
T221=w*s21[s21]-s31*h2[s21].                     (2)
```

Both (1) and (2) retain every formal Schur term. Neither is asserted
Schur-positive as an unrestricted degree-ten symmetric function.

The original characteristic h3[s221] has degree fifteen. Exact Bernstein
extraction gives

```text
[s_nu] T221=<h3[s221],s_(5,nu)>,       |nu|=10.   (3)
```

Only nu1<=5 is an ordinary actual target. This is30 of42 partitions of
ten; the twelve other written prefixes remain generalized signed/zero
descriptors and must not be passed to an ordinary character oracle.
No value is assigned to those virtual slots in this note.

## 2. Two fully specified finite cancellation problems, not solved here

The reviewed full square has four actual primitive origins:

```text
h2[S]: (42,B), (321,B), (222,B), (3111,C).
```

B retains its unique22/content2 horizontal-strip witness; C retains
the31/content11 vertical strip at3111. These are not anonymous copies.
The full mixed-outer primitive s21[S] is the union of four square-origin
left-gamma/content21 LR families, less its sixteen specific reviewed
image labels. Its direct geometric predicate and origin-sensitive inverse
are available in the pinned full mixed-cube note.

Thus (2) defines complete positive families by a mixed-primitive label at
size nine followed by one addable corner; the negative family is each of
the four square origins multiplied by31 using a fixed ordinary LR order.
One must preserve the sixteen exclusions inside the mixed primitive.

Alternatively (1) has eight positive ordered product specifications:
for each square-origin gamma, take the full left-gamma/content22 family
and the full left-gamma/content211 family, with separate content tags.
Its negative set is each of the sixteen original h3[S] survivor labels
followed by an addable corner. Retain the old cube origin and the corner,
not merely the final shape. Include forced initial LR unit links if using
the earlier full-chain convention. Restrict only the final output nu1<=5
when asking for an actual positive rule.

These are exact finite signed models with no new plethysm primitive or
coefficient-dependent witness range. No complete shape-preserving injection
between either pair of families has been supplied here. In particular,
the rank-four argument below does not silently cancel the remaining
length-five-or-more actual targets or provide a bijection between arbitrary
LR factor trees.

## 3. A different exact route in four variables

Let V have dimension four. For a partition mu with at most four rows,
pad it with zeros and set

```text
comp_k(mu)=(k-mu4,k-mu3,k-mu2,k-mu1),
```

omitting trailing zeros and requiring k>=mu1. The highest weight of
S_mu(V)^* is (-mu4,-mu3,-mu2,-mu1); a det(V)^k twist adds k to
each entry. Hence

```text
det(V)^k tensor S_mu(V)^* = S_comp_k(mu)(V).      (4)
```

This standard highest-weight identity keeps the determinant twist; it
does not set det(V)=1 or identify different polynomial degrees.
At mu=(2,1,0,0) and k=2, comp_2(mu)=(2,2,1,0), so

```text
S221(V)=det(V)^2 tensor S21(V)^*,
Sym^3(S221(V))=det(V)^6 tensor (Sym^3(S21(V)))^*. (5)
```

The determinant exponent is six, not two or three. Symmetric powers
commute with dualization in characteristic zero and pull a one-dimensional
twist to its third power. The old cube has degree nine, so complement
in the6-by4 rectangle has degree24-9=15, exactly the actual ambient
degree, not a transformed tail degree.

Specializing a Schur expansion to four variables kills exactly shapes
with more than four rows. The remaining Schur characters are linearly
independent. Thus this specialization preserves every stable coefficient
of an actual output lambda having at most four rows. It says nothing
about coefficients at five or more rows; their vanished characters do
not imply stable zeros.

## 4. The entire inherited sixteen-origin palette under specialization

The following table retains ALL sixteen old objects, including the two
distinct4221 origins and the four objects that vanish in rank four.
It is a hand application of(4) to the already proved full cube palette,
not an extracted coefficient table.

| Old degree-nine output/origin | Rank-four determinant-complement output | Prefix-five tail, if any |
| --- | --- | --- |
| 63,H | 663 | none |
| 531,H | 6531 | none |
| 522,F | 6441 | none |
| 5211,F | 5541 | 541 |
| 441,G | 6522 | none |
| 432,H | 6432 | none |
| 4311,G | 5532 | 532 |
| 4221,D4 | 5442 | 442 |
| 4221,F | 5442 | 442 |
| 42111,F | vanishes: five rows | not applicable |
| 411111,F | vanishes: six rows | not applicable |
| 333,H | 6333 | none |
| 3321,E4 | 5433 | 433 |
| 33111,G | vanishes: five rows | not applicable |
| 3222,D4 | 4443 | none |
| 32211,D5 | vanishes: five rows | not applicable |

Every retained old shape has first part<=6, so its complement is a
partition. Complementation is involutive on the padded rectangle, so
different old shapes do not become one new shape. The multiplicity at5442
comes exactly from two inherited origins, not indices1,2.

There are exactly five ordinary nu of size ten having at most three
rows:55,541,532,442,433. The table and (5) give their complete palettes:

| nu | Actual lambda=(5,nu) | Complete original labels |
| --- | --- | --- |
| 55 | 555 | empty |
| 541 | 5541 | old5211/F |
| 532 | 5532 | old4311/G |
| 442 | 5442 | old4221/D4 and old4221/F |
| 433 | 5433 | old3321/E4 |

The empty case is certified, not omitted: padding555 to(5,5,5,0), its
unique inverse complement is6111, which is absent from the FULL reviewed
h3[S] palette. All sixteen old objects have been accounted for above.
The five resulting cardinalities are0,1,1,2,1, derived symbolically.
They are not measurements, numerical verification or proposed copy bounds.

The underlying positive labels remain literal tableaux. F is the unique
vertical-three-strip object on411; G is the one on33. Old D4 is the
left321/content21 tableau with sole2 in row four; old E4 is the
left33/content21 tableau with sole2 in row four. At4221 the old D4
and F objects therefore remain genuinely different after the twist.

A mathematical recognizer for this restricted subfamily checks nu is one
of the five ordinary tails, restores lambda=(5,nu), computes comp_6(lambda)
in four padded rows, and validates an original cube survivor with that
output and its actual LR/Pieri payload. It never queries a coefficient,
chooses an empirical copy index or searches a generated family. The
determinant complement identifies representation constituents, not a
claimed cellwise bijection between tableaux of unrelated shapes.

## 5. Precisely unresolved scope

The other25 ordinary tails have at least four rows; their actual
lambda=(5,nu) has at least five rows and is killed by the specialization.
For clarity their complete domain is

```text
5311,5221,52111,511111;
4411,4321,43111,4222,42211,421111,4111111;
3331,3322,33211,331111,32221,322111,3211111,31111111;
22222,222211,2221111,22111111,211111111;
1111111111.
```

No sign, vanishing, multiplicity or full cancellation for those tails is
inferred here. Equations(1)--(3) still describe them exactly, but a
positive object map for the full ordinary family remains unresolved.
Even the five rank-four cases do not provide the requested explicit
injection between the larger signed LR families of Section2; they provide
an alternative positive coefficient model at those five actual shapes.

Any later cyclic-color application must retain the original named beta221
color and its standard-tableau label. This note does not count how often
that color occurs in f5. Actual output degree fifteen must be restored
before any independently justified mixed-sector LR assembly. It is not
valid to insert the transformed size-ten tail as an ambient factor.

All new witnesses discussed here are fixed finite-degree labels. No
enumeration complexity, implemented parser, experimental expansion,
full f5 cube, all-cycle row-sum rule, novelty or unrestricted Stanley
Problem12 conclusion is asserted. The working reductions and the new
rank-four subfamily require a separate whole review before promotion.

## Frozen symbolic inputs

```text
d8820ccba12f22c1c9218ec58ae83a1995a279873e62cf8e518d6b4c4b842b56  research-notes/P12-first-excess-tail-leading-route.md
bd102366e0e87bea2b413a39ad0decd1df56886a4b5e7453c0e03553bf02f925  research-notes/P12-full-mixed-nonhook-three-cube-cancellation-candidate.md
08e8875ac18d941ae76f4c7ce47dea5ac0419c0e551ff124dff243b7475bffb6  experiments/P12-full-mixed-nonhook-three-independent-static-review.md
9a91b66285f371d8eee2729bdd42243aec1d3550a6e506c00e49566713815e00  research-notes/P12-leading-nonhook-three-cubic-LR-cancellation.md
f6cb7ad733286ba9d3bbced13f2851db6cbe09f3b67050177f737ea31933d799  experiments/P12-leading-nonhook-three-cubic-static-review.md
adb765513c09caf665769144e62d2145e678a7d4f3e2389befc860a11ef9978d  proofs/P12-quadratic-transfer-and-positive-base-telescope.md
```
