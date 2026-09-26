# Static review: pure beta221 at cubic tail ten

2026-09-20 local. **PASS for the complete signed reductions and the
five-target rank-four positive subfamily. No full beta221 positivity or
cancellation claim is approved by this review.** No correction to the
working note is required.

The degree14-host-audit agent authored the new note and derived its
rank-four complement argument from root's proposed endpoint. I independently
checked its branching, grading, all determinant complements, ordinary
target domain and original primitive payloads. I authored the inherited
mixed-nonhook primitive and first-excess presentation; those dependencies
have separate reviews. This is therefore independent review of the new
note, not independent authorship of every prior ingredient. Only static
reading, hand algebra and source-hash bookkeeping were used; there was no
scientific import, enumeration, coefficient query, experiment, external
call, source edit or ledger operation.

## 1. Complete branching and exact signed models

For beta221, one-letter interlacing forces gamma1=2, permits gamma2=1,2,
and permits gamma3=0,1. Its four outputs are exactly21,22,211,221.
Thus the stated complete branching is correct, with unique minimum
degree three and degree-four layer22+211. The first-row addable31
corner is absent at the saturated first row, not implicitly restored.

The degree-nine part of the complete cubic is h3[s21]; the degree-ten
part is h2[s21]*(s22+s211). A degree-five branching term cannot enter
that latter degree, and the direct-sum rule inserts no factor three.
Only the degree-one part -w of inverse H can lower the degree-nine
contribution to the required signed transform. Consequently

```text
T221=(s22+s211)*h2[s21]-w*h3[s21]
    =w*s21[s21]-s31*h2[s21]
```

is a full degree-ten identity. The second equality uses
w*s21=s31+s22+s211 and outer Pieri
s21[A]=A*h2[A]-h3[A]. No ordinary projection or empirical coefficient
table is used in either step.

The original characteristic h3[s221] has degree fifteen. Exact
Bernstein extraction therefore restores prefix five: the ordinary
targets are `(5,nu)` with |nu|=10 and nu1<=5. The statement correctly
distinguishes these 30 outputs from all 42 formal tails and makes no
ordinary-oracle or positivity claim on the twelve virtual written shapes.

The proposed two finite signed-object models are complete at the level
of definitions. The first has eight positive product specifications
(four full square origins times either22 or211) and the sixteen full
cube survivors followed by an addable corner as negatives. The alternative
uses the full mixed primitive, including its sixteen exclusions, times
w against the four square origins times31. The four square origins and
all sixteen cube origins match the reviewed full palettes. Different
origins remain distinct even when raw tableaux or final shapes coincide.
Neither presentation by itself supplies the missing injection, and the
working note explicitly preserves that gap.

## 2. Determinant duality in exactly four variables

For a four-row padded highest weight mu, dualization gives
(-mu4,-mu3,-mu2,-mu1); tensoring with det^k adds k to each coordinate.
When k>=mu1 the resulting polynomial highest weight is comp_k(mu).
Applying this at mu=(2,1,0,0), k=2 gives precisely221, so

```text
S221(V)=det(V)^2 tensor S21(V)^*,
Sym^3(S221(V))=det(V)^6 tensor Sym^3(S21(V))^*.
```

The cubic multiplies the determinant exponent by three. Complementation
in the 6-by-4 rectangle changes degree nine to 24-9=15, as required.
The argument does not set the determinant to one or identify the ambient
output with its ten-box tail.

Specialization to four variables kills Schur functions with more than
four rows; those with at most four rows remain linearly independent.
Thus it preserves their stable coefficients exactly. There is no
deduction about the stable coefficients of vanished five-or-more-row
outputs. This limitation is essential and stated throughout the note.

## 3. All sixteen origins and the five complete target palettes

I checked the entire inherited sixteen-object palette against its full
origin-labeled construction. Under det^6 duality, the retained old output
shapes have the following complements:

```text
63 -> 663;        531 -> 6531;      522 -> 6441;
5211 -> 5541;     441 -> 6522;      432 -> 6432;
4311 -> 5532;     4221 -> 5442;     333 -> 6333;
3321 -> 5433;     3222 -> 4443.
```

The shape4221 has two distinct old objects, D4 and F, both retained.
The remaining four old objects at42111,411111,33111,32211 vanish
in rank four. Thus all sixteen are accounted for. Every retained shape
has width at most six; the displayed complements are valid partitions.
Complementation is an involution on padded weights, so it creates no
new identification between different old shapes.

The ordinary tails with at most three rows are exactly55,541,532,442,433.
Their actual outputs and full labels are:

| Tail | Actual output | Positive labels |
| --- | --- | --- |
| 55 | 555 | empty |
| 541 | 5541 | old5211/F |
| 532 | 5532 | old4311/G |
| 442 | 5442 | old4221/D4 and old4221/F |
| 433 | 5433 | old3321/E4 |

For the empty case, comp_6(5,5,5,0)=6111 is absent from the full
reviewed h3[s21] palette, so no rank-four constituent is missing from
this zero argument. The nonzero labels have their original geometry:
F is the unique vertical-three strip on411; G is that on33; D4 is
the left321/content21 tableau with its sole2 in row four; E4 is the
left33/content21 tableau with its sole2 in row four. In particular the
two5442 objects retain different real parents, not anonymous copy indices.

The five cardinalities0,1,1,2,1 therefore follow symbolically from the
known full palette and the exact highest-weight identity. They were not
obtained by this reviewer through numerical extraction or enumeration.
Directly validating the old survivor payload after complementing the
actual output is a complete fixed-degree mathematical recognition rule
for precisely these five cases. No cellwise bijection between unrelated
tableau shapes is required or claimed.

## 4. Unresolved scope and conclusion

The listed remaining 25 ordinary tails are exactly those with at least
four rows. Their actual outputs have at least five rows and are lost in
the four-variable specialization. They are not proved zero or positive
by this argument. The signed identities still apply, but a positive
model for those targets, or an injection between either full pair of
signed families, remains unprovided by this note.

Even for the five resolved cases, the determinant route is an alternative
positive coefficient model, not the requested LR deletion map between
the Section2 families. A later f5 construction must additionally retain
each named cyclic beta221 color and its standard tableau, and must restore
degree-fifteen local outputs before mixed-product LR assembly. The note
does not assert the cyclic-color multiplicity, the whole f5 cube or a
full tail-ten result. These boundaries are correct and should be retained.

```text
f061a9c9deafc24e3cdba6347ff9989f50e7697d45cad9a8f8deca159143e058  research-notes/P12-tail-ten-cubic-length-five-beta221-working-route.md
9a91b66285f371d8eee2729bdd42243aec1d3550a6e506c00e49566713815e00  research-notes/P12-leading-nonhook-three-cubic-LR-cancellation.md
f6cb7ad733286ba9d3bbced13f2851db6cbe09f3b67050177f737ea31933d799  experiments/P12-leading-nonhook-three-cubic-static-review.md
bd102366e0e87bea2b413a39ad0decd1df56886a4b5e7453c0e03553bf02f925  research-notes/P12-full-mixed-nonhook-three-cube-cancellation-candidate.md
08e8875ac18d941ae76f4c7ce47dea5ac0419c0e551ff124dff243b7475bffb6  experiments/P12-full-mixed-nonhook-three-independent-static-review.md
d8820ccba12f22c1c9218ec58ae83a1995a279873e62cf8e518d6b4c4b842b56  research-notes/P12-first-excess-tail-leading-route.md
```

**PASS for this bounded working reduction and five-target subfamily only.**
No source or experiment scope is expanded. This is internal static review,
not external peer review, formal verification, novelty certification or
a solution of the remaining ordinary beta221 cases or Stanley Problem12.
