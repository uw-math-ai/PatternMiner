# Construction-directed learning: a negative pilot for Problem 12

Paper-candidate text only; independent E072 review and literature novelty
assessment are pending. See the [full report](../research/e072-beta221/results.md).

We tested learning on a concrete cancellation problem arising in a restricted
counting interpretation, rather than on numerical positivity of character-table
row sums. For the beta=(2,2,1) cubic component at tail size ten, exact tableau
enumeration gave97 positive and50 negative ordinary objects. All30 ordinary
signed counts agreed with ambient character calculations. A finite lookup
assignment exists, but does not supply the desired geometric cancellation rule.

We therefore trained a small origin- and corner-sensitive linear priority
model to select images without global collision repair. Output shapes were
grouped into training, validation and test populations. The selected model
had zero training collisions, one validation collision and five test
collisions. Random linear scores had a median of four test collisions;
the experiment did not establish a benefit from learning. Exact failure
witnesses isolate source-origin and corner distinctions that the selected
rule fails to preserve. These results guide subsequent construction search,
but do not supply a new injection or solve Stanley's Problem12.

This is not evidence that mechanistic interpretation generated the existing
mathematical constructions. The model is an inspectable linear scorer with
feature ablations, not a neural circuit recovered by activation patching.
The finite symbolic family was previously studied, so the shape-held-out
model assessment is not a blind mathematical discovery claim.
