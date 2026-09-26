# Circuit analysis, September 16

Use the fixed final checkpoint, including its errors, under the explicit user instruction recorded in PROTOCOL_AMENDMENT.json. No training or coefficient queries are planned.

1. Verify native and intervened computation on development inputs.
2. Scan all 96 heads and six MLP outputs on the fixed round-robin discovery sample of at most 512 pairs. Nominate two pairs of heads inherited from widening, by the sum of their group-mean effects on pairs with two correct native predictions. Controls match the layer and pairing of heads.
3. Evaluate the frozen selection on the preassigned confirmation sample, retaining all model errors. Test queries, keys, values, attention patterns, output replacement, ablation, retention, and shuffled donors.
4. Test the previously fixed first-MLP hidden-unit ranking through second-layer keys and values. The primary comparison remains the top 16 pairs against twenty fixed magnitude-matched random sets; group bootstrap over factor orbits. Do not revise the ranking using confirmation data.
5. Trace successive edges from the first MLP through the second-layer attention and later MLPs. These new multistage experiments are exploratory; report controlled transmission separately from effects obtained when downstream components recompute freely. Freeze their precise intervention plan before running them.

All numerical layers use zero-based indices. Evidence of a circuit contribution does not assert an exact rule for Schubert multiplication. The earlier manuscript and evidence manifests remain frozen.
