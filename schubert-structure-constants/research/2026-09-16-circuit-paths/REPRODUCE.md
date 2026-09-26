# Reproduction

Use the existing `schubml` environment, one PyTorch thread, and the retained T4 checkpoint. All scripts use `circuit_resources.py`, whose ledger continues the prior cumulative allocation. An experiment name cannot be reused. Reproduction requires a separate output directory and an authorized resource ledger; do not delete this frozen run's records to rerun it.

`circuit_context.py` reads model and mathematical helpers from the earlier experiment directories. It does not redirect their output paths or accounting. The checkpoint is `../2026-09-14-full-circuits/artifacts/SECOND_REFINEMENT_READOUTS.pt`. Circuit populations and exact coefficients are the retained `S9_discovery`, `S10_discovery`, `S9_confirmation`, and `S10_confirmation` compact archives under `../2026-09-14-reliable-circuits/cache/data`. The frozen restoration shards in that experiment provide their portable source. No Schubert products are recomputed.

Execution order:

1. `check_interfaces.py --checkpoint CHECKPOINT --name INTERFACE_CHECKS`
2. `circuit_scan.py --checkpoint CHECKPOINT --assessment ../2026-09-14-full-circuits/SECOND_ASSESSMENT.json --name DISCOVERY_SCAN --part discovery`
3. `circuit_confirm.py --checkpoint CHECKPOINT --assessment ../2026-09-14-full-circuits/SECOND_ASSESSMENT.json --discovery DISCOVERY_SCAN.json --name HEAD_CONFIRMATION`
4. `circuit_path.py --checkpoint CHECKPOINT --assessment ../2026-09-14-full-circuits/SECOND_ASSESSMENT.json --name MLP_CONFIRMATION`
5. `trace_paths.py` and `endpoint_confirmation.py`
6. `head_coalitions.py --part discovery`, then `head_coalitions.py --part confirmation`
7. `trace_all_heads.py` and `trace_matched_features.py`
8. `second_hop.py`
9. `check_prior_evidence.py`, `summarize_results.py`, and `freeze_evidence.py`

Run from this directory for the relative arguments above. Set `CHECKPOINT` to the path given above. The recorded shell logs include elapsed charges, completed sample counts, and verification results. Source hashes in each JSON report identify the executed implementation. The three path-tracing sources are retained separately so that the implementation of an earlier executed experiment is not rewritten by an adaptive follow-up.

For each directed pair, the score contrast is the log-sum-exp over indices with positive donor coefficients minus the log-sum-exp over indices with positive destination coefficients. These supports are disjoint. Positive contrast change means movement toward the donor support. Summary effects give equal weight to each factor orbit. Accuracy requires validity of both selected indices in a pair; it is distinct from agreement with a particular model choice.

In the all-head tracing reports, the internal condition name `kv1_selected_32` denotes the explicitly selected set of all 16 L1 heads. The plan records this set; it should not be confused with the four heads selected in the separate head-confirmation study. Hidden-unit counts refer to pairs `(k,k+384)` inherited from widening. Head groups are `(h,h+8)`; no exact equality of their parameters is assumed.
