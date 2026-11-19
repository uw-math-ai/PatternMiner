# Schubert ablations

Each matched comparison used one Codex CLI session per condition with GPT-6.1 Sol at extra-high reasoning. Appendix B.1 reports the hard-negative-mining and model-interpretation ablations. Appendix C reports the matched Schubert projection search.

| Comparison | Result |
|---|---|
| A. Hard-negative mining (B.1.5) | In 2,048 coefficient evaluations, ordinary sampling found 6 hard negatives and sampling among triples satisfying the current necessary conditions found 432. The resulting rules certified 188 and 189 of 190 held-out hard negatives. [Sampling counts](A/sampling.json) |
| B. Instruction to train a small model (Appendix C) | Both agents found zeros requiring four, five and six retained flag dimensions in ranks seven, eight and nine. The first certificates used 438, 7,112 and 18,591 queries without the training instruction and 78, 784 and 2,953 with it. [Query logs and certificates](B/matched-projection/README.md) |
| C. Open problem given directly to the LLM (supplementary comparison) | The direct agent gave a zero-or-one formula for Boolean targets. The matched full run classified all 8,881,334 homogeneous ordered triples in rank six and found a single remaining component of zeros under descent cycling. [Boolean-target proof](C/boolean-target-proof.tex) · [Enumeration](C/enumeration.json) |
| D. Model interpretation (B.1.4, Table 8) | The outputs-only agent selected the second-most influential set of four heads and its rule succeeded on 174/249 held-out pairs. The intervention-guided agent selected the most influential set and its rule succeeded on 214/249 pairs. [Held-out predictions](D/confirmation.jsonl) |

The Boolean-target comparison gave the agents the Schubert open problem and a 60-minute budget. The [direct LLM baselines on all five problems](../../direct-llm-baselines/README.md) used 15-minute attempts with GPT-6.2 Astra and GPT-6.2 Sol.

[Experimental details and scoring](METHODS.md) · [Verification script](verify.py) · [Prompts](prompts/)
