# Choosing and checking mechanistic interpretability methods

Use this guide when a trained model offers useful evidence about a mathematical question. Start with a hypothesis, select an appropriate method, and preserve the distinction between model behavior and mathematical truth. The [source guide](sources.md) records the editions consulted, including developments through September 2026.

## Select a method by the uncertainty

| Question | Possible method | Check needed for the proposed conclusion |
|---|---|---|
| Is a statistic accessible in the representation? | Direct inspection; a low-capacity probe | Input and random controls, probe capacity, held-out examples |
| Does the model use particular information? | Activation or subspace intervention | Valid comparisons, intervention specificity, output effects |
| Which connection carries the contribution? | Path patching; edge attribution | Direct intervention on decisive paths and retained behavior |
| Why does attention select this position? | Query-key analysis | Positional effects, softmax, and intervention on the selection computation |
| Are distributed features obscuring a pattern? | SAE, transcoder, cross-layer analysis | Reconstruction, downstream behavior, error terms, and feature stability |
| Does a proposed algorithm describe the model? | Causal abstraction or explicit algorithm substitution | Restricted alignment, competing explanations, fresh-family and random-model controls |

Use advanced methods when they resolve an uncertainty left by simpler inspection. A method's recent publication does not make it appropriate for every model.

## Make interventions interpretable

Specify the source and destination examples, the changed mathematical property, and what is matched. Keep examples valid when investigating behavior on the mathematical domain. A deliberately invalid input can diagnose sensitivity, but its effect has a different interpretation.

Record the patched layer, component, positions, direction or subspace, alignment, and downstream recomputation. Check that replacing an activation by itself preserves the output. Compare relevant matched controls. Measure raw effects and uncertainty; explain normalized quantities and small denominators. Restoration and removal tests answer different conditional questions, and redundant computation can obscure a component's contribution. [Patching guide](https://arxiv.org/abs/2404.15255v1), [metrics and methods](https://arxiv.org/abs/2309.16042v2).

For attention, distinguish the query-key computation of scores from the value and output transformations that transport information. The 2025 QK-attribution work extends earlier tracing by decomposing score formation into feature interactions. If selection is central to a mathematical hypothesis, examine its mechanism rather than only the transported value. Account for the actual model's normalization, positional encoding, and nonlinear attention rule. [Attention computation](https://transformer-circuits.pub/2025/attention-qk/index.html).

## Treat circuit explanations as testable claims

Approximate edge attribution can help select interventions. Check important paths directly and evaluate the behavior retained by the proposed circuit on specified examples. High overlap with a reference circuit is not a substitute for such a test. [EAP-IG and faithfulness](https://arxiv.org/abs/2403.17806v2).

Cross-layer transcoders can simplify feature-level tracing, but they introduce a replacement model whose approximation must be evaluated. Keep unexplained error contributions visible. A graph for one input is not automatically a global algorithm. [Circuit tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

Large virtual weights need not have substantial effects on the observed distribution. The 2026 tiny-model study separates effects on outputs from effects on loss. For our purposes, inspect functional effects before assigning a mathematical role to a prominent edge, and distinguish faithfully reproducing the model from improving its answers. [Interference weights](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html).

## Evaluate sparse representations on the question of interest

An SAE reconstructs activations; a transcoder approximates a sublayer's mapping. Check the effect of substituting either into the original computation. In a feature edit, specify whether the original reconstruction residual is retained. Examine the difficult examples that motivated the investigation as well as aggregate errors. [SAE evaluation](https://arxiv.org/abs/2406.04093v1), [transcoders](https://arxiv.org/abs/2406.11944v2).

SAEBench broadens evaluation beyond sparsity and reconstruction, but a 2026 audit reports reliability problems in several metrics and recommends avoiding SCR and TPP at the evaluated settings. Do not use an aggregate benchmark score as sufficient evidence for a mathematical feature. Check evaluation variability and the relevant behavior directly. The audit is a preprint with its own stated scope. [SAEBench](https://arxiv.org/abs/2503.09532v4), [reliability audit](https://arxiv.org/abs/2605.18229v1).

Synthetic controls with known features can test whether an analysis recovers correlations, hierarchies, or distinctions it claims to recover. Such a control diagnoses the method; it does not establish correctness on a real mathematical model. [SynthSAEBench](https://arxiv.org/abs/2602.14687v2).

## Constrain explanations expressed by another model

Causal abstraction makes it possible to compare interventions on a proposed algorithm with interventions on a neural model. Record the mapping and the intervention family explicitly. [Causal abstraction](https://jmlr.org/papers/v26/23-0058.html).

A sufficiently expressive alignment can make this comparison uninformative under the conditions studied by Sutter et al. Restrict and report auxiliary-model capacity; compare with appropriate random models and distinct evaluation families. A learned interpreter may itself learn the task. [Nonlinear representation dilemma](https://arxiv.org/abs/2507.08802v1).

Natural Language Autoencoders offer an additional hypothesis-generation tool for compatible language models. Their authors report confabulation, limited mechanistic grounding, and substantial training cost. Treat a verbalized activation as a proposal to investigate. Applying such a tool to a new small mathematical model requires an explicit compatibility and training argument. [NLAs](https://transformer-circuits.pub/2026/nla/).

## Make the agent's validation explicit

Record the predicted effect of a confirmatory experiment before execution, then check the actual measurements, their units, and the implemented intervention. For exploratory work, record what was selected after inspecting the data. Independent recomputation of a decisive statistic can be more informative than another prose critique.

SAEScientist-Bench reports measurement-interpretation problems in agents selecting SAE features for supplied concepts. AgenticInterpBench studies explanation of already localized circuits and reports weaknesses in validation. These motivate targeted checks of the agent's conclusions; neither study establishes open-ended mathematical conjecture generation. [SAEScientist-Bench](https://arxiv.org/abs/2609.09113v1), [AgenticInterpBench](https://arxiv.org/abs/2606.24026v2).

Conclude with the mathematical statement suggested by the investigation, the evidence about the model, and the unresolved alternatives. A good conjecture can remain worth pursuing when its neural explanation is incomplete.

## Record a failed assessment before adapting

If an independent accuracy assessment fails and its errors inform the next
model or data decision, explicitly reclassify that population as development
data. Preserve its original results and freeze replacement assessment
membership before inspecting new predictions. Keep circuit discovery and
confirmation separate. A later numerical pass on the exposed population
cannot establish independent reliability.

Label-preserving transformations can help diagnose failures. If a source
transformation is selected because its prediction is correct, disclose that
label-dependent selection. Compare a fixed, label-independent prediction
policy separately, and use compatible shuffled sources in activation tests.
A correction under a large activation replacement does not by itself
identify the mathematical computation.

## Optional tools

Use existing hooks for a small custom model when they suffice. For larger analyses, consult current official documentation and verify compatibility before installation or conversion:

- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/) for inspection and interventions on supported transformer implementations.
- [NNsight](https://nnsight.net/getting-started/quickstart/) for tracing and intervention; confirm how the installed version exposes and executes hooks.
- [SAELens](https://github.com/decoderesearch/SAELens) for sparse-autoencoder tooling.
- [circuit-tracer](https://github.com/safety-research/circuit-tracer) for feature-circuit tracing with compatible model and transcoder weights.

These are optional tools, not package dependencies. Verify that any converted model reproduces the original outputs. Inspect memory, activation-storage, and training costs, and confirm any resource expansion before use. Never assume a pretrained dictionary applies to a different model or layer.
