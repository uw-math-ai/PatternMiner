# Sources and reading routes

This guide was updated on 13 September 2026. It combines established mathematical case studies with recent primary work on interpretation and its evaluation. Read the sources relevant to the current experiment rather than treating every method as mandatory. The recommendations in this package are adaptations to conjecture generation; most interpretation studies below evaluate language models rather than mathematical discovery.

## Mathematical research with learned models

| Source | Why consult it? |
|---|---|
| Davies et al., [Advancing mathematics by guiding human intuition with AI](https://www.nature.com/articles/s41586-021-04086-x), Nature 2021 | Prediction and attribution informing mathematical questions and subsequent reasoning. Read the method and representation-theory case study. |
| Huang, Jackson, and Lee, [Discovering a Zeta Map Algorithm on Dyck Paths via Mechanistic Interpretability](https://arxiv.org/abs/2605.30482v1), 2026 preprint | A deliberately small model, interpretation experiments, and an explicit combinatorial algorithm. Read Sections 4–7 and the mathematical statement. |
| Kvinge et al., [Can Neural Networks Learn Small Algebraic Worlds?](https://proceedings.mlr.press/v321/kvinge26a.html), PMLR 2026 | Tests for algebraic structure beyond group-operation prediction, including unsuccessful extractions and variation among runs. |
| Charton et al., [PatternBoost](https://arxiv.org/abs/2411.00566v1), 2024 archived preprint | Learned proposals coupled to local search as a source of mathematical examples. |
| Hashemi, Corominas, and Giacchetto, [Can Transformers Do Enumerative Geometry?](https://arxiv.org/abs/2408.14915v3), ICLR 2025 | Structured numerical targets, architecture choices, probes, and interventions in an enumerative problem. |

## Model design and algorithmic tasks

| Source | Why consult it? |
|---|---|
| McLeish et al., [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399v1), 2024 archived edition | Position representation, recurrence, and their evaluation on arithmetic. |
| Cho et al., [Position Coupling](https://arxiv.org/abs/2405.20671v1), 2024 archived edition | Encoding positional relationships justified by the task. |
| Nanda et al., [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217v3), ICLR 2023 | Direct weight analysis, a task-appropriate Fourier basis, ablations, and training dynamics. |

## Mechanistic interpretability

| Source | Why consult it? |
|---|---|
| Zhang and Nanda, [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042v2) | How metric and corruption choices affect localization. |
| Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255v1), 2024 | Intervention design, conditional interpretation, redundancy, and metrics. |
| Conmy et al., [Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997v4), NeurIPS 2023 | Automated localization and sensitivity to the experimental setting. |
| Hanna, Pezzelle, and Belinkov, [Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806v2), COLM 2024 | Edge attribution with integrated gradients and behavioral evaluation of circuits. |
| Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093v1), 2024 | SAE training and evaluation beyond one reconstruction statistic. |
| Dunefsky, Chlenski, and Nanda, [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944v2), NeurIPS 2024 | Approximating sublayer computations for feature-level analysis. |
| Ameisen et al., [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), 27 March 2025 | Cross-layer transcoders, attribution graphs, approximation errors, and evaluations. |
| Kamath et al., [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html), 31 July 2025 | Extending tracing to query-key interactions that determine attention scores. |
| Geiger et al., [Causal Abstraction](https://jmlr.org/papers/v26/23-0058.html), JMLR 2025 | A precise framework for comparing computations through interventions. |
| Sutter et al., [The Non-Linear Representation Dilemma](https://arxiv.org/abs/2507.08802v1), 2025 preprint | Why unrestricted alignment expressivity can undermine an algorithmic explanation. |
| [Characterizing interference weights in a tiny language model](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html), 21 August 2026 | Distinguishing weight magnitude, functional effect, and effect on loss. |
| [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/), 7 May 2026 | Verbal descriptions as investigatory leads, with explicit limitations and costs. |

## Evaluation and agent studies

| Source | Why consult it? |
|---|---|
| Karvonen et al., [SAEBench](https://arxiv.org/abs/2503.09532v4), ICML 2025 | Multiple SAE evaluations and the dependence of conclusions on the chosen task. |
| Chanin, [Are Sparse Autoencoder Benchmarks Reliable?](https://arxiv.org/abs/2605.18229v1), May 2026 preprint | Reliability audit; read alongside SAEBench rather than assuming its metrics are definitive. |
| Chanin and Garriga-Alonso, [SynthSAEBench](https://arxiv.org/abs/2602.14687v2), July 2026 revision | Controlled feature recovery under specified synthetic assumptions. |
| Tan et al., [SAEScientist-Bench](https://arxiv.org/abs/2609.09113v1), 8 September 2026, work-in-progress preprint | Agent selection of features for supplied concepts; evidence and validation failures in a restricted setting. |
| Khan et al., [Can Language Model Agents be Helpful Circuit Explainers?](https://arxiv.org/abs/2606.24026v2), 2 September 2026 revision | Explanation of already localized circuits. The paper reports acceptance to Findings of EMNLP 2026. |
| Long et al., [MECA](https://arxiv.org/abs/2607.27709v1), 2026 preprint | Refining mathematical conjectures from broad sources. Its mathematical reasoning principles are distinct from neural mechanistic interpretability. |
| Aggarwal et al., [Discovering mathematical concepts through a multi-agent system](https://arxiv.org/abs/2603.04528v2), 2026 preprint | Interaction among conjecturing, data, and proof attempts in a historical rediscovery task. Multiple agents are not required by this package. |

The studies differ in task, models, intervention semantics, and evaluation. They are not a single ranking of methods. Their results do not establish the effectiveness of this portable workflow or the novelty of any conjecture it produces. The Schubert example supplies a separate record of research decisions, with its own attribution and limitations.

## Software documentation

Consult the installed version and official documentation before adopting a tool. Repository defaults and supported models change.

- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/)
- [NNsight quick start](https://nnsight.net/getting-started/quickstart/)
- [SAELens](https://github.com/decoderesearch/SAELens)
- [circuit-tracer](https://github.com/safety-research/circuit-tracer)

No library is required to use the prompts. New model downloads, remote services, and auxiliary training must fit the confirmed resource agreement.
