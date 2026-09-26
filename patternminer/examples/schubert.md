# Schubert structure constants: changes of question and method

This example draws on the research record frozen at [withheld for anonymous review], commit `[withheld for anonymous review]`. It illustrates decisions made during an investigation. It is not a prospective evaluation of these prompts, and the general mathematical arguments were developed through symbolic computation and geometric reasoning rather than established extraction of a neural mechanism.

## Starting from a mathematical problem

Ordinary Schubert structure constants are defined by expanding a product in the Schubert basis,

\[
\sigma_u\sigma_v=\sum_w c_{u,v}^{w}\sigma_w.
\]

Questions about their vanishing, positive formulas, and exact computation are related but distinct. The research program investigated vanishing criteria and examples that escaped specified necessary conditions. It also used equivariant structure constants, with their root conventions stated separately in the mathematical record.

A suitable project brief describes the varieties and coefficient system, permutation conventions, available exact computations, known tests, and the larger question. It does not need to prescribe the final conjecture.

## A symmetry test changed the experimental question

The product is symmetric in its factors. Interchanging them should preserve its structure constants. An audit of the historical predictor found substantial changes under this transformation. The next useful action was to investigate the sampling and representation conventions.

The corrected experiment used one label-independent proposal procedure and a model designed to preserve factor symmetry. It evaluated distinct products, ambient sizes, and flag types separately. An elementary gradient-boosted baseline outperformed the small transformer on all three reported holdouts. That result limited what could be attributed to neural prediction and helped direct the subsequent investigation toward more specific mathematical tasks.

Evidence: symmetry audit and corrected experiment. The latter is a dated historical account; later mathematical results do not retroactively change its experiment.

Workflow lesson: test a known mathematical symmetry before assigning a structural explanation to predictive success. A negative model comparison is useful information about which research method to try next.

## A more specific learning task exposed new examples

A later model selected witnesses relevant to a proposed support relation. This produced a concrete way to examine valid alternatives and failed proposals. Head-output ablations and substitutions of valid input witnesses tested reliance on aspects of the input and learned computation.

The observations were functional diagnostics. They did not establish an extracted mathematical algorithm, and the motivating inequality predated that model. The workflow should record both the useful selected examples and this attribution boundary. An exact zero is difficult only relative to the specified filters it survives.

Evidence: witness experiment and interventions.

Workflow lesson: change the learning task when a particular mathematical operation is more informative than a broad prediction label. Mechanistic experiments can yield useful questions without resolving the full computation.

## A plausible conjecture led to a decisive counterexample

The equivariant investigation developed statements about orders of vanishing and leading forms along parabolic walls. Further calculations suggested an integral positive expansion in specified minimum-degree Levi structure constants. Selected tests supported that proposed strengthening, so it was a meaningful conjecture to investigate.

A subsequent search targeted the arithmetic condition behind the expansion. It found an ordinary global structure constant equal to one whose only minimum-degree local structure constant is two. An integral expansion would require

\[
1=2F,
\]

which is impossible over the integral polynomial ring in question. Independent exact computations and an elementary Monk–Pieri calculation checked the counterexample. Its interpretation distinguishes the index of the generic Levi incidence fiber from degrees obtained from the specified local Schubert conditions.

Evidence: counterexample and argument and verification record.

Workflow lesson: a good conjecture focuses the next experiment even if it is ultimately false. Preserve the conjecture and its evidence, change its truth status, and formulate subsequent questions around the structural distinction revealed by the counterexample.

## What the portable workflow carries over

The common procedure is to choose informative questions, select appropriate models or computations, interpret evidence, formulate precise conjectures, and revise the research direction. The Schubert-specific mathematics belongs in the project brief and its computation tools. Another mathematical project need not have vanishing labels, permutations, or an equivariant coefficient oracle.

The frozen manuscript records the general arguments separately from finite verification and retains the status of external mathematical review and priority assessment. This example supplies provenance for the workflow design, not a causal claim that the workflow or mechanistic interpretability produced those theorems.
