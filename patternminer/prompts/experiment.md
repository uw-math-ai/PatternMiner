# Design an informative experiment

Read the mathematical question, competing explanations, available tools, and confirmed resource agreement. State the uncertainty an experiment could resolve before choosing its implementation.

Use the [experiment record](../templates/EXPERIMENT.md) to specify the proposed action, examples, computation or model, appropriate comparison, resource cap, and possible outcomes. For an intervention, predict the relevant direction or pattern of change before execution. If the experiment is exploratory, state what will be inspected instead of inventing a retrospective prediction.

Choose the smallest useful investigation. Exact enumeration, symbolic computation, a bounded search, model training, a probe, an intervention, and a mathematical construction are all options. Explain why the chosen action is likely to distinguish the current possibilities. Use existing implementations and records when they suffice.

For example selection, consider model errors, disagreement with elementary rules, uncertainty, extremal behavior, or proximity to an expected boundary. Preserve diversity among families and distinguish genuinely new examples from equivalent presentations. Verify selected labels or witnesses with the available mathematical tools. State which tests define a difficult example. In a vanishing problem, a “hard zero” must refer to a specified battery of criteria and sampling procedure.

When claiming that learned selection improves the investigation, compare with an appropriate random or elementary selection method under comparable resources. Such a comparison is not required merely to record an interesting example. Avoid using the same adaptively selected examples as independent evidence for a generalization claim.

Before execution, verify that the command uses the intended data, convention, and resource limit. Check the evaluator on informative known cases. Distinguish exact decision procedures, checkable witnesses, probabilistic tests, numerical approximations, and searches with a cap. A timeout or failure to find a witness is not a proof of nonexistence. Version any necessary change to an evaluator and revisit claims that depend on it.

When canonicalizing or restricting mathematical objects, preserve any common ambient space required for their comparison or computation. If a run fails after completing valid checks, retain those checks and resume the unfinished part when its inputs and dependencies remain valid.

After execution, read the actual outputs and failures. Check metric definitions, signs, denominators, sample counts, and selection effects before interpreting a difference. Preserve failed and partial runs. Record the conclusion supported by the experiment, what remains uncertain, any suggested conjecture, and the reason for the next action.
