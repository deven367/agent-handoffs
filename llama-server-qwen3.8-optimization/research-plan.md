# Reasoning Budgets, MTP, and Speculative Decoding Research Plan

**Status:** Proposed  
**Model:** openai-codex/gpt-5.6-sol  
**Date:** 2026-09-14

## Objective

Determine whether hard reasoning budgets interact with Qwen's multi-token prediction (MTP) speculative decoding, and whether draft rejection can predict reasoning difficulty, non-convergence, or degraded answers near a forced cutoff.

The core distinction is:

- **Reasoning budget:** limits committed tokens inside the reasoning block.
- **MTP/speculative decoding:** attempts to produce those committed tokens with fewer target-model decoding steps.

For lossless speculative decoding, rejected drafts are uncommitted proposals. They should consume compute but should not enter the KV context, count against the reasoning budget, or change the target model's output distribution.

## Current server baseline

The repository's `Makefile` enables model-native MTP with:

```text
--spec-type draft-mtp
```

It does not currently pass a reasoning budget. Under current `llama.cpp` defaults, the effective hard budget is therefore unrestricted (`--reasoning-budget -1`), while reasoning mode and effort are delegated to the model's chat template.

The primary null hypothesis is:

> With exact verification and the same checkpoint, enabling MTP changes latency and compute usage but does not systematically change answer correctness, style, calibration, or reasoning-budget behavior.

A systematic semantic change would indicate approximate acceptance, sampler/RNG differences, numerical effects, state leakage, or an implementation bug.

## Research questions

1. How does answer quality change as the hard reasoning budget moves from zero through extremely small, moderate, and unrestricted values?
2. Is forced partial reasoning worse than native non-thinking mode at the same or lower token cost?
3. Does advance notice of the budget produce more graceful degradation than a surprise hard cutoff?
4. Does MTP preserve committed token sequences and budget boundaries under deterministic decoding?
5. Under stochastic decoding, does MTP preserve aggregate answer distributions even when individual trajectories differ?
6. Do draft entropy, rejection rate, or accepted-run length predict reasoning pivots, eventual non-convergence, or incorrect answers?
7. Does the forced `</think>` boundary cause a local rejection spike or throughput penalty?
8. Do tiny budgets create measurable style changes—such as increased terseness, premature confidence, hedging, or clarification requests—without invoking the unsupported concept of a new "persona"?

## Hypotheses

- **H1 — Budget cliff:** Quality will degrade nonlinearly below a task-dependent budget rather than smoothly with every removed token.
- **H2 — Broken-chain penalty:** On multi-step tasks, truncated reasoning can perform worse than native non-thinking mode because the final answer conditions on an incomplete reasoning trajectory.
- **H3 — Budget awareness helps:** An explicit remaining-budget instruction will outperform an unexpected hard cutoff at equal committed-token limits.
- **H4 — MTP semantic neutrality:** Lossless MTP will not produce a systematic quality or style shift relative to ordinary target decoding.
- **H5 — Rejection is a useful but limited signal:** Rejection and draft entropy will predict local target/drafter disagreement and wasted compute, but will be weaker predictors of semantic uncertainty or final correctness.
- **H6 — Boundary artifact:** Forced budget closure will reduce MTP acceptance around the transition because the sampler, rather than the draft head, determines the closing sequence.

## Experimental design

### Factors

| Factor | Conditions |
|---|---|
| Reasoning budget | `0`, `10`, `32`, `128`, `512`, `2048`, unrestricted |
| MTP | off, `draft-mtp` |
| Termination policy | native non-thinking, silent forced close, forced close with budget message, advance budget instruction |
| Decoding | deterministic diagnostic, production sampling |
| Task class | arithmetic, multi-step math, coding, factual QA, open-ended judgment |

Use one checkpoint and one quantization for all MTP comparisons. Comparing separately trained MTP and non-MTP checkpoints would confound inference behavior with the MTP training objective.

### Stage 1: Implementation invariants

Use short deterministic prompts and MTP off/on.

Verify:

1. committed output is token-identical when no hard budget is reached;
2. exactly the configured number of reasoning tokens is committed before forced closure;
3. rejected drafts do not count toward the reasoning budget;
4. rejected drafts do not survive in the KV context;
5. forced closing tokens and the final answer still consume the overall generation/context allowance;
6. the server leaves enough output allowance for the forced close and final answer;
7. MTP proposals that extend past a budget boundary are rejected or clipped without committing excess reasoning tokens.

A failure here is an implementation issue, not a model-behavior result.

### Stage 2: Budget-response curves

Run the complete budget grid with MTP disabled first. This isolates budget behavior before adding speculative decoding.

For each prompt and condition, record:

- reasoning tokens;
- final-answer tokens;
- whether the reasoning ended naturally or was forced;
- whether the answer completed normally;
- correctness or task score;
- explicit confidence when the task supports it;
- wall time and tokens per second.

Keep enough total generation capacity for the largest bounded reasoning condition plus a complete final answer. Otherwise `max_tokens` truncation becomes a second, uncontrolled stopping mechanism.

### Stage 3: MTP interaction

Repeat Stage 2 with `draft-mtp` enabled and collect:

- proposed draft tokens;
- accepted draft tokens;
- rejected draft tokens;
- average accepted length per verification step;
- rejection position within each draft;
- target and draft entropy, if available;
- verification-step latency;
- acceptance statistics split across early reasoning, late reasoning, forced-close boundary, and final answer.

The main interaction test is whether the budget-response curve changes systematically when MTP is enabled. Under lossless decoding it should not.

### Stage 4: Budget awareness and apparent metacognition

Compare three distinct mechanisms rather than treating them as equivalent:

1. **Invisible hard budget:** the server counts tokens and intervenes without advance warning.
2. **Advance budget information:** the prompt or supported template tells the model its limit before generation.
3. **Remaining-budget feedback:** periodically insert a remaining-token signal, requiring a custom loop or a budget-aware fine-tune.

Measure whether the model:

- reaches a conclusion before the limit;
- shortens or reorganizes intermediate reasoning;
- abandons alternatives sooner;
- asks for clarification on under-specified questions;
- expresses calibrated uncertainty;
- emits an answer that is complete but wrong versus visibly incomplete.

This tests functional self-regulation. It does not establish subjective awareness or human-like introspection.

## Workloads

Use established benchmarks where possible:

- **Arithmetic/basic reasoning:** GSM8K or a deterministic subset.
- **Hard mathematical reasoning:** MATH-500 or AIME-style problems.
- **Coding:** HumanEval with executable evaluation.
- **Factual QA:** questions with fixed reference answers.
- **Open-ended judgment:** paired decisions with explicit rubrics and under-specified cases where clarification is valid.

Start with a small stratified pilot. Expand only after logs confirm that the budget and MTP measurements are correct.

## Metrics

### Quality

- exact match or benchmark score;
- executable pass rate for code;
- completion/valid-format rate;
- calibration error or Brier score where confidence is elicited;
- clarification rate for intentionally under-specified prompts.

### Efficiency

- end-to-end latency;
- reasoning and final-answer tokens per second;
- target-model verification calls;
- mean accepted draft length;
- accepted/proposed token ratio;
- rejected-draft compute ratio.

### Behavioral style

Avoid assigning anthropomorphic personas. Quantify observable behavior instead:

- answer length;
- hedging frequency;
- unsupported-confidence rate;
- number of alternatives considered;
- self-correction frequency;
- instruction adherence;
- premature-answer and incomplete-answer rates.

### Analysis

- Use paired prompts across every condition.
- For stochastic decoding, run multiple seeds and compare aggregate distributions; same-seed text identity is not guaranteed by distribution-preserving speculation.
- Report bootstrap confidence intervals for score and latency differences.
- Model the primary interaction as outcome versus budget, MTP, task class, and `budget × MTP`.
- Separate natural completions from forced completions; pooling them hides the phenomenon of interest.

## Interpretation rules

1. **Rejected drafts are not hidden thoughts.** They are counterfactual proposals unless an implementation accidentally commits them.
2. **Rejection is not direct confidence.** It measures disagreement between the drafter and target; disagreement may come from model mismatch, entropy, sampling temperature, or domain shift.
3. **MTP training and MTP inference are different interventions.** Training can alter capability; verified drafting should only alter inference efficiency.
4. **Exact and approximate speculation must not be mixed.** Relaxed acceptance can change output quality and distribution by design.
5. **A tiny hard budget is not equivalent to non-thinking mode.** One leaves a partial trace in context; the other begins on a direct-answer trajectory.
6. **A weak activation or entropy predictor is not proof of metacognition.** It only shows that an external controller can extract predictive information.

## Expected outcomes

The likely result is:

- MTP improves throughput without a systematic semantic effect;
- accepted length falls in locally unpredictable reasoning regions;
- rejected-draft rate is useful for scheduling speculation depth but only a noisy proxy for task-level uncertainty;
- hard tiny budgets produce unstable and task-dependent quality cliffs;
- advance budget awareness degrades more gracefully than surprise termination;
- any repeatable MTP-induced style or correctness shift warrants an implementation audit.

A practical follow-up would be an adaptive controller that uses entropy or recent acceptance length to change MTP draft depth while leaving the reasoning budget independent. Rejection rate should not directly terminate reasoning until it is shown to predict correctness beyond simple token entropy and task difficulty.

## Prior work

### Reasoning budgets and test-time compute

- Snell et al., [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314), 2024. Shows that useful test-time compute allocation depends on prompt difficulty.
- Muennighoff et al., [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393), 2025. Introduces budget forcing through forced termination and repeated `Wait` continuations.
- Han et al., [Token-Budget-Aware LLM Reasoning](https://aclanthology.org/2025.findings-acl.1274/), ACL Findings 2025. Studies prompt-level budgets and task-adaptive token allocation.
- Wen et al., [BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens](https://arxiv.org/abs/2508.17196), 2025. Periodically exposes remaining budget through control tokens and trains for budget adherence.
- Alomrani et al., [Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs](https://arxiv.org/abs/2507.02076), 2025. Separates fixed-budget controllability from difficulty-aware adaptiveness.
- Su et al., [Broken Chains: The Cost of Incomplete Reasoning in LLMs](https://arxiv.org/abs/2602.14444), 2026 preprint. Reports that truncated reasoning can be worse than no reasoning and that robustness is model-dependent.
- Oladri et al., [Token Budget Saturation and Mechanistic Early Detection of Reasoning Non-Convergence](https://arxiv.org/abs/2607.21433), 2026 preprint. Finds a modest early hidden-state signal for eventual non-convergence.

### Speculative decoding and MTP

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2022/2023. Establishes distribution-preserving speculative decoding.
- Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023. Formalizes modified rejection sampling that preserves the target distribution within hardware numerics.
- Xia et al., [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851), 2024. Surveys drafter and verification designs.
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737), 2024. Studies MTP as both a training objective and an inference accelerator.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024. Uses an MTP objective and retains the auxiliary module for speculative decoding.
- Cai et al., [Medusa](https://arxiv.org/abs/2401.10774), 2024. Predicts multiple future tokens with extra heads and verifies candidate trees.
- Li et al., [EAGLE](https://arxiv.org/abs/2401.15077), 2024. Moves speculative prediction to the feature level.
- Zhang et al., [Draft Model Knows When to Stop](https://arxiv.org/abs/2411.18462), EMNLP 2025. Uses draft entropy to choose dynamic speculative lengths and evaluates long-form QwQ reasoning.
- Cai et al., [FastMTP](https://arxiv.org/abs/2509.18362), 2025. Aligns MTP training with recursive drafting and reports improved acceptance and lossless speedup.
- Chen, [EntMTP](https://arxiv.org/abs/2606.27550), 2026 preprint. Selects MTP tree depth using local generation entropy.
- Su et al., [Entropy-Aware Token Rejection for Improving Speculative Decoding](https://arxiv.org/abs/2512.23765), 2026 revision. Intentionally changes rejection behavior to improve reasoning, moving beyond semantically neutral acceleration.

### Thought-level speculation

- Wang et al., [Efficient Reasoning for LLMs through Speculative Chain-of-Thought](https://arxiv.org/abs/2504.19095), 2025. Drafts and verifies complete thought-level candidates.
- Shi et al., [SpecCoT: Accelerating Chain-of-Thought Reasoning through Speculative Exploration](https://aclanthology.org/2025.findings-emnlp.1326/), EMNLP Findings 2025. Uses step-level drafting and verification rather than local token speculation.

## Deliverables

1. Reproducible launcher matrix for budget, termination, and MTP conditions.
2. Per-request JSONL logs containing committed-token and draft-verification metrics.
3. Deterministic invariant report for MTP and budget boundaries.
4. Budget-response curves by task class and termination policy.
5. Acceptance/rejection traces aligned to reasoning position and forced-close boundaries.
6. Short findings report separating verified results from speculation.
