# HisFree Experimental Design

**Version:** 0.1

**Date:** 10 November 2026

**Scope:** text-first validation of Training-Free GRPO experience discovery followed by gradient-free weight editing; multimodal extension only after the text pipeline passes its gates.

## 1. Objective

HisFree tests whether a general-purpose open-weight model already contains useful historical-language capability that is difficult to access reliably, and whether this capability can be improved at low cost in two stages:

1. **Experience discovery:** Training-Free GRPO compares groups of target-model rollouts and distills their relative semantic advantages into a reusable historical reading prior, `E*`, without updating target-model parameters.
2. **Experience compilation:** Weight-Space Programming (WSP) or a closed-form prompt-to-weight method converts the validated effect of `E*` into a bounded persistent parameter update, `delta_theta`, without backpropagation through the target model.

The primary claim is not that sparse edits can inject an archive into a model. It is:

> A group-relative semantic experience can expose a latent historical capability, and part of that behavioral improvement can be compiled into the model's weights while preserving general capability.

The first paper evaluates the model and adaptation method. Integration with the existing semantic-search system is a later deployment project, not part of the core method.

## 2. Research questions and preregistered hypotheses

### RQ1: Does the base model contain accessible historical capability?

**H1.** The base model will perform substantially better on normalized or clean text than on equivalent diplomatic, OCR-corrupted, or historical-spelling views. A known-good instruction will repair a non-trivial subset of these failures.

### RQ2: Does Training-Free GRPO learn more than an ordinary prompt?

**H2.** A group-relative experience prior `E*` will outperform both a manually written historical prompt and a directly generated equal-length prompt on held-out historical examples.

### RQ3: Does the experience transfer?

**H3.** `E*` will improve at least one corpus not used during experience discovery and at least one input view or task not directly optimized in the reward.

### RQ4: Can the experience be compiled into weights?

**H4.** A bounded gradient-free update will retain a predeclared fraction of the prompt-only gain when `E*` is removed at inference.

### RQ5: Is the compiled model safe and cost-effective?

**H5.** The compiled model will beat matched random edits, remain within the general-capability regression tolerance, and reach a plausible inference-volume break-even point relative to sending `E*` on every request.

## 3. Claims that are out of scope

The initial experiments do not claim that:

- the model has memorized ECCO, Gale, COHA, or any complete archive;
- Training-Free GRPO is equivalent to gradient-based GRPO;
- a weight edit can add genuinely missing historical facts or a missing modality;
- improved OCR alone constitutes a general historical language model;
- retrieval quality is improved by the proposed model;
- one edited checkpoint is optimal for every period, genre, language, or task.

## 4. Experimental stages and gates

```text
E0  Reproduce TF-GRPO and WSP independently
 |
E1  Audit latent historical capability
 |   Gate A: a repeated, known-good behavior exists
 v
E2  Learn historical experience E*
 |   Gate B: E* beats equal-length prompt controls
 v
E3  Test prompt-only transfer
 |   Gate C: E* transfers beyond discovery data
 v
E4  Compile E* into weights
 |   Gate D: compiled model retains prompt gain and passes preservation
 v
E5  Cross-corpus, cross-task, and cost evaluation
 |   Gate E: result supports scale-up
 v
E6  Optional multimodal extension
```

Failure at a gate is a valid result. The next stage must not be run merely to produce a positive headline.

## 5. Model roles

### 5.1 Target models

Use three roles rather than committing every experiment to one scale:

| Role | Selection criterion | Purpose |
|---|---|---|
| Harness target | Small open model used by a released WSP experiment | Reproduce mechanics cheaply |
| Main target | Open text model in approximately the 4B--9B range with accessible weights and tokenizer | Primary historical experiments |
| Confirmation target | Different open model family at a comparable scale | Test family transfer and rule out architecture-specific effects |

The harness target is not evidence for a historical foundation model. The main target must show non-trivial base capability on clean historical text; otherwise there is no latent ability to expose.

### 5.2 Experience updater

Compare two updater settings:

- **Self-updater:** the target model summarizes the semantic advantage of its own rollout group.
- **Teacher-updater:** a stronger external or local model summarizes the target model's grouped rollouts.

The teacher-updater may improve experience quality but can inject its own historical assumptions. Report updater identity, version, prompt, tokens, and cost. The updater never sees validation or test examples.

### 5.3 Weight editor

The weight editor may be the same model as the teacher-updater or a separate reasoning model. It receives only:

- target architecture and permitted parameter summaries;
- discovery examples and grouped rollout traces;
- `E*` and its revision history;
- prompted/unprompted activation or logit contrasts;
- bounded practice feedback;
- the validation API.

It cannot access the test data, unrestricted internet, or arbitrary repository paths.

## 6. Data design

### 6.1 HisBench-mini

Build a 600--1,200-example development suite before scaling. It should cover three capabilities:

| Capability | Candidate source | Required output | Main metric |
|---|---|---|---|
| Historical QA and evidence localization | ChroniclingAmericaQA plus held-out ECCO material | answer and supporting span/page | answer F1 and evidence F1 |
| Entity/event extraction | HIPE plus project newspaper data | typed spans, links, or event tuples | micro/macro F1 and linking accuracy |
| Historical form and diachronic meaning | COHA/ECCO contrasts, SemEval-style items | sense/period judgment or diplomatic preservation | accuracy/correlation and preservation metrics |

OCR correction is not the sole task. OCR and historical spelling are controlled input views used to test whether semantic capabilities survive document noise.

### 6.2 Multi-view items

Where possible, each semantic item should have aligned views:

```json
{
  "item_id": "...",
  "work_id": "...",
  "edition_family": "...",
  "year": 1784,
  "genre": "newspaper",
  "normalized_text": "...",
  "diplomatic_text": "...",
  "raw_ocr": "...",
  "synthetic_ocr": "...",
  "page_image": "...",
  "question_or_task": "...",
  "gold_output": "...",
  "gold_evidence": "..."
}
```

The text-first study uses the first four textual views. `page_image` is reserved for E6.

### 6.3 Split policy

Never randomly split pages or sentences. Split by complete work and edition family, and separate reprints where metadata permits.

| Split | Purpose | Visible to updater/editor? |
|---|---|---:|
| Discovery/edit | Generate rollout groups and semantic advantages | yes |
| Validation | Select `E*`, accept/reject candidate edits, tune fixed budgets | through evaluation API only |
| Public test | Comparable final evaluation on public resources | no |
| Blind archive test | Private ECCO/Gale/newspaper evaluation | no; run once after freezing |
| General regression | Modern/unrelated capabilities | validation subset available; final subset hidden |

Recommended pilot sizes are approximately 100 discovery items, 100 validation items, 300 public-test items, and at least 300 blind-test items. Exact counts may change with annotation cost, but they must be frozen before the main runs.

### 6.4 Contamination controls

- hold out authors, titles, printers, and edition families where possible;
- include recently digitized or access-controlled material in the blind test;
- use counterfactual historical sentences for selected linguistic phenomena;
- check exact and fuzzy overlap across all splits;
- record whether each public benchmark may have appeared in pretraining;
- do not use test failures to revise `E*`, rewards, prompts, or edits.

## 7. Task formulation

### 7.1 Historical QA

Input: a question and a historical passage/view.

Output: a short answer plus an evidence span or sentence identifier.

Score:

- exact match and token F1;
- evidence span F1 or sentence recall;
- unsupported-claim rate;
- answer consistency across aligned views.

### 7.2 Entity and event extraction

Input: a historical passage/view.

Output: structured JSON with typed entity spans and, when available, links or event arguments.

Score:

- strict and relaxed span F1;
- type F1;
- entity-linking accuracy;
- JSON validity;
- cross-view entity consistency.

### 7.3 Historical form and diachronic meaning

Two subtests are permitted:

1. **Diplomatic preservation:** preserve valid historical forms while correcting only attested OCR errors.
2. **Temporal sense judgment:** select or describe the sense appropriate to a specified period and context.

Score:

- CER/WER where a transcription target exists;
- Historical Character Preservation Rate and Archaic Insertion Rate;
- modernization-error rate;
- task accuracy or correlation for semantic-change items.

## 8. E0: independent method reproduction

### E0.1 Training-Free GRPO reproduction

Reproduce a released small reasoning or search experiment with:

- the documented group size and number of experience-update epochs;
- the documented reward/ground-truth setting;
- an equal-length directly generated experience control;
- frozen sampling and evaluation settings;
- full API/token/cost logs.

Pass condition: the direction of the reported improvement is reproduced and grouped experience beats or clearly differs from the directly generated experience control.

### E0.2 WSP or prompt-to-weight reproduction

Reproduce one released edit end to end:

- read and hash the base checkpoint;
- apply, revert, and reapply the update deterministically;
- compare with norm-, layer-, rank-, and parameter-count-matched random edits;
- run the released held-out and preservation evaluations;
- save the exact changed tensor indices or low-rank factors.

Pass condition: the harness can reproduce a meaningful edit effect or identify and document why the released result cannot be reproduced.

## 9. E1: latent-capability audit

Run each main target under the following conditions:

```text
C0  no special prompt
C1  concise manual historical prompt
C2  few-shot historical demonstrations
```

Evaluate each condition on normalized, diplomatic, raw-OCR, and synthetic-OCR views.

### Known-good behavior criterion

A failure cluster is eligible for experience discovery only when:

1. the base model has non-trivial performance on the clean/normalized view;
2. performance drops repeatedly under a historical or noisy view;
3. at least one fixed prompt or equivalent rewording repairs a meaningful subset;
4. repaired and unrepaired cases can be separated on held-out validation;
5. the repair does not simply reveal the answer or copy test-specific facts.

Prioritize one cluster for the pilot. Recommended order:

1. modernization bias and diplomatic-form preservation;
2. evidence localization under historical spelling/OCR variation;
3. entity robustness under recurring OCR patterns;
4. temporal word-sense calibration.

## 10. E2: Training-Free GRPO experience discovery

### 10.1 Initial configuration

Use the following pilot defaults, then freeze them after validation:

| Variable | Pilot value |
|---|---:|
| Discovery items | 100 |
| Rollouts per group, `G` | 5 |
| Experience-update epochs | 3 |
| Target sampling temperature | 0.7 |
| Maximum experience length | 600 tokens |
| Independent experience trajectories | 5 main / 3 pilot |

These are starting values, not claimed optima. All decoding settings must remain identical across comparison arms.

### 10.2 Reward vector

Use decomposed rewards rather than one unconstrained LLM-judge score:

```text
R_total = w_task * task_correctness
        + w_evidence * evidence_grounding
        + w_preserve * historical_form_preservation
        + w_temporal * temporal_consistency
        + w_view * cross_view_consistency
        - w_halluc * unsupported_claims
        - w_modern * incorrect_modernization
        - w_cost * generation_and_tool_cost
```

Reward priority:

1. deterministic programmatic checks;
2. gold spans, labels, or citations;
3. calibrated LLM judge for genuinely open-ended criteria;
4. human adjudication for a stratified sample.

Reward weights are tuned only on discovery and validation data, then frozen. Report every component separately so an aggregate improvement cannot hide historical-form damage.

### 10.3 Semantic advantage update

For each group:

1. generate `G` outputs from the target conditioned on current `E[t]`;
2. calculate and expose the component rewards;
3. contrast the best, median, and worst rollouts;
4. ask the updater to state reusable procedural differences, not answers;
5. merge new rules into `E[t+1]`;
6. remove duplicate, contradictory, document-specific, and unverifiable rules;
7. evaluate the candidate experience on validation data.

The updater may return **no update**. Keep the shortest experience on a statistical tie.

### 10.4 Experience controls

Compare:

| ID | Condition |
|---|---|
| P0 | no historical prompt |
| P1 | expert/manual prompt |
| P2 | directly generated prompt of the same token length as `E*` |
| P3 | best-of-N prompt search with the same rollout budget |
| P4 | TF-GRPO self-updater `E*` |
| P5 | TF-GRPO teacher-updater `E*` |
| P6 | shuffled or semantically unrelated experience of the same length |

Gate B passes only if a TF-GRPO condition beats P1 and P2 on held-out validation and the gain is not explained only by additional prompt length.

## 11. E3: prompt-only transfer

Freeze `E*` and evaluate it without further updates on:

- unseen works from the discovery corpus;
- a different historical corpus;
- an unseen time band or genre;
- alternate text views;
- one task not directly represented in the experience reward;
- modern/general regression tasks.

Classify the result:

| Outcome | Interpretation |
|---|---|
| Same-task, same-corpus only | likely feedback overfitting |
| Same-task, cross-corpus | domain strategy transfer |
| Cross-task historical gain | broader historical reading strategy |
| Modern-task gain or neutrality | possible general robustness |
| Historical gain with modern regression | specialization trade-off |

Only transferable `E*` proceeds to weight compilation.

## 12. E4: experience-to-weight compilation

### 12.1 Paired traces

For the same discovery inputs, record paired model behavior:

```text
unprompted: model(x)
prompted:   model(E* + x)
```

Collect:

- output tokens and log probabilities;
- residual-stream, attention, and MLP activation summaries;
- token-embedding and tokenizer diagnostics;
- prompted-minus-unprompted deltas;
- repaired and newly broken examples.

The prompt-conditioned model acts as a behavioral teacher, not as ground truth. Only behaviors that also pass task rewards are eligible for compilation.

### 12.2 Compilation routes

**Route A: Agentic WSP**

The editor may inspect paired traces and propose changes within a declared parameter mask. It can evaluate candidates on discovery and validation APIs, but it receives no gradients.

**Route B: Closed-form prompt-to-weight distillation**

Fit a bounded low-rank or least-norm update that makes selected unprompted internal states approximate validated `E*`-conditioned states.

**Route C: WSP-only control**

Give the editor historical failures and validation feedback but withhold `E*` and paired prompt traces. This tests whether experience discovery provides useful structure beyond direct editing.

### 12.3 Edit budgets

Run at least three predeclared edit-size tiers when feasible:

```text
tiny:    <= 0.001% of target parameters or equivalent stored factors
small:   <= 0.01%
medium:  <= 0.1%
```

Report both stored edit size and the number of dense weight values numerically changed after materialization. A low-rank update is not necessarily sparse.

Every candidate is checked for:

- validation historical gain;
- compilation-retention ratio;
- general regression;
- new failures;
- edit norm and affected layers;
- reproducibility after save/load;
- robustness after deployment quantization.

The editor may retain the base checkpoint when no candidate passes.

## 13. Experimental comparison matrix

### 13.1 Core arms

| ID | Method | Runtime experience | Weight update | Target gradients |
|---|---|---:|---:|---:|
| M0 | Base model | no | no | no |
| M1 | Manual prompt | yes | no | no |
| M2 | Direct generated prompt | yes | no | no |
| M3 | Training-Free GRPO `E*` | yes | no | no |
| M4 | WSP-only | no | yes | no |
| M5 | TF-GRPO -> WSP | no | yes | no |
| M6 | TF-GRPO -> closed-form edit | no | yes | no |
| M7 | Compiled model + runtime `E*` | yes | yes | no |
| M8 | LoRA/PEFT | no | yes | yes |

### 13.2 Extended baselines

- activation steering derived from prompted/unprompted contrasts;
- in-context editing or external memory;
- continued pretraining where compute permits;
- conventional model editing on compatible targets;
- matched random edits;
- historical pretrained models on compatible tasks.

M0--M6 are required for the central claim. M7 diagnoses complementarity; M8 quantifies the quality/cost trade-off against a trained baseline.

## 14. Primary metrics

### 14.1 Historical task performance

Report each task metric separately. A macro `HistAvg` may be reported only after all metrics are mapped to a predeclared 0--100 scale.

### 14.2 Experience gain

```text
Gain_E = Score(M3) - Score(M0)
```

### 14.3 Compilation-retention ratio

For compiled model `Mc`:

```text
Retention = (Score(Mc) - Score(M0)) / (Score(M3) - Score(M0))
```

Predeclare the main success threshold before the blind test. A reasonable initial threshold is retention of at least 50% of the prompt-only gain, but the final threshold should reflect metric variance and practical effect size.

### 14.4 General retention

Report:

- average change across general benchmarks;
- worst individual-task regression;
- number of examples repaired and broken;
- unrelated historical-task regression.

A recommended initial gate is no more than one absolute point of average regression and no unexplained large single-task drop. Freeze the exact tolerance after pilot variance analysis.

### 14.5 Transfer

Report gains separately for:

- cross-work;
- cross-corpus;
- cross-period;
- cross-genre;
- cross-view;
- cross-task;
- cross-model family.

### 14.6 Cost

Record:

- discovery examples and rollouts;
- target and updater tokens;
- updater/editor API cost;
- target forward passes;
- wall-clock and GPU/CPU hours;
- stored edit size;
- runtime prompt tokens and latency;
- quantized and unquantized model size.

Let:

```text
C_compile = one-time experience discovery, compilation, and validation cost
C_prompt  = per-query cost with E*
C_edit    = per-query cost of the compiled model

N_break-even = C_compile / (C_prompt - C_edit)
```

Report `N_break-even` only when `C_prompt > C_edit`; otherwise compilation has no measured serving-cost advantage.

## 15. Statistical protocol

- Use paired evaluation because every model sees the same items and decoding budget.
- Use paired bootstrap confidence intervals for F1, accuracy, correlation, and aggregate scores.
- Use exact McNemar tests for paired binary repairs/regressions where appropriate.
- Run at least five independent main experience/edit trajectories when budget permits; use three only for pilots.
- Correct families of secondary comparisons with Holm or Benjamini-Hochberg procedures.
- Report effect sizes and confidence intervals, not only p-values.
- Freeze prompts, rewards, decoding, model versions, and splits before the blind test.
- Run the blind test once. Any subsequent use must be labeled post hoc.

## 16. Failure analysis

Assign every sampled failure to one or more categories:

```text
missing historical knowledge
historical spelling/tokenization
temporal word sense
OCR corruption
entity rarity/linking
layout or reading order
language-prior modernization
evidence localization
multi-document synthesis
instruction following/format
reward or judge error
edit-induced regression
```

For each method, report transitions between categories, not only aggregate accuracy. This reveals whether the method repairs a general bottleneck or moves errors elsewhere.

## 17. Reproducibility and artifact contract

Every run stores:

```text
run_id
git_commit
base_model_id_and_hash
tokenizer_or_processor_hash
dataset_manifest_hash
split_hashes
prompt_and_experience_versions
rollout_group_and_reward_components
updater_and_editor_identity
changed_tensor_hashes_or_low_rank_factors
decoding_configuration
task_metrics
repair_and_regression_lists
token_compute_and_cost_log
```

Release when licenses permit:

- evaluation code and public-data manifests;
- `E*` histories and grouped semantic advantages;
- editor prompts and complete trajectories;
- edit patches or factors rather than restricted source data;
- matched random-edit generators;
- failed and no-edit runs;
- a model card describing intended periods, genres, and limitations.

## 18. Quantization and deployment robustness

Deployment is not part of the first paper's primary claim, but the edited checkpoint must survive ordinary packaging if low cost is claimed.

Compare:

1. full-precision base and edit;
2. edit first, then quantize;
3. quantized base with an edit applied where technically valid.

Small edits can disappear or change under 8-bit/4-bit quantization. Report historical gain and general regression again after quantization; do not assume the full-precision result transfers.

## 19. Multimodal extension gate

Run E6 only after the text pipeline demonstrates transferable `E*` or an informative compilation boundary.

For aligned ECCO/newspaper pages, compare:

```text
OCR only
image only
image + OCR
image + clean transcription
```

Learn an experience prior that explicitly compares visual evidence with language priors. Compile first into an isolated vision projector or cross-modal adapter, then test selected original-weight regions. Required additional metrics include page evidence localization, reading order, image/OCR disagreement resolution, and visual hallucination.

## 20. First executable pilot

The first pilot should be small enough to complete before scaling:

1. Select one main 4B--9B open target and one small harness target.
2. Reproduce one Training-Free GRPO result and one WSP/prompt-to-weight result.
3. Build 500 HisBench-mini items:
   - 200 historical QA/evidence items;
   - 150 entity/event items;
   - 150 historical-form or temporal-semantics items.
4. Freeze 100 discovery, 100 validation, and 300 public-test items by work/edition family.
5. Audit normalized, diplomatic, raw-OCR, and synthetic-OCR views.
6. Select exactly one known-good failure cluster.
7. Run P0--P6 and test whether TF-GRPO learns a transferable `E*`.
8. If Gate C passes, run WSP-only, TF-GRPO-to-WSP, and closed-form compilation.
9. Compare with one LoRA baseline using the same discovery examples.
10. Decide whether to scale, change the target failure, or publish a boundary/negative result.

## 21. Paper-ready result tables

Prepare the following tables from the first implementation:

1. Base-model capability audit by task and input view.
2. Prompt/experience comparison P0--P6.
3. Core method comparison M0--M8.
4. Cross-corpus, cross-task, and cross-view transfer.
5. General-capability regression and repairs/breaks.
6. Edit size, rollout/API cost, latency, and break-even analysis.
7. Quantization robustness.
8. Failure-category transitions and no-edit outcomes.

The central paper result is supported only if the method tables show that group-relative experience adds value beyond ordinary prompting and that weight compilation adds persistent value beyond keeping that experience in context.
