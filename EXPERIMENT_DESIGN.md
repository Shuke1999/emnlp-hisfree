# HisFree Experimental Design

**Version:** 0.2

**Date:** 10 November 2026

**Companion document:** [BACKGROUND_AND_PLAN.md](BACKGROUND_AND_PLAN.md) v0.2. This design implements that plan. Section numbers like "Plan §6.3" refer to it.

**Scope:** text only. Two main target models from different families, one primary behavior (B1), one contrast behavior (B3), and one optional behavior (B2).

## 1. Objective

Measure, for historical reading behaviors that a model shows only when prompted, how much of the prompt's gain survives when the behavior is compiled into the weights **without gradients**. Compare this with gradient-based compilation, and test whether locality diagnostics measured before editing predict the result.

Primary claim to test:

> Gradient-free compilability of a prompted behavior is predicted by how concentrated the prompt's effect is inside the model. Modernization suppression (B1) is concentrated and compiles into a small edit; evidence-first answering (B3) is distributed and does not. Gradient-based distillation compiles both.

## 2. Preregistered hypotheses and tests

Each hypothesis has a fixed test and threshold. Thresholds marked *pilot* may be revised once, after Phase 0, and are then frozen before any validation-set compilation result is seen.

| ID | Hypothesis | Test | Threshold |
|---|---|---|---|
| H1a | B1 has headroom | Modernization-error rate (ModErr) under C0 on validation | ≥ 10% of valid archaic tokens (*pilot*) |
| H1b | A prompt removes much of it | Relative ModErr reduction C1 vs C0, paired bootstrap 95% CI | lower CI bound > 0; point estimate ≥ 30% (*pilot*) |
| H1c | B3 has headroom | Evidence-F1 gain C1 vs C0 | lower CI bound > 0; ≥ 5 points (*pilot*) |
| H2a | B1 compiles without gradients | Best gradient-free route (R1–R3) retention on B1 test, small or medium tier | ≥ 50%, beats all matched random edits (R6), passes preservation |
| H2b | B3 compiles less | Best gradient-free retention on B3 minus on B1 | negative, CI excludes 0 |
| H2c | Gradients compile both | R5 retention on B1 and B3 | ≥ 70% on both (*pilot*) |
| H3 | Locality predicts compilability | Spearman ρ between the primary diagnostic L1 and best gradient-free retention across analysis units (§10.6) | ρ > 0, permutation p < 0.05, same sign on both families |
| H4 | Short instructions compile better than TF-GRPO experience at matched gain | Retention(S1) − Retention(S4) under R1, restricted to pairs with prompt gains within 20% of each other | > 0 (secondary; reported only if S4 gives a significant prompt gain) |
| H5 | B1 edit transfers and preserves | Retention on cross-corpus and counterfactual test; preservation gate (§11.4) | retention ≥ 30% cross-corpus; gate passed |

## 3. Out of scope

- Page images, vision projectors, and other multimodal inputs.
- Injecting archive facts into weights.
- Claims that gradient-free compilation is cheaper than LoRA.
- Building a historical foundation model or matching TypewriterLM.
- Retrieval system quality.

## 4. Stages and gates

```text
E0  Kill test (Phase 0, weeks 1-2)
 |   G1: H1a, H1b (and H1c) pass; R1 shows any B1 retention
 v
E1  Harness and reproduction (weeks 3-4)
 |   G2: R1 reproduces a Transmuting Prompts result; R5 reaches high B1 retention
 v
E2  Full B1 study on family A (weeks 5-8)
 |   G3: some gradient-free route beats matched random edits on B1 validation
 v
E3  B3 (+ B2) and family B (weeks 9-11)
 |   G4: B3 complete on both families
 v
E4  Locality-retention analysis (week 11)
 |   G5: diagnostics frozen before any retention value was seen
 v
E5  Transfer, robustness, final test run once (weeks 12-14)
```

Failing a gate is a result, not a reason to skip ahead. What to do on failure is specified in Plan §9.

## 5. Models

### 5.1 Target models

| Role | Selection criteria | Use |
|---|---|---|
| Reproduction target | Gemma 3 1B Instruct (the model in Transmuting Prompts) | E1 only: validate R1 against a published result |
| Family A main target | Open weights, 4B–9B, instruction-tuned, SwiGLU MLP with RMSNorm, untied or easily untied embeddings | All experiments |
| Family B main target | Same criteria, different pretraining family and tokenizer | E3 onward; confirms that results are not architecture-specific |

Candidate pairs: a Qwen-family model and a Llama- or Gemma-family model of similar size. The final choice is made in Phase 0 by two criteria: H1a headroom on the B1 pilot, and whether archaic and modern spellings tokenize into different token sequences (if they do not, R2 cannot target them).

Record for each target: model ID, revision hash, tokenizer hash, chat template, whether embeddings are tied, number of layers, hidden size, MLP width.

### 5.2 External models

| Role | Constraint |
|---|---|
| TF-GRPO teacher-updater (S4) | Any strong model; identity, version, prompts, and token counts logged. Never sees validation or test items |
| Direct-prompt generator (S3) | Same constraint; sees only the task description |
| WSP editor (R3) | A frontier reasoning model with a code workspace; sees only what §9.3 permits |
| LLM judge | **Not used** for primary metrics. All primary metrics are deterministic |

## 6. Data

### 6.1 B1: historical-form preservation

**Sources.**

- *Discovery, validation, public test:* ECCO-TCP diplomatic transcriptions (18th century) aligned with ECCO OCR for the same pages, where licenses allow. If ECCO OCR is unavailable, use synthetic OCR (below) over ECCO-TCP text.
- *Cross-corpus test:* BLN600 (19th-century British Library newspapers, OCR plus human transcription), or the KB historical newspaper ground truth if an English source is not usable. This tests a different century, genre, and OCR engine.
- *Counterfactual test:* 100 recent modern-English sentences (published after both targets' training cutoffs) converted to 18th-century spelling with a rule set built from attested variant pairs (§6.1.3). These cannot have been memorized.

**Item construction.**

1. Align OCR and diplomatic gold at the character level with edit-distance alignment; cut segments of 80–200 tokens at sentence boundaries.
2. Mark **valid archaic tokens** `A`: gold tokens whose standard modern spelling differs (e.g. *shew*, *publick*, *compleat*, *antient*, *-eth* verb forms, *ye* as article). Use a variant lexicon built from paired diplomatic and normalized editions plus a modern dictionary check. A second annotator checks 10% of items; report agreement.
3. Mark **OCR-error tokens** `O`: positions where the OCR input differs from gold.
4. Keep items with |A| ≥ 2 and |O| ≥ 1, so that copying the input and modernizing everything both score badly.
5. Long s, ligatures, and capitalization follow each source's gold convention; they are scored only if the gold encodes them.

**Synthetic OCR.** Estimate a character confusion model (substitution, insertion, deletion, common confusions such as *s/f*, *c/e*, *rn/m*) from aligned real OCR. Generate corrupted versions of diplomatic text at three CER levels: 2%, 5%, and 10%. Each item records its CER level.

**Variant rule set for counterfactuals.** Rules are derived only from discovery data and a published list of 18th-century spelling conventions. Rules seen in discovery are flagged, so that counterfactual items can be split into seen-pattern and unseen-pattern subsets.

**Sizes.**

| Split | Items | Source |
|---|---:|---|
| Discovery | 100 | ECCO-TCP + OCR |
| Validation | 100 | ECCO-TCP + OCR, different works |
| Public test | 300 | ECCO-TCP + OCR, different works and edition families |
| Cross-corpus test | 300 | BLN600 or equivalent |
| Counterfactual test | 100 | Synthetic period spelling over unseen modern text |
| Normalization-on-request | 100 | Public test items with the instruction "modernize the spelling" (§6.4) |

### 6.2 B3: evidence-first answering under historical noise

**Source.** ChroniclingAmericaQA records, which include question, answer, raw OCR context, corrected context, date, and source URL.

**Gold evidence.** For each item, find the sentence in the corrected context that contains the answer, and map it to the OCR context through character alignment. Keep only items where the match is unique. A human checks 10% of items.

**Unanswerable variants.** For 15% of items, replace the context with an article from the same newspaper and year that does not support the answer. The gold output is "not supported by the passage."

**Splits.** Split by newspaper title and decade. Use one set of decades for discovery, validation, and public test, and hold out the later decades for a cross-period test.

| Split | Items |
|---|---:|
| Discovery | 100 |
| Validation | 100 |
| Public test | 300 |
| Cross-period test | 200 |
| Cross-corpus test (optional) | 100 hand-verified QA items over held-out ECCO-TCP passages |

### 6.3 B2 (optional): period word sense

Built only if E2 finishes on schedule.

- 25 target words with well-documented sense change (from SemEval-2020 Task 1 English targets and standard examples such as *nice*, *awful*, *silly*, *gay*, *broadcast*).
- For each word, 8–12 dated contexts from COHA or ECCO-TCP, half before and half after the shift.
- Multiple-choice sense labels from a historical dictionary, double-annotated; report Cohen's κ and drop items without agreement.
- About 250 items, split by word into discovery/validation/test so that test words are unseen.

### 6.4 Preservation suite

| Component | Size | What it catches |
|---|---:|---|
| MMLU subset | 500 | General knowledge regression |
| GSM8K subset | 300 | Reasoning regression |
| IFEval | full (541) | Instruction-following regression |
| Modern spelling correction | 300 | **False archaisms**: an edit that archaizes modern text |
| Modern copy/transcription | 200 | Edit leaking into ordinary text reproduction |
| Normalization-on-request (B1) | 100 | Whether the edited model still modernizes when asked; tests that the edit changed a default, not removed an ability |

A 20% slice of each component is visible during validation; the remaining 80% is used only in the final run.

### 6.5 Split and contamination policy

- Split by complete work and edition family for ECCO material, and by newspaper title and decade for newspapers. Never split randomly by page or sentence.
- Check exact and fuzzy (MinHash, Jaccard ≥ 0.8) overlap across all splits; remove collisions.
- Report, for each public source, whether it is likely in pretraining data. The counterfactual B1 test is the contamination-free check.
- Discovery data is the only data that updaters, editors, closed-form solvers, and LoRA ever see as input. Validation is accessed only through the evaluation API, as scores. Test sets are run once at the end.

## 7. Behavior sources and prompts

### 7.1 Sources

| ID | Source | Construction |
|---|---|---|
| S1 | Short manual instruction | Written once by the authors, frozen before Phase 0 results (drafts in §7.2) |
| S2 | S1 + 3 demonstrations | Demonstrations drawn from discovery, fixed across runs |
| S3 | Directly generated instruction | An external model writes an instruction from the task description only, length matched to the longest of S1/S2/S4 |
| S4 | TF-GRPO experience prior `E*` | §7.3 |
| S5 | Length-matched unrelated text | Shuffled sentences from an unrelated instruction, matched in tokens to S4 |

### 7.2 Draft S1 instructions

B1:

```text
This text is an OCR reading of an eighteenth-century printed source. Produce a
diplomatic transcription. Correct only clear OCR mistakes. Keep every historical
spelling, word form, and capitalization exactly as printed, even where modern
English differs (for example "shew", "publick", "compleat", "hath"). Do not
modernize.
```

B3:

```text
Answer the question using only the passage below, which is OCR text from a
historical newspaper and may contain recognition errors. First quote the exact
sentence from the passage that supports your answer, then give the answer. If
the passage does not support an answer, say "not supported by the passage".
```

The final wording is frozen at the end of Phase 0 week 1, before any compilation run.

### 7.3 TF-GRPO configuration (S4)

| Variable | Value |
|---|---:|
| Discovery items | 100 |
| Rollouts per group `G` | 5 |
| Epochs | 3 |
| Rollout temperature | 0.7 |
| Max experience length | 600 tokens |
| Trajectories | 3 |
| Updaters | self-updater and teacher-updater, reported separately |

Reward is the primary behavior metric (§11.1) computed deterministically, plus guard terms: for B1, OCR-fix recall and Archaic Insertion Rate; for B3, answer F1 and unsupported-claim rate. The updater sees component rewards and best/median/worst rollouts, and is told to write procedures, not answers. Select `E*` on validation; keep the shortest experience on a statistical tie.

## 8. Decoding and evaluation settings

- **Evaluation:** greedy decoding, fixed chat template, fixed max new tokens (B1: 1.5 × input length; B3: 128). Free generation only.
- **TF-GRPO rollouts:** temperature 0.7, same template.
- **Activation collection for compilation and diagnostics:** teacher-forced on the prompted model's own validated outputs (§9.1). This is used only to compute edits and diagnostics, never to score success.
- Serving through vLLM for evaluation; HuggingFace with forward hooks for activation collection and edits. Edited weights are exported and reloaded into vLLM for evaluation, and hashes are checked.

## 9. Compilation routes

### 9.1 Paired activation collection

For each discovery input `x` and source `S`:

1. Generate `y_P = model(S + x)` with greedy decoding.
2. Keep `y_P` only if it scores at least as well as the unprompted output `y_U` on the primary metric. These are the **validated targets**.
3. Run the model teacher-forced on `(S + x, y_P)` and on `(x, y_P)`, and record at the positions of `y_P`, for every layer: residual stream, MLP input after RMSNorm, MLP output, and final hidden state.

The prompted run is the teacher. Differences at aligned `y_P` positions define the prompt effect.

### 9.2 Routes

**R1: closed-form prompt-to-weight compilation.** Following Transmuting Prompts:

- For layer `l`, let `A` be the stacked unprompted MLP inputs and `D` the prompted-minus-unprompted discrepancy at the MLP input after projection by the target matrix (`W δ` in that paper's notation).
- Solve the ridge problem `ΔW = D Aᵀ (A Aᵀ + ρ I)⁻¹` for `W_up` and `W_gate`, and absorb the mean shift into `W_down` as in that paper's appendix.
- Truncate `ΔW` to rank `r ∈ {1, 4, 16, 64}` by SVD, and scale by `η`.
- Apply layer by layer from early to late, recollecting activations after each edited layer.
- Layer sets: the single best layer by L1 (§10.1), the top-3 layers by L1, and all layers.
- Hyperparameters `ρ`, `η`, `r`, and the layer set are chosen on validation within each size tier.

**R2: closed-form token-row edit (B1 only).**

- From discovery outputs, collect pairs `(a, m)` where gold has archaic form `a` and the unprompted model produced modern form `m`. Use the first differing token of each form.
- Compute a historical-context direction `c` as the mean final hidden state on historical inputs minus the mean on modern inputs, both from unprompted runs.
- For each pair, update the unembedding rows `u_a ← u_a + β ĉ` and `u_m ← u_m − β ĉ`, with `ĉ = c / ‖c‖`. This raises the archaic form's logit relative to the modern form in proportion to how historical the context looks.
- Tied embeddings: untie and edit only the output matrix, and record that.
- `β` is chosen on validation.
- Report retention separately on **seen pairs** (in discovery) and **unseen pairs**. Retention only on seen pairs means lexical memorization, not a general prior.

**R3: agentic WSP.**

- The editor receives: architecture summary; the permitted parameter mask (the R1 layer sets plus embedding and unembedding rows); discovery items with paired prompted/unprompted outputs; activation summaries and the L1–L5 reports; the source prompt; a Python workspace with read, patch, revert, and hash tools; and an evaluation API returning **only** aggregate validation scores.
- Budget: at most 200 validation evaluations and a fixed editor-token cap per trajectory. Each candidate must fit a declared size tier.
- 3 independent trajectories per behavior and model. The editor may return no edit.
- Not allowed: test data, internet, gradients, and repository paths outside the workspace.
- Run on B1 for both families. Run on B3 for family A only, unless B3 R3 results differ meaningfully from R1.

**R4: activation steering (non-persistent reference).** Add `α (mean h_P − mean h_U)` at layer `l` at every generated position. Choose `α` and `l` on validation. This is the upper reference for what a single fixed direction can do without changing weights.

**R5: gradient-based context distillation (reference).**

- *R5-full:* LoRA, rank 16, on all linear layers. Off-policy variant: supervised on `(x, y_P)` for validated targets. On-policy variant (OEL-style): token-level reverse KL from the prompted teacher on the student's own samples. At most 3 epochs; learning rate from {1e-5, 5e-5, 1e-4} on validation.
- *R5-matched:* LoRA restricted to the **same layers and the same stored parameter count** as the best R1 edit. This isolates the effect of gradients from the effect of capacity, and is the key comparison for "what gradients buy."
- Report GPU minutes for each run.

**R6: matched random edits.** For each accepted R1, R2, or R3 edit, draw 5 random edits with the same layers, shapes, rank, and Frobenius norm, using random orthonormal factors. For R2, use random token pairs with the same row norms.

### 9.3 Size tiers

```text
tiny:    <= 0.001% of target parameters (stored)
small:   <= 0.01%
medium:  <= 0.1%
```

For every edit, report stored parameters, materialized dense values changed, affected layers, and `‖ΔW‖_F / ‖W‖_F` per matrix. R5-full is reported outside the tiers at its actual size.

### 9.4 Acceptance

An edit counts only if, on validation, it improves the primary metric over M0, does not break any guard (§11.1), and passes the validation slice of the preservation gate. Otherwise the route's result for that cell is **no edit**, with retention 0. No-edit outcomes are reported, not hidden.

## 10. Locality diagnostics

All diagnostics are computed on **discovery data only**, from the paired activations in §9.1. Their code and outputs are frozen and hashed before any compiled model is evaluated on validation. None of them uses gradients.

Let `m(·)` be the mean teacher-forced log-probability of the validated targets `y_P`. This is used only for diagnostics.

### 10.1 L1: layer concentration (primary predictor)

For each layer `l`, run the unprompted model on `(x, y_P)` and replace the residual stream at the `y_P` positions with the prompted values at layer `l` only. Recovery:

```text
r_l = (m(patched_l) − m(U)) / (m(P) − m(U))
```

L1 = sum of the three largest `r_l`, divided by the sum of all positive `r_l`. Higher means more concentrated.

Secondary summaries: `max_l r_l`, and the entropy of the normalized positive `r_l`.

### 10.2 L2: effective rank

At each layer, stack the per-position deltas `Δ_l = H_P,l − H_U,l` and compute the number of singular values needed for 90% of the squared Frobenius norm. L2 is the `r_l`-weighted mean over layers, normalized by hidden size. Lower means more concentrated.

### 10.3 L3: component concentration

Rank MLP neurons and attention heads by `|mean activation delta| × ‖output weight column‖`. Patch the top-k components from the prompted run into the unprompted run, for k ∈ {8, 32, 128, 512}. L3 is the smallest k reaching 50% recovery, in log scale. Lower means more concentrated.

### 10.4 L4: token concentration (B1 and B2)

The share of the log-probability gain `m(P) − m(U)` located at positions where the gold token is an archaic form (B1) or the target word (B2).

### 10.5 L5: direction consistency

The mean pairwise cosine similarity between item-level mean deltas `mean_positions Δ_l` at the best L1 layer. Higher means the prompt pushes every item in the same direction.

### 10.6 Analysis units for H3

With only two or three behaviors and two models, a correlation across behaviors has at most six points. To make H3 testable, retention and diagnostics are both computed per **item cluster**:

- **B1:** clusters by archaic pattern type (*-ick/-ic*, *-our/-or*, *-eth*, *ye/the*, doubled consonants, *-ed/-'d*, other), crossed with OCR CER level.
- **B3:** clusters by OCR noise tercile, crossed with answerable or unanswerable.
- **B2:** clusters by target word.

Each cluster with at least 15 validation items and 15 test items is one unit. Expect about 20–30 units per model. H3 is tested on units with the primary predictor L1, with a permutation test that shuffles retention across units within each model. Behavior-level points are shown as well, but the claim rests on the unit-level test.

L2–L5 are secondary, Holm-corrected.

## 11. Metrics

### 11.1 Behavior metrics

**B1** (token-level, after alignment of output to gold):

| Metric | Definition | Role |
|---|---|---|
| ModErr | valid archaic tokens output in modern form ÷ valid archaic tokens | **Primary** (lower is better) |
| HCPR | valid archaic tokens output exactly as gold ÷ valid archaic tokens | Secondary |
| AIR | tokens archaized where gold is modern ÷ all tokens | Guard: must not rise by more than 1 point |
| OCR-fix recall | OCR-error tokens corrected to gold ÷ OCR-error tokens | Guard: must not fall by more than 2 points |
| CER / WER | against diplomatic gold | Secondary |

**B3:**

| Metric | Definition | Role |
|---|---|---|
| Evidence F1 | token F1 between quoted span and gold evidence sentence | **Primary** |
| Answer EM / F1 | standard SQuAD-style | Guard: F1 must not fall by more than 2 points |
| Unsupported-claim rate | answers given on unanswerable items | Secondary |
| Clean/OCR consistency | answer agreement between corrected and raw-OCR contexts | Secondary |

**B2:** accuracy on sense labels (primary).

### 11.2 Retention

```text
Retention = (Score(Mc) − Score(M0)) / (Score(M0 + S) − Score(M0))
```

computed on the primary metric (sign-adjusted for ModErr), with paired bootstrap CIs (10,000 resamples). Retention is reported only where the prompt gain's lower CI bound is above zero.

### 11.3 Transfer

Retention on: public test (cross-work), cross-corpus test, cross-period test (B3), counterfactual test (B1; seen-pattern and unseen-pattern subsets separately).

### 11.4 Preservation gate

On the final preservation suite:

- mean absolute change across MMLU, GSM8K, and IFEval ≤ 1 point;
- no single component drops by more than 3 points;
- false-archaism rate on modern spelling correction rises by no more than 1 point;
- on normalization-on-request items, the rate of archaic tokens left unmodernized despite the instruction rises by no more than 5 points.

These tolerances are *pilot* values, frozen after the Phase 0 variance check.

### 11.5 Cost

For each route: forward passes, linear-solve wall time, GPU hours, editor tokens and dollars (R3), LoRA GPU minutes (R5), stored and materialized edit size, and runtime prompt tokens saved per query. The serving break-even point `N = C_compile / (C_prompt − C_edit)` is reported for completeness, but it is not a headline claim.

## 12. Run matrix

Per model family and behavior:

| Block | Cells |
|---|---|
| Sources at inference | M0, S1, S2, S3, S4 (self, teacher), S5 |
| Gradient-free compilation | R1 × {S1, S2, S4} × 3 tiers; R2 × S1 × 3 tiers (B1 only); R3 × S1 × 3 trajectories |
| References | R4 × S1; R5-full × S1 (off-policy and on-policy); R5-matched × S1 at the best R1 tier |
| Controls | R6 × 5 for every accepted edit |
| Complementarity | best compiled model + S1 at inference |

Rough compute estimate for both families, B1 and B3 (to be checked in Phase 0): evaluation and activation collection on the order of 100–200 A100-hours; LoRA under 20 A100-hours; TF-GRPO a few thousand rollouts plus a few hundred updater calls; R3 editor cost capped per trajectory and dominated by API spend.

## 13. Statistical protocol

- Every arm sees the same items with the same decoding, so all comparisons are paired.
- Paired bootstrap 95% CIs for every metric and retention value.
- Exact McNemar tests for item-level repairs and breaks.
- Three seeds or trajectories for every stochastic route (TF-GRPO, R3, R5, R6); closed-form routes are deterministic given data, so they are repeated over three discovery resamples instead.
- Holm correction within each family of secondary comparisons.
- Effect sizes with CIs are reported alongside every p-value.
- Prompts, rewards, diagnostics, thresholds, and splits are frozen and hashed before the final run. Each test set is run once; anything after that is labeled post hoc.

## 14. Phase 0 kill test (executable checklist)

Week 1:

1. Shortlist 3–4 candidate targets; check tokenization of 200 archaic/modern pairs.
2. Build the B1 pilot: 200 ECCO-TCP + OCR (or synthetic OCR) items and 50 counterfactual items.
3. Build the B3 pilot: 150 ChroniclingAmericaQA items with gold evidence.
4. Freeze S1 for B1 and B3.

Week 2:

5. Run C0, C1 (S1), and C2 (S2) on both pilots for every candidate; compute ModErr, guards, and evidence F1 with bootstrap CIs.
6. Pick two families by H1a headroom and tokenization.
7. Implement R1 on one and three layers for B1 on family A; measure retention on 50 held-out pilot items.
8. Measure run-to-run variance of the preservation-suite validation slice to set the §11.4 tolerances.
9. Write the go/no-go memo with all numbers and any threshold revisions.

**Go** if H1a, H1b, and H1c pass and R1 shows any retention above the matched random edit on B1. **No-go actions** are listed in Plan §8, Phase 0.

## 15. Quantization robustness

For the best edit per route and behavior, compare: full-precision edited model; edit-then-quantize (8-bit and 4-bit); and, where valid, an edit applied to the quantized base. Report retention and the preservation gate again. Loss of the edit under quantization is reported as a finding about edit size.

## 16. Reproducibility and artifact contract

Every run stores:

```text
run_id, git_commit
base_model_id, revision_hash, tokenizer_hash, chat_template_hash
dataset_manifest_hash, split_hashes
source_prompt_version (S1-S5, E* history)
diagnostic_outputs_hash (L1-L5)
route, tier, hyperparameters
changed_tensor_hashes or low-rank factors
updater/editor identity, prompts, token counts, cost
decoding_configuration
per-item outputs and metrics
repair and break lists
compute log
```

Released when licenses permit: evaluation code; data manifests and derived annotations; S1–S5 and `E*` histories; diagnostic code and outputs; edit factors (not restricted source text); R3 editor logs; random-edit generators; no-edit and failed runs.

## 17. Paper tables and figures

1. Headroom audit: ModErr and evidence F1 by view and prompt condition (C0–C2), both families.
2. Source comparison S1–S5 at inference.
3. Main result: retention by route (R1–R6) and tier, for B1 and B3, both families.
4. **Central figure:** gradient-free retention against L1 per analysis unit, both families, with R5 retention shown as a reference band.
5. Gradients versus capacity: R1 versus R5-matched versus R5-full.
6. B1 edit localization: which layers, directions, and rows carry the edit; R2 seen-pair versus unseen-pair retention.
7. Transfer: cross-corpus, cross-period, counterfactual.
8. Preservation and false archaisms; normalization-on-request.
9. Cost and edit size.
10. Quantization robustness.
11. No-edit outcomes and R3 trajectories.
