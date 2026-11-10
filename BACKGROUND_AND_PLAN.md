# HisFree: Where Does Historical Reading Live?

**Compiling historical reading behaviors from context into weights without gradients, and predicting when it works**

**Version:** 0.3 (revised 10 November 2026 after a novelty and feasibility review; 11 November 2026 adds causal-necessity, factorial-steering, and edit–prompt alignment analyses, and an optional SAE module)

**Companion document:** [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) v0.3 gives the executable design (data construction, route specifications, diagnostics, metrics, thresholds, and the Phase 0 checklist).

## 0. What changed from v0.1 and why

v0.1 framed HisFree as a pipeline: Training-Free GRPO discovers a historical reading prior, Weight-Space Programming (WSP) then compiles it into a sparse edit, and the result is evaluated across text, OCR, page images, many tasks, and many model families. A review found four problems with that framing.

| Problem in v0.1 | Evidence | Change in v0.2 |
|---|---|---|
| "Learn an experience prior, then compile it into weights" was presented as open | Online Experiential Learning (Ye et al., 2026) and EDGE (Xie et al., EMNLP 2026) already distill extracted experience into weights; Prompt Baking and context distillation already move prompt effects into weights | The novelty claim is narrowed to **gradient-free** compilation and to **predicting compilability**. Gradient-based context distillation becomes a reference upper bound rather than a missing baseline |
| No convincing reason to avoid gradients | A LoRA or context-distillation run on ~100 examples costs minutes on one GPU for a 4B–9B model; a frontier-model WSP editor probably costs more | Section 3.7 states what gradient-free compilation must buy (inspectability, edit size, attribution) and drops "cheaper than training" as a claim |
| The core hypothesis was likely to fail as stated | Published WSP successes repair a single localized pathology; a 600-token conditional reading strategy is probably distributed | The paper now studies a **spectrum of behaviors** chosen to differ in predicted locality, so both success and failure are informative |
| Scope far beyond a 12-week project | 3 tasks × 4 views × 9 arms × 7 extra baselines × 5 trajectories × several model families, plus multimodal and a private blind set | One primary behavior, one or two contrast behaviors, two model families, text only. Multimodal work moves to a follow-up paper |

Training-Free GRPO remains in the project, but as one **source** of prompt behavior to compile, not as the central method. That avoids a paper that reads as "method A plus method B applied to domain C."

## 1. Project summary

Modern open language models can often behave like careful historical readers when prompted: preserve archaic spellings instead of modernizing them, quote evidence from a noisy OCR passage before answering, or read a word in its period sense. Without the prompt, they frequently do not. The prompt therefore unlocks a behavior that the weights already support.

HisFree asks:

> For a historical reading behavior that a model already exhibits under a prompt, can that behavior be moved into the weights without gradients, and which measurable properties of the behavior predict whether it can?

The working thesis is:

> Behaviors whose prompt effect is concentrated in a few directions and layers, such as suppressing a learned modernization prior, can be compiled into small gradient-free edits that retain most of the prompt's gain. Procedural behaviors whose prompt effect is spread across many layers and steps, such as evidence-first answering, cannot be compiled this way and belong in context or retrieval. Gradient-based context distillation can compile both, which isolates what is lost by removing gradients.

This thesis is falsifiable in both directions. If locality does not predict retention, or if gradient-free compilation matches gradient-based distillation everywhere, the paper reports that instead.

### 1.1 Why historical text

Historical text is used as a **controlled testbed**, not as the whole contribution.

- **Paired views.** The same content exists as normalized text, diplomatic transcription, raw OCR, and synthetically corrupted OCR. That makes it possible to hold meaning fixed and vary only form, which few other domains allow.
- **A clean "knows but does not express" case.** Models read *shew*, *publick*, *compleat*, and long-s forms correctly, yet they rewrite them as modern spellings when transcribing or correcting. The knowledge is present and the output prior suppresses it. That is exactly the kind of failure an edit could plausibly remove.
- **Deterministic metrics.** Historical-form preservation, modernization-error rate, CER against a diplomatic gold, and evidence-span F1 can all be computed without an LLM judge.
- **A useful application.** Digital-humanities users need models that do not silently modernize evidence. A compiled model that preserves historical forms without a long system prompt is directly usable.

## 2. Terminology

**Gradient-free compilation.** A persistent parameter change computed without backpropagation through the target model and without an optimizer. Allowed computations are forward passes, closed-form solutions (least squares, ridge regression, projections), and edits proposed by an external reasoning model and checked with forward passes. This is not data-free or compute-free. All required diagnostics are forward-only as well; the only gradients anywhere in the project are in the gradient-based reference (R5) and in an optional attribution-assisted SAE analysis. In the paper, the precise term is **backprop-free, optimization-free weight compilation**.

**Behavior source.** The prompt or context that induces the target behavior: a manual instruction, few-shot demonstrations, a directly generated instruction, or a Training-Free GRPO experience prior `E*`.

**Prompt gain.** `Gain_P = Score(model + prompt) − Score(model)` on held-out data.

**Retention ratio.** For a compiled model `Mc` evaluated **without** the prompt:

```text
Retention(Mc) = (Score(Mc) − Score(M0)) / (Score(M0 + prompt) − Score(M0))
```

Retention is reported only where the prompt gain is significantly positive.

**Compilability.** The best retention a compilation route achieves on a behavior under a fixed edit-size budget, subject to the preservation gate.

**Locality diagnostics.** Measurements, taken before any edit, of how concentrated a prompt's effect is inside the model. They are defined in Section 6.4 and serve as candidate predictors of compilability.

## 3. Background and related work

### 3.1 Historical language and document models

Existing work mostly builds historical capability through training.

- **Pretraining from scratch.** MacBERTh (English, 1450–1950) showed that historical pretraining helps downstream tasks. TypewriterLM (Luo et al., EMNLP 2026) is a 7.24B decoder trained on a 54B-token pre-1913 corpus with temporally aligned instruction tuning. Its general benchmarks remain far below a modern model of similar size (for example 43.3 versus 77.4 for Qwen3-4B on ARC-Easy), which shows the cost of strict temporal grounding.
- **Continued pretraining.** Historical BERT variants improve after period-specific masked-LM training. This works, but needs gradients and risks forgetting.
- **Pixel and VLM models.** PHD (EMNLP 2023) adapts a pixel encoder to historical newspapers. CHURRO (EMNLP 2025) trains a 3B VLM on 99,491 pages from 155 corpora and outperforms frontier systems on historical transcription. HisDoc-OCR (ACL Findings 2026) diagnoses language-prior domination in Chinese historical OCR and repairs it with training.
- **Benchmarks.** ChroniclingAmericaQA, HIPE-2022, SemEval-2020 Task 1, AncientDoc, and ProHist-Bench cover QA, entity linking, semantic change, OCR-to-reasoning, and professional historical reasoning.

HisDoc-OCR is the closest diagnosis to ours. It finds that a strong language prior overrides visual evidence, which is the multimodal analogue of the modernization prior studied here. It repairs the problem with training, so it does not address gradient-free compilation.

None of these works asks whether a prompted historical behavior can be moved into weights without gradients. Building a historical model is not novel by itself, and this project does not claim it.

### 3.2 Model editing and activation interventions

- **Locate-and-edit.** ROME and MEMIT make closed-form rank-one or multi-rank MLP edits for factual associations. AlphaEdit projects edits into a null space to protect other knowledge.
- **Learned editors and memories.** MEND trains a hypernetwork. SERAC and WISE keep corrections in a separate memory.
- **Activation steering.** Activation addition and related methods change hidden states at inference without persistent weight changes. Steering is the natural non-persistent reference for a compiled edit.
- **Sparse-autoencoder feature analysis.** OpenAI's emergent-misalignment work and its follow-ups find behavior-driving SAE latents by model diffing and by paired attribution over completions with and without a behavior, then confirm them by steering in both directions; they also separate latents that drive a behavior from latents that protect against it. Their metagaming study combines several discovery methods (activation-frequency difference, gradient-direction similarity, attribution, checkpoint diffing) and checks whether they agree. HisFree borrows the validation logic (sufficiency, necessity, two-sided steering, driver versus suppressor) but keeps its required diagnostics forward-only. Public SAEs exist for every layer of Gemma 3 (Gemma Scope 2, including chat-tuned models) and of Llama-3.1-8B Base (Llama Scope).
- **Evaluation pitfalls.** RippleEdits shows that edits fail to propagate and that an in-context baseline is often best. *The Mirage of Model Editing* shows that teacher-forced evaluation inflates single-edit success from 38.5% to 96.8%. RULE-EDIT shows near-perfect direct reliability alongside under 52% portability. Large-scale sequential editing causes forgetting.

Factual editors target single associations, whereas HisFree targets **behaviors**. That distinction matters for the locality hypothesis: a modernization prior may look more like a set of token-level associations, and evidence-first answering does not.

### 3.3 Moving prompt or experience effects into weights (closest prior work)

This is the line of work the paper must position against most carefully.

| Work | Behavior source | Compilation | Gradients | Models and tasks | Relevance |
|---|---|---|---:|---|---|
| Context distillation (Snell et al., 2022) | Instructions and scratchpads | Fine-tune the student to match the prompted teacher | yes | Various LMs | Original prompt-to-weights idea |
| Prompt Baking (Bhargava et al., 2024) | Any prompt | Minimize KL between prompted and baked model | yes | Instruction and reasoning prompts | Same goal, gradient-based |
| Online Experiential Learning (Ye et al., 2026) | Experience extracted by the model from its own trajectories | On-policy context distillation with reverse KL | yes | Qwen3 1.7B–8B on Frozen Lake and Sokoban | Shows **experience** can be consolidated, sometimes exceeding the in-context gain (e.g. Sokoban 18.2 in context → 21.4 consolidated) |
| EDGE (Xie et al., EMNLP 2026) | Retrieved experiences, split into conditioned and free rollouts inside a GRPO group | Reverse-KL distillation inside RL training | yes | Embodied, web, and search QA agents | Closest to the v0.1 pipeline; up to +12.5 points over RL baselines |
| Transmuting Prompts into Weights (Mazzawi et al., 2025/2026) | Prompts and in-context examples | Closed-form least-squares "thought matrices" on MLP up/gate projections | no | Gemma 3 1B: arithmetic, detox, translation, capitals, synthetic dictionaries | The main gradient-free compiler to build on |
| Distilling Instructions into Weights (Abbas and Seraj, 2026) | A known-good rewording | Closed-form MLP edit | no | Robot policy, 27/128 → 106/128 held-out success | Works only when the skill already exists and a working instruction is known |
| WSP and follow-ups (2026) | Diagnosed failures | Edits proposed by a frontier reasoning model | no | Qwen3.5 0.8B and up, robot policies, small CNNs | Successes are localized repairs; many sessions correctly decline to edit |

Two observations shape the project.

1. **Experience-to-weights is no longer open as such.** OEL and EDGE already do it with gradients, and OEL reports consolidated performance at or above the in-context gain. A paper whose main claim is "we compile experience into weights" would be incremental.
2. **Gradient-free compilation has uneven retention, and the pattern is unexplained.** In Transmuting Prompts, arithmetic retains 100% of the prompted score, detoxification and translation retain most of it, but Country→Capital drops from 79.0 with context to 47.6 when patched. No published work explains which properties of a prompt effect make it compilable without gradients, or tests this at 4B–9B scale on natural-language behaviors.

The open question HisFree targets is therefore the second one.

### 3.4 What Weight-Space Programming has and has not shown

*Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, and Khattab, 2 November 2026) reports that a frontier model edited 0.016% of Qwen3.5-0.8B to remove a reasoning-loop pathology, raising MATH-500 from 7.9% to 48.1% under the reported sampling setup. The authors say WSP falls short when the problem is not a single localized fault.

The first follow-ups support that caution.

- **WSP-Scope** (Hu et al.) ran 16 trajectories on seven benchmarks and two Qwen3.5 scales. Gains reached 29.45 points from 1,023 parameters, but transfer was uneven, and 6 of 16 trajectories correctly kept the original weights.
- **Robot policy repair** (Zhang et al.) raised held-out success from 86.3% to 92.0% by editing 65 token-embedding rows, but 12 of 13 further sessions declined to edit. The authors conclude the evidence supports lexical repair, not new skills.
- **WeightLoop** (Li et al.) turns WSP into a reusable editing program driven by experimental feedback, in a small prediction setting.
- **Medical imaging WSP** (Lindholz et al.) found LLM-written CNNs above chance, but Adam on 256 labeled images beat them on every MedMNIST task.
- **A weight-reading audit** (Ahmed) found that apparent weight understanding is often local arithmetic plus dataset priors.

Implication: WSP is a credible tool for removing a **lexical or localized** obstruction, which is why B1 below is the primary target. It is not evidence that a distributed procedure can be written into weights. The robot-policy result, a lexical repair through embedding rows, is the closest analogue to suppressing modernized spellings.

### 3.5 Training-Free GRPO

Training-Free GRPO (Cai et al., 2025) keeps the model frozen. For each query it samples a group of outputs, compares them, and asks an LLM to state the semantic advantage of the better outputs as natural-language experience, which accumulates into a prompt prior. It reports gains on math and web search with tens to about one hundred examples, using large policy models such as DeepSeek-V3.1.

In v0.2 Training-Free GRPO is one behavior source among several. Its role is to test a secondary question: does a group-discovered experience, which tends to be longer and more conditional than a manual instruction, compile better or worse than a short manual instruction with a similar prompt gain? Its effectiveness on 4B–9B targets is not established, and a self-updater may produce weak experience. A teacher-updater is allowed but must be reported as such.

### 3.6 Progress of the field, 2021–2026

| Period | Representative progress | What is now established | Remaining gap for HisFree |
|---|---|---|---|
| 2021–2023 | MacBERTh, historical continued pretraining, PHD | Historical pretraining helps; page pixels can be modeled | All gradient-trained |
| 2022–2024 | Context distillation, Prompt Baking, ROME/MEMIT | Prompt effects can be moved into weights with gradients; factual associations can be edited in closed form | Behavior-level, gradient-free compilation is untested |
| 2025 | CHURRO, Training-Free GRPO, Transmuting Prompts | Specialized historical VLMs; frozen-model experience learning; closed-form prompt-to-weight on a 1B model | Retention varies by task with no explanation |
| 2026 (H1) | TypewriterLM, OEL, HisDoc-OCR, AncientDoc, ProHist-Bench | Billion-scale history LMs; experience consolidation into weights with gradients; language-prior domination diagnosed in historical OCR | No gradient-free compilation of historical behaviors |
| 2026 (Oct–Nov) | EDGE, WSP, WSP-Scope, robot WSP, WeightLoop | Experience distillation inside RL; sparse frontier-model edits sometimes repair localized failures | No account of which behaviors are compilable without gradients |

### 3.7 Why compile without gradients at all

This is the first question reviewers will ask, and the paper must answer it directly. The honest answer is **not** "it is cheaper than training": for a 4B–9B model and about 100 examples, LoRA or context distillation is cheap, and a frontier-model WSP editor may cost more.

What gradient-free compilation can offer, and what the paper will measure:

1. **Attribution.** A closed-form or agentic edit is written in terms of named components: specific embedding rows, a rank-k update to specific MLP layers, specific directions. If a small edit retains most of the prompt's gain, that localizes where the behavior lives. A gradient-trained LoRA spread across all layers does not give this answer.
2. **Edit size and composability.** If gradient-free edits are orders of magnitude smaller than a LoRA with similar retention, they can be stored, inspected, reverted, and combined per behavior.
3. **No training stack.** Closed-form compilation needs only forward passes and a linear solve, so it can run where fine-tuning infrastructure is unavailable. This is a secondary practical point, not a headline.
4. **A scientific probe.** The gap between gradient-free and gradient-based retention on the same behavior measures how much the behavior depends on distributed, jointly optimized changes. That gap is the quantity the locality hypothesis predicts.

Gradient-based context distillation (LoRA, OEL-style on-policy distillation) is therefore a required **reference**. The paper reports retention for both, plus edit size and cost, and does not claim gradient-free compilation is better overall.

### 3.8 Novelty verdict (revised)

| Claim | Verdict | Reason |
|---|---|---|
| "We build a historical language model" | Not novel | MacBERTh, PHD, CHURRO, TypewriterLM |
| "We compile a learned experience prior into weights" | Not novel | OEL, EDGE, Prompt Baking, context distillation |
| "We apply WSP or Training-Free GRPO to historical text" | Weak alone | An application, likely to be crowded by the WSP wave within months |
| "Gradient-free compilation of historical reading behaviors at 4B–9B scale" | Moderate | New setting for Transmuting-Prompts-style compilers; needs the comparison against gradient-based distillation to be meaningful |
| "Measurable locality of a prompt's effect predicts gradient-free compilability across behaviors and model families" | **Strongest; currently open** | No located work explains why retention varies (e.g. capitals versus arithmetic in Transmuting Prompts) |
| "Modernization bias in historical transcription is a localized output prior that a tiny edit can remove" | Strong, specific, and checkable | Connects to HisDoc-OCR's language-prior diagnosis and to the lexical-repair evidence from robot WSP |
| "Procedural historical reading cannot be compiled without gradients and belongs in context" | Strong if supported | Establishes a boundary rather than a headline gain |

The paper thesis:

> A historical reading behavior can be compiled into weights without gradients in proportion to how concentrated its prompt effect is. Modernization suppression is concentrated and compiles into a tiny edit; evidence-first answering is distributed and does not. Gradient-based distillation compiles both, and the gap between the two quantifies the cost of removing gradients.

### 3.9 Timing risk

WSP appeared on 2 November 2026, and more than seven follow-ups were on arXiv within a week. By the next ACL Rolling Review deadlines there will be many "WSP for domain X" papers. A paper whose contribution is applying WSP to a new domain is likely to be scooped or judged incremental. The locality-predicts-compilability framing does not depend on being first to apply WSP, which makes it more robust to this competition. The schedule in Section 8 is still built to produce a submittable result quickly.

## 4. Research questions and hypotheses

### RQ1: Is there prompt-accessible headroom?

**H1.** On diplomatic transcription and correction of historical text, a current 4B–9B open model makes frequent modernization errors without a special prompt, and a short instruction removes a large share of them. Under raw OCR, evidence-first prompting improves evidence-span F1 in historical QA.

This is the kill test. Without prompt headroom there is nothing to compile.

### RQ2: How much prompt gain survives gradient-free compilation?

**H2.** For the primary behavior B1, at least one gradient-free route retains at least 50% of the prompt gain at an edit size of at most 0.1% of parameters, while passing the preservation gate. For the contrast behavior B3, gradient-free retention is substantially lower than for B1, while gradient-based distillation retains most of the gain for both.

### RQ3: Does locality predict compilability?

**H3.** Locality diagnostics measured **before** editing (layer concentration of the prompt effect, effective rank of the prompted-minus-unprompted activation delta, and the share of the effect recovered by patching the top-k components) rank behaviors in the same order as their gradient-free retention, across at least two model families.

Pre-registered predicted order: B1 (modernization suppression) > B2 (period word sense, optional) > B3 (evidence-first answering).

H3 is tested on a **prediction track** whose compilation configuration never reads the diagnostics, so that a diagnostic is never used both to choose an edit and to predict its success. A separate **routing track** lets diagnostics choose layers, and tests whether that improves compilation.

A further exploratory question is *how* the B1 prompt acts: does it activate a historical-preservation direction, suppress a default modernizing tendency, or gate between the two depending on context? Section 6.6 specifies the experiments; their outcome is reported as a candidate mechanism, not as a claim.

### RQ4: Does the behavior source matter?

**H4.** For prompts with matched prompt gain, a short manual instruction compiles with higher retention than a longer Training-Free GRPO experience prior, because the experience prior encodes more conditional, procedural content. This is secondary and may be dropped if Training-Free GRPO does not produce a prompt gain on the target model.

### RQ5: Does a compiled edit transfer and preserve general ability?

**H5.** A compiled B1 edit transfers to an unseen corpus and an unseen period, beats norm-, layer-, and size-matched random edits, and stays within the general-regression tolerance.

## 5. Behaviors under study

Behaviors are chosen to span the predicted locality spectrum, so that both positive and negative compilation results are informative.

### B1 (primary): Historical-form preservation and modernization suppression

- **Task.** Given a historical passage with OCR errors, output a corrected diplomatic transcription that fixes OCR errors but keeps valid period spellings, long s where the target convention keeps it, archaic morphology (*-eth*, *-ick*), and period capitalization.
- **Failure.** The model fixes OCR errors and also modernizes valid forms (*shew* → *show*, *publick* → *public*).
- **Prompt that should fix it.** An instruction to preserve attested historical forms and correct only clear OCR errors, optionally with a few demonstrations.
- **Why it is predicted to be local.** The failure looks like an output prior over a closed set of token pairs, which can live in embedding or unembedding rows or in a few late MLP directions. Robot WSP repaired a similar lexical issue through 65 embedding rows.
- **Metrics.** Modernization-error rate on valid archaic tokens; Historical Character Preservation Rate; Archaic Insertion Rate (false archaisms); CER and WER against the diplomatic gold; OCR-fix recall, so that a model that simply copies input cannot score well.

### B3 (primary contrast): Evidence-first answering under historical noise

- **Task.** Historical QA on raw-OCR passages (ChroniclingAmericaQA and held-out ECCO material). The output is an answer plus a quoted supporting span.
- **Failure.** The model answers from modern background knowledge or paraphrases without grounding, especially when OCR noise is heavy.
- **Prompt that should fix it.** An instruction to locate and quote the supporting text before answering, and to say when the passage does not support an answer.
- **Why it is predicted to be distributed.** The behavior is multi-step, depends on the input, and involves attention over the passage rather than a fixed output prior.
- **Metrics.** Answer EM and F1; evidence-span F1; unsupported-claim rate; consistency between clean and OCR views.

### B2 (optional middle point): Period word sense

- **Task.** Given a passage with a date and a target word whose sense has shifted (for example *nice*, *awful*, *silly*, *gay*), choose or describe the period-appropriate sense.
- **Prompt.** An instruction to interpret words as a reader of the stated year would.
- **Data.** A small instance-level set built from COHA or ECCO contexts with sense annotations, guided by SemEval-2020 Task 1 target words. SemEval itself scores graded change over corpora, so it is not used directly.
- **Status.** Included only if B1 and B3 are on schedule after Phase 2. It adds a third point on the locality curve.

## 6. Method

### 6.1 Stage A: headroom audit

For each target model, evaluate B1 and B3 on matched views:

```text
normalized text
diplomatic transcription
raw OCR
synthetic OCR (controlled corruption of diplomatic text)
```

under three prompt conditions:

```text
C0  no special instruction
C1  short manual behavior instruction
C2  manual instruction + few-shot demonstrations
```

Report prompt gain with paired bootstrap confidence intervals. A behavior proceeds only if `Gain_P` is significant and practically meaningful (thresholds in Section 9).

### 6.2 Behavior sources

| ID | Source | Notes |
|---|---|---|
| S1 | Short manual instruction | Primary source for all behaviors |
| S2 | Manual instruction + few-shot demonstrations | Tests whether demonstrations compile differently from instructions |
| S3 | Directly generated instruction of matched length | Control for prompt length and wording |
| S4 | Training-Free GRPO experience prior `E*` | Secondary; self-updater and teacher-updater reported separately |
| S5 | Shuffled or unrelated text of matched length | Negative control for compilation of "any long prompt" |

The compilation target is always the behavior induced by a source on discovery data, and only outputs that also score well on the task metric are used as compilation targets. The prompted model is a teacher, not ground truth.

### 6.3 Compilation routes

All routes are evaluated without the prompt at inference.

| ID | Route | Gradients | Edit space | Role |
|---|---|---:|---|---|
| R1 | Closed-form prompt-to-weight (Transmuting Prompts style) | no | MLP up/gate (and down for bias) in selected layers; rank capped | Main gradient-free compiler |
| R2 | Closed-form token-row edit | no | Selected input-embedding and/or unembedding rows for identified token pairs | Tests the lexical-prior hypothesis for B1 directly |
| R3 | Agentic WSP | no | Declared mask; editor sees paired prompted/unprompted traces and locality reports | Tests whether a reasoning model finds better edits than closed form; may return no edit |
| R4 | Activation steering from prompted-minus-unprompted means | no | Inference-time hidden-state addition (non-persistent) | Upper reference for what a fixed direction can do |
| R5 | Gradient-based context distillation (LoRA; OEL-style on-policy reverse KL) | yes | LoRA on all linear layers at a standard rank | Required gradient-based reference |
| R6 | Matched random edits | no | Same layers, rank, norm, and parameter count as R1–R3 | Control |

Edit-size tiers, reported for every route:

```text
tiny:    <= 0.001% of parameters (or equivalent stored factors)
small:   <= 0.01%
medium:  <= 0.1%
```

Both stored size and the number of materialized dense values changed are reported, since a low-rank update is not sparse once materialized.

R3 cost is reported in editor tokens and dollars next to R5's GPU time. If R3 does not beat R1 at matched size, the paper says so and treats R3 as a negative result rather than tuning it further.

### 6.4 Locality diagnostics (measured before editing)

On discovery data, for each behavior and model, compare runs with and without the prompt and compute:

1. **Layer concentration.** Activation-patch the prompted residual stream into the unprompted run one layer (or one block of layers) at a time, and measure how much of the behavioral gain each recovers. Summarize as the share recovered by the best `k` layers and as an entropy over layers.
2. **Effective rank of the prompt effect.** Singular-value spectrum of the stacked prompted-minus-unprompted activation deltas at each layer, summarized as the rank needed for 90% of the variance.
3. **Component sufficiency and necessity.** Rank MLP neurons and attention heads separately by activation delta times output-weight norm (no gradients). Sufficiency: the share of the gain recovered by patching the top-k prompted components into the unprompted run. Necessity: the share lost by ablating the same components in the prompted run. Sizes are normalized by the number of components of each type, so families with different widths are comparable.
4. **Token concentration (B1).** The share of the gain carried by a closed set of identified token pairs (archaic form versus modern form).
5. **Direction consistency.** How similar the prompt's push is across items.

These are computed on discovery data only, frozen, and then correlated with retention measured on validation and test data. The correlation is the central test of RQ3. One primary predictor (layer concentration or normalized component sufficiency) is chosen in Phase 0 by a fixed rule based only on measurement reliability. The rest are secondary. No composite score of several diagnostics is used.

### 6.6 Mechanism analyses

These explain results and are not success criteria.

- **Factorial steering (B1).** Crossing prompt (none, S1), steering along the prompt direction (none, positive, negative), input (historical, modern), and instruction (default, "modernize the spelling") separates a single causal direction, a context-gated mechanism, a global spelling bias, and a mainly suppressive mechanism. Whether a suppressive mechanism favors token-row edits (R2) and a driving one favors MLP edits (R1) is tested, not assumed.
- **Edit–prompt alignment.** For each accepted edit, compare the edit-induced activation change with the prompt-induced change, layer by layer. High alignment with good generalization indicates that the edit replicates the prompt's mechanism; low alignment with good generalization indicates functional substitution; low alignment with gains only on seen token pairs indicates a shallow shortcut. Alignment is not required for success.
- **Optional SAE module.** Only on Gemma 3 4B-IT, only for B1 and B3, at 2–3 layers fixed in advance, using Gemma Scope 2 SAEs. It repeats sufficiency, necessity, and factorial steering in the SAE basis, and adds paired latent attribution as a gradient-using reference, to test whether the forward-only diagnostics predict compilability as well as an attribution-assisted one. It is not part of any core claim.

### 6.5 Retrieval boundary

Compilation targets reusable reading behaviors, not facts. Long-tail entities, page-specific claims, and bibliographic evidence stay in retrieval. If a behavior's prompt gain turns out to come from knowledge carried in the prompt (for example, period spellings listed explicitly in demonstrations), it is reclassified as knowledge, and a retrieval baseline is added for it.

## 7. Evaluation protocol

### 7.1 Required comparisons

For each behavior and model:

- base model (M0);
- base + each behavior source (S1–S5) at inference;
- each compilation route (R1–R6) without the prompt;
- compiled model + prompt, to test whether compilation and context are complementary.

Historical pretrained models (TypewriterLM, CHURRO) appear only as reference points on compatible B1/B3 subsets, not as competitors. The paper's claim concerns compilation, not the best historical model.

### 7.2 Metrics

1. **Behavior score.** Task metrics from Section 5, reported per component.
2. **Retention ratio.** Primary quantity, with bootstrap confidence intervals.
3. **Preservation.** Change on a fixed general suite (e.g. MMLU subset, GSM8K subset, IFEval, and a modern spelling-correction set, which catches an edit that archaizes modern text). Report average change, worst single-task change, and repaired/broken counts.
4. **Edit size.** Stored parameters, materialized changed values, affected layers, norm.
5. **Compilation cost.** Forward passes, linear-solve time, editor tokens (R3), GPU time (R5).
6. **Transfer.** Cross-work, cross-corpus, cross-period, and cross-view retention.
7. **Robustness.** Retention after 8-bit and 4-bit quantization of the edited model.

### 7.3 Statistical requirements

- Free generation only; never teacher-forced likelihood as the success measure.
- Paired bootstrap confidence intervals; exact McNemar tests for paired binary outcomes; Holm correction across secondary comparisons.
- At least three seeds or independent trajectories per route in the main runs.
- Locality–retention correlation reported with Spearman's ρ and a bootstrap interval over items, plus a permutation test over behaviors and models. With few behaviors, the analysis is also reported per behavior at the item-cluster level so that it does not rest on three points.
- All splits, prompts, diagnostics, and thresholds frozen before the test run; the test run happens once.
- All tensors hashed; full edit logs and no-edit outcomes retained.

### 7.4 Success criteria

The thesis is supported if all of the following hold:

1. B1 has a gradient-free route with retention ≥ 50% at the small or medium tier, passing preservation, and beating matched random edits.
2. B3's best gradient-free retention is significantly lower than B1's, while R5 retains a comparable share for both.
3. The pre-registered locality diagnostics rank the behaviors in the same order as gradient-free retention on both model families.
4. The B1 edit transfers to at least one unseen corpus or period.

Partial outcomes are still reportable:

- If (1) holds but (2) does not, the paper reports broad gradient-free compilability of historical behaviors, which is a stronger positive result.
- If (1) fails while R5 succeeds, the paper reports that even a lexical-looking behavior needs gradients, and the diagnostics explain why.
- If prompt headroom is absent (Gate 1), the project pivots before investing in compilation.

## 8. Plan

Timeline assumes one or two people and access to one or two 80GB GPUs. Target the next feasible ACL Rolling Review cycle; check the current ARR deadlines before fixing dates.

### Phase 0: kill test (weeks 1–2)

Minimal, cheap experiments that decide whether the project is viable.

1. Pick two main targets from different families in the 4B–9B range with open weights and a standard tokenizer. Gemma 3 4B-IT is the preferred family A because public SAEs cover every layer; it is kept if it shows B1 headroom.
2. Build a 200-item B1 pilot from ECCO-TCP diplomatic text aligned with ECCO OCR or BLN600, split by work. Add 50 synthetic counterfactual items (unseen modern sentences converted with documented period spelling rules) to check contamination.
3. Build a 150-item B3 pilot from ChroniclingAmericaQA raw-OCR records.
4. Run C0/C1/C2 on both pilots for both models.
5. Implement R1 on one or two layers for B1 and measure retention on 50 held-out items.
6. Measure the split-half reliability of the candidate primary predictors on two pilot resamples, and fix the primary predictor by the preregistered rule.

**Pass:** B1 modernization-error rate under C0 is high enough to matter (initial threshold: ≥ 10% of valid archaic tokens modernized); C1 reduces it by ≥ 30% relative; R1 retains ≥ 20% of that gain at any tier. B3 shows a significant C1 gain in evidence F1.

**If B1 fails headroom:** try a smaller or older model, or switch the primary behavior to period word sense (B2). **If B3 fails headroom:** choose another procedural contrast (for example "flag uncertain readings").

**Deliverable:** a one-page go/no-go memo with the numbers.

### Phase 1: harness and reproduction (weeks 3–4)

- Reproduce one Transmuting Prompts result on Gemma 3 1B (e.g. a translation or detox task) to validate the R1 implementation.
- Implement R2, R4, R5, and R6, the locality diagnostics (including necessity), and the edit–prompt alignment measures.
- Implement the R3 editor harness: read, patch, revert, hash, and a bounded forward-pass API. Reproduce one released WSP edit if logs permit; otherwise document why not.
- Freeze the data manifests and the discovery/validation/test splits by work and edition family.

**Deliverable:** a validated harness and frozen splits.

### Phase 2: full B1 study (weeks 5–8)

- Full B1 data: about 100 discovery, 100 validation, 300 public test, and 300 held-out ECCO/newspaper test items.
- All sources S1–S5 and all routes R1–R6 at three size tiers, on the first model family.
- Locality diagnostics for B1; prediction-track and routing-track R1 runs.
- Factorial steering for B1.
- Preservation suite and quantization check for the best edits.
- Analysis of where the best gradient-free edit lives (rows, layers, directions).

**Deliverable:** the B1 result table and an edit-localization analysis.

### Phase 3: contrast behaviors and second family (weeks 9–11)

- B3 with the same pipeline; B2 if on schedule.
- Repeat B1 and B3 on the second model family.
- Locality–retention correlation analysis.
- Training-Free GRPO as source S4 on B1 and B3 (secondary; drop if it produces no prompt gain).
- Edit–prompt alignment for all accepted edits.
- Optional SAE module on Gemma 3 4B-IT (B1 and B3, 2–3 fixed layers), only if Phases 0–2 are on schedule.

**Deliverable:** the compilability-versus-locality figure, the paper's central result.

### Phase 4: transfer, robustness, and writing (weeks 12–14)

- Cross-corpus and cross-period transfer for the best B1 edit.
- Final held-out test run, once.
- Writing, with a limitations section on scale, language (English only), and number of behaviors.

**Deliverable:** a submission-ready paper.

## 9. Go/no-go gates

| Gate | Condition to proceed | If it fails |
|---|---|---|
| G1 Headroom | Significant, meaningful prompt gain on B1 (and B3) on held-out data | Change model or behavior; do not proceed to compilation |
| G2 Harness | R1 reproduces a published Transmuting Prompts effect; R5 reaches high retention on B1 | Fix the implementation before interpreting any null result |
| G3 B1 compilability | Some gradient-free route beats matched random edits on B1 validation | Report the negative result; shift emphasis to the gradient-free/gradient gap and the diagnostics |
| G4 Contrast | B3 runs complete on both families | Report B1 alone as a narrower localization paper (weaker; likely Findings) |
| G5 Predictors | Diagnostics frozen before retention is measured on validation and test | No post hoc diagnostics in the main claim |

## 10. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Modern models already preserve historical forms | No headroom for B1 | Phase 0 measures this in two weeks; fall back to B2 or older/smaller models |
| Public benchmark contamination | Prompt gain or retention reflects memorization | Counterfactual archaic sentences; work and edition-family splits; held-out ECCO/newspaper test |
| Gradient-free compilation fails on every behavior | No positive headline | Paper becomes "what gradients buy," supported by R5 and the diagnostics; still a contribution if the diagnostics explain the gap |
| Locality diagnostics do not predict retention | Central thesis fails | Report it; the B1 localization analysis and the gradient-free/gradient gap remain |
| Only two or three behaviors | Correlation claim is thin | Item-cluster analysis within behaviors (e.g. by token-pair frequency or OCR noise level); two families; B2 as a third point |
| Edit archaizes modern text | Damages general use | Modern spelling-correction set in the preservation suite; count false archaisms |
| WSP editor cost and opacity | Reproducibility concerns | R3 is not the main route; release prompts and logs; report cost next to R5 |
| Crowding by WSP follow-ups | Novelty erodes | The claim does not depend on being first to apply WSP; prioritize Phase 0–2 speed |
| ECCO/Gale licensing | Cannot release data or checkpoints | Use ECCO-TCP and BLN600 where possible; release IDs, hashes, derived annotations, and edit factors |
| Teacher-updater injects knowledge into `E*` | Confounds S4 | Report updater identity; compare with self-updater; S4 is secondary |
| Quantization erases small edits | Edits look fragile | Report post-quantization retention; treat failure as a finding about edit size |

## 11. Expected contributions

1. **A compilability analysis of historical reading behaviors.** Retention under gradient-free and gradient-based compilation for behaviors that span the locality spectrum, on two model families.
2. **Pre-edit locality diagnostics that predict gradient-free compilability.** A practical rule for deciding whether a prompted behavior can be moved into weights without gradients or should stay in context.
3. **A localized account of modernization bias.** Where in the model the prompt's effect on historical-form preservation lives, and a tiny edit that removes the bias without a system prompt.
4. **A controlled multi-view historical testbed.** Paired normalized, diplomatic, raw-OCR, and synthetic-OCR items with deterministic metrics, released where licenses allow.
5. **Full logs and negative results.** All edits, no-edit outcomes, random-edit controls, and costs.

## 12. Out of scope for this paper

- Multimodal page images and vision-projector editing (follow-up paper; HisDoc-OCR suggests the language-prior finding may transfer).
- Building a general historical foundation model or matching TypewriterLM.
- Injecting archive facts into weights.
- Retrieval quality and integration with a semantic-search system.
- Multilingual evaluation, except as a stretch goal if B1 transfers easily.

## 13. Venue strategy

- **EMNLP / ACL main (primary target).** A compilability analysis with a historical testbed and a clear negative/positive boundary fits the "analysis and interpretability" and "computational social science and cultural analytics" tracks.
- **ICLR / NeurIPS.** Only if the locality–compilability result holds beyond historical text. That would require adding non-historical behaviors and is not planned for the first paper.
- **Findings.** A realistic fallback if only B1 succeeds without the contrast and predictor results.

For CCF classification, verify the current official list. Main-conference long papers count; Findings, short papers, demos, and workshops do not.

## 14. Repository structure

```text
emnlp-hisfree/
├── README.md
├── BACKGROUND_AND_PLAN.md
├── EXPERIMENT_DESIGN.md
├── configs/        # models, behaviors, routes, budgets
├── data/manifests/ # provenance, license, split hashes
├── src/
│   ├── data/
│   ├── behaviors/  # B1, B2, B3 task definitions and metrics
│   ├── sources/    # manual prompts, few-shot, TF-GRPO
│   ├── compile/    # R1–R6
│   ├── locality/   # diagnostics
│   └── eval/
├── scripts/
├── results/
└── docs/
```

## 15. Selected references

### Historical language and document models

- Manjavacas Arevalo, E. and Fonteyn, L. (2021). [MacBERTh](https://aclanthology.org/2021.nlp4dh-1.4/).
- Borenstein, N. et al. (2023). [PHD: Pixel-Based Language Modeling of Historical Documents](https://aclanthology.org/2023.emnlp-main.7/).
- Semnani, S. et al. (2025). [CHURRO](https://aclanthology.org/2025.emnlp-main.1763/).
- Harju, A. and van der Goot, R. (2025). [How to age BERT Well](https://aclanthology.org/2025.loreslm-1.21/).
- Fittschen, E. et al. (2026). [Pretraining Language Models for Diachronic Linguistic Change Discovery](https://aclanthology.org/2026.findings-eacl.241/).
- Luo, X. et al. (2026). [A Language Model from 1913: Pretraining on Historical Text](https://aclanthology.org/2026.emnlp-main.2474/).
- Levchenko, M. A. (2025). [Evaluating LLMs for Historical Document OCR](https://aclanthology.org/2025.lm4dh-1.7/).
- Kanerva, J. et al. (2025). [OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches](https://aclanthology.org/2025.resourceful-1.8/).
- Thomas, M. et al. (2026). [Image-Informed Post-OCR Correction with Vision-Language Models](https://aclanthology.org/2026.findings-emnlp.60/).
- Cao, J. et al. (2026). [HisDoc-OCR](https://aclanthology.org/2026.findings-acl.301/).

### Prompt and experience to weights

- Snell, C., Klein, D., and Zhong, R. (2022). [Learning by Distilling Context](https://arxiv.org/abs/2209.15189).
- Bhargava, A. et al. (2024). [Prompt Baking](https://arxiv.org/abs/2409.13697).
- Ye, T., Dong, L., Dong, Q., et al. (2026). [Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856).
- Xie, C., Zhou, Y., Yang, W., et al. (2026). [EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning](https://aclanthology.org/2026.emnlp-main.624/).
- Mazzawi, H., Dherin, B., Munn, M., et al. (2025/2026). [Transmuting Prompts into Weights](https://arxiv.org/abs/2510.08734).
- Abbas, A.-A. and Seraj, E. (2026). [Distilling Instructions into Weights: Closed-Form Edits Fix a Robot Policy's Misreading](https://arxiv.org/abs/2611.11894).

### Model editing, steering, and weight-space programming

- Meng, K. et al. (2022). [ROME](https://arxiv.org/abs/2202.05262).
- Mitchell, E. et al. (2022). [MEND](https://openreview.net/forum?id=0DcZxeWfOPt) and [SERAC](https://proceedings.mlr.press/v162/mitchell22a.html).
- Meng, K. et al. (2023). [MEMIT](https://arxiv.org/abs/2210.07229).
- Ilharco, G. et al. (2023). [Task Arithmetic](https://arxiv.org/abs/2212.04089).
- Turner, A. M. et al. (2023). [Activation Engineering](https://arxiv.org/abs/2308.10248).

### Sparse autoencoders and feature-level analysis

- Gao, L. et al. (2024). [Scaling and Evaluating Sparse Autoencoders](https://cdn.openai.com/papers/sparse-autoencoders.pdf).
- He, Z. et al. (2024). [Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders](https://arxiv.org/abs/2410.20526).
- Wang, M. et al. (2025). [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823) ([blog](https://openai.com/index/emergent-misalignment/)).
- Dupré la Tour, T. and Mossing, D. (2025). [Debugging Misaligned Completions with Sparse-Autoencoder Latent Attribution](https://alignment.openai.com/sae-latent-attribution/).
- Dupré la Tour, T. (2025). [Helpful Assistant Features Suppress Emergent Misalignment](https://alignment.openai.com/helpful-assistant-features/).
- Google DeepMind (2025). [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/).
- Xu, X. et al. (2026). [Studying Metagaming Latents in Language Models](https://alignment.openai.com/metagaming-latents/).
- Makelov, A. et al. (2026). [Training Activation Classifiers for Safety and Alignment](https://alignment.openai.com/training-activation-classifiers/) (supervised probes; related work only).
- Cohen, R. et al. (2024). [RippleEdits](https://aclanthology.org/2024.tacl-1.16/).
- Gupta, A. et al. (2024). [Model Editing at Scale Leads to Gradual and Catastrophic Forgetting](https://aclanthology.org/2024.findings-acl.902/).
- Fang, J. et al. (2024). [AlphaEdit](https://arxiv.org/abs/2410.02355).
- Wang, P. et al. (2024). [WISE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html).
- Yang, W. et al. (2025). [The Mirage of Model Editing](https://aclanthology.org/2025.acl-long.745/).
- Zhang, J. et al. (2026). [RULE-EDIT](https://aclanthology.org/2026.findings-emnlp.1924/).
- Takehi, R., Teoh, J., Zhang, A., and Khattab, O. (2026). [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
- Hu, J. et al. (2026). [Exploring Weight-Space Programming](https://arxiv.org/abs/2611.09382).
- Li, M. et al. (2026). [WeightLoop](https://arxiv.org/abs/2611.07319).
- Zhang, J., Wang, K., and Lin, L. (2026). [Can LLMs Debug Robots by Editing Their Weights?](https://arxiv.org/abs/2611.09376).
- Ahmed, N. S. (2026). [Can an LLM Actually Reason and Program about Neural Network's Weights?](https://arxiv.org/abs/2611.08070).
- Lindholz, M. et al. (2026). [Weight-Space Programming for Medical Imaging](https://arxiv.org/abs/2611.12066).

### Training-free policy and prompt optimization

- Pryzant, R. et al. (2023). [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://aclanthology.org/2023.emnlp-main.494/).
- Zheng, C. et al. (2023). [Can We Edit Factual Knowledge by In-Context Learning?](https://arxiv.org/abs/2305.12740).
- Cai, Y. et al. (2025). [Training-Free Group Relative Policy Optimization](https://arxiv.org/abs/2510.08191) and [implementation](https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO).

### Benchmarks and corpora

- Piryani, B., Mozafari, J., and Jatowt, A. (2024). [ChroniclingAmericaQA](https://github.com/datascienceUIBK/ChroniclingAmericaQA) ([paper](https://arxiv.org/abs/2403.17859)).
- HIPE team. [HIPE-2022 data](https://github.com/hipe-eval/HIPE-2022-data).
- Schlechtweg, D. et al. (2020). [SemEval-2020 Task 1](https://aclanthology.org/2020.semeval-1.1/).
- Dell, M. et al. (2023). [American Stories](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ffeb860479ccae44d84c0de32acd693d-Abstract-Datasets_and_Benchmarks.html).
- Text Creation Partnership. [ECCO-TCP FAQ](https://www.textpartnership.net/pages/faq.html).
- KB National Library of the Netherlands. [Historical newspaper OCR ground truth](https://lab.kb.nl/dataset/historical-newspapers-ocr-ground-truth).
- Yu, H. et al. (2026). [AncientDoc](https://aclanthology.org/2026.findings-acl.1438/).
- Gao, L. et al. (2026). [ProHist-Bench](https://aclanthology.org/2026.acl-long.1378/).

## 16. Immediate decisions

1. Which two 4B–9B open model families are the main targets?
2. Is the B1 pilot built from ECCO-TCP aligned with ECCO OCR, from BLN600, or from both? Which is legally usable for release?
3. Which ARR cycle is the target, and does the 14-week plan fit it?
4. Is R3 (agentic WSP) worth the editor cost, or should it be limited to a small number of trajectories on B1 only?
5. After Phase 0, revise the *pilot* thresholds in [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) once and freeze them.
