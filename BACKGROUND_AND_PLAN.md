# HisFree: Training-Free Historicalization of Foundation Models

**Literature review status:** updated 10 November 2026. The assessment below prioritizes peer-reviewed ACL Anthology papers, official datasets/repositories, and the original Weight-Space Programming reports. The WSP literature is unusually new: the initiating report appeared on 2 November 2026 and most independent follow-ups are currently preprints, so their results should be treated as promising but not yet settled.

## 1. Project summary

**HisFree** investigates whether general-purpose language and vision-language models already contain useful historical-language and historical-document capabilities that are not reliably expressed at inference time, and whether those capabilities can be improved through **bounded, gradient-free weight editing** rather than continued pretraining or supervised fine-tuning.

The central question is:

> Can an existing open-weight foundation model be turned into a better historical text/document model by identifying and removing localized representational bottlenecks, without backpropagation or conventional training?

Historical corpora are a particularly demanding test bed because they combine several forms of distribution shift:

- diachronic changes in spelling, syntax, vocabulary, meaning, and genre;
- OCR noise and uncertain diplomatic transcription;
- historical fonts, page degradation, multi-column layouts, marginalia, and illustrations;
- duplicated editions and reprints that complicate contamination control;
- long-tail people, places, events, and publication metadata;
- modernizing biases, where a model silently replaces historical evidence with a plausible modern form.

The intended outcome is not merely an OCR corrector. It is a general historical model that can support retrieval, question answering, entity and event extraction, temporal and genre classification, diachronic semantic analysis, page understanding, transcription, and evidence-grounded generation.

## 2. Terminology and scope

The phrase *training-free* needs a precise operational definition. In this project it means:

- no gradient computation or backpropagation through the target model;
- no optimizer-driven update of the target model;
- no conventional supervised fine-tuning or continued language-model pretraining;
- candidate edits may be proposed by an external reasoning model and evaluated with a bounded number of forward passes;
- the target model's parameters may be changed directly, subject to strict edit-size and regression constraints.

This is therefore more accurately described as **gradient-free, optimizer-free, forward-feedback weight editing**. It is not data-free or compute-free.

The initial scope is English historical material, with text-only experiments first and multimodal page-image experiments second. Multilingual evaluation is desirable because it tests whether the method repairs general historical representations rather than overfitting English orthography.

## 3. Background

### 3.1 Historical language models

Existing work largely follows one of three strategies.

1. **Pretraining from scratch on historical text.** MacBERTh was pretrained on English from 1450--1950 and showed that historical pretraining can outperform adapting a contemporary model on several downstream tasks. More recently, TypewriterLM was pretrained as a 7.24B-parameter model on a 54B-token, pre-1913 corpus and introduced temporally aligned instruction tuning and evaluation. These results demonstrate the value of temporal grounding, but require substantial data curation and training compute.

2. **Continued pretraining/domain adaptation.** Historical variants of BERT and related models improve after additional masked-language-model training on period-specific corpora. This is effective but still relies on gradient training and can entangle historical adaptation with catastrophic forgetting or temporal leakage.

3. **Pixel- or image-based historical models.** PHD adapts PIXEL into a general-purpose, OCR-free encoder of historical documents. It pretrains on synthetic historical scans and real newspapers from 1700--1900, reconstructs masked image patches, and evaluates language understanding, semantic search, and historical question answering. PHD establishes that page images can carry both linguistic and document-structural information, but it is an encoder trained with continued pretraining rather than a training-free historicalization method.

These approaches provide strong baselines. They do not answer whether a modern, broadly pretrained language or multimodal model already contains comparable historical capability that can be elicited through localized parameter intervention.

### 3.2 Model editing and weight-space interventions

Model editing predates weight-space programming and contains several distinct paradigms. MEND trains a hypernetwork to transform task gradients into fast edits; SERAC stores corrections in an external memory and routes relevant inputs through a counterfactual model. ROME identifies and updates localized factual associations through rank-one changes to transformer MLP weights, while MEMIT extends direct editing to many factual associations. WISE separates frozen main memory from an editable side memory and highlights a practical tension among reliability, generalization, and locality. Task arithmetic represents behavior changes as directions between base and fine-tuned checkpoints, while activation engineering steers behavior at inference time without changing persistent weights. Unlike these approaches, WSP asks a general reasoning model to design the parameter intervention without a learned editor or target-model gradients.

Weight-Space Programming (WSP) proposes a more agentic setting: a strong reasoning model inspects another model's architecture, weights, activations, and evaluation feedback, then writes targeted parameter edits without gradient descent. The initial WSP experiments reported that editing 0.016% of a small Qwen model's weights reduced reasoning loops and sharply improved several benchmarks. Crucially, the same study could not add mathematical ability to a model whose errors lacked a single localized obstruction. Recent follow-up work similarly finds substantial but inconsistent gains and often retains the original checkpoint when an edit cannot be validated.

The most defensible interpretation is therefore:

> Weight editing may remove a localized obstacle that hides an existing ability, but it should not be assumed to create missing domain knowledge or a missing modality from scratch.

This distinction motivates HisFree. Historical competence may already be partly present because modern foundation models train on large web, book, PDF, and synthetic mixtures, and smaller open models may inherit capabilities through distillation. The project tests this hypothesis rather than assuming it.

### 3.3 Research gap

No established line of work currently combines all of the following:

- systematic auditing of latent historical capability in modern open foundation models;
- contamination-aware separation of memorization from historical-language generalization;
- direct, gradient-free editing for broad domain adaptation rather than isolated factual replacement;
- joint treatment of clean text, noisy OCR, page images, and metadata as views of the same historical object;
- explicit preservation gates for modern/general capabilities;
- evaluation across historical QA, information extraction, diachronic semantics, retrieval, and document understanding.

HisFree should be framed around this gap, not around the weaker claim that applying an existing WSP recipe to ECCO is itself sufficient novelty.

### 3.4 Progress of the field: 2021--2026

The relevant literature has moved through five partially overlapping stages.

| Period | Representative progress | What is now established | Remaining limitation for HisFree |
|---|---|---|---|
| 2021--2023 | MacBERTh, GHisBERT, continued historical pretraining | Historical-domain pretraining helps lexical and downstream NLP tasks | Mostly encoder models, text-only, and trained with gradients |
| 2023--2024 | PHD, American Stories, large structured newspaper corpora | Page pixels, layout, and OCR noise can be modeled jointly; historical document modeling need not be OCR-only | PHD is an encoder and requires synthetic/real-image pretraining; broad generative historical reasoning remains untested |
| 2024--2025 | CHURRO, LLM post-OCR studies, stronger editing audits | Specialized VLM training can outperform frontier systems on historical transcription; naive LLM correction can also modernize or corrupt valid historical forms | Recognition gains do not automatically transfer to QA, retrieval, semantics, or historical reasoning |
| 2026 | TypewriterLM, AncientDoc, ProHist-Bench, HisDoc-OCR, image-informed correction | Billion-scale History LMs and broad OCR-to-reasoning VLM benchmarks now exist; visual grounding and temporal leakage are first-class problems | These systems are trained or retrieval-based; none establishes gradient-free broad historicalization of a modern open model |
| Nov. 2026 | WSP, WSP-Scope, WeightLoop, robot-policy editing | Extremely sparse edits can sometimes repair a localized failure and transfer beyond the exact edit task | Reliability is inconsistent, evidence is mostly preprint-level, and no study shows that WSP can create a broad missing domain capability |

This timeline makes two claims unsafe. First, it is no longer accurate to present a general-purpose historical model as novel by itself: TypewriterLM is explicitly a History LM, and PHD/CHURRO are strong historical document models. Second, it is not yet justified to claim that WSP is a replacement for domain training. The novelty has to come from the **causal test of latent capability, bounded gradient-free intervention, and broad transfer evaluation**.

### 3.5 What Weight-Space Programming has and has not demonstrated

The original *Sparks of Weight-Space Programming* report edited 0.016% of Qwen3.5-0.8B to address a reasoning-loop pathology. Under the reported sampling setup, MATH-500 rose from 7.9% to 48.1%; the edit also improved instruction following. This is a striking result, but it is one target model, one diagnosed pathology, and an early technical report rather than a peer-reviewed study.

The first follow-up evidence is informative:

- **WSP-Scope** ran 16 editing trajectories over seven benchmarks and two Qwen scales. It found gains as large as 29.45 percentage points from only 1,023 modified parameters, but transfer was uneven and 6 of 16 trajectories correctly retained the original weights because an edit could not be validated.
- **Robot policy repair** provides the cleanest analogy for historical orthography. Editing 65 token-embedding rows raised held-out instruction success from 86.3% to 92.0%, with 48 repairs and four regressions. Yet 12 of 13 further sessions declined an edit, and the authors explicitly conclude that the evidence supports lexical repair, not acquisition of missing robot skills.
- **WeightLoop** shows that an LLM can write a reusable bounded editing program driven by subsequent experimental feedback. This expands WSP from a one-off edit into an update loop, but the application is a small prediction/discovery setting rather than a foundation-model domain conversion.
- **Medical imaging WSP** found that LLM-written CNNs performed above chance, but Adam training on only 256 labeled images beat them on every MedMNIST task; one Astra result was systematically below chance. This is direct evidence against the strongest "no training data needed" interpretation.
- A weight-reading audit found that apparent understanding can often be explained by local arithmetic plus priors about familiar datasets; label permutation broke semantic interpretations. Raw-weight access must therefore be separated experimentally from benchmark or architecture memorization.

The broader model-editing literature is also cautionary. RippleEdits found that prominent editors fail to propagate edits consistently and that an in-context baseline performed best. Large-scale sequential editing causes forgetting and downstream degradation. *The Mirage of Model Editing* showed that a realistic QA evaluation reduced apparent single-edit performance from 96.8% to 38.5%, partly because common teacher-forced evaluations leak target content and length. RULE-EDIT further showed that near-perfect direct edit reliability can coexist with less than 52% instance portability and less than 26% rule understanding.

**Implication for this project:** WSP is a credible mechanism for discovering and removing a shared bottleneck. It is not evidence that a model can absorb ECCO, COHA, or Gale as new parametric memory without training. HisFree must use archive data as diagnostic and validation evidence, while retrieval remains responsible for rare facts and document-specific memory.

### 3.6 Current state of historical language and document models

The strongest adjacent systems now cover several different meanings of "historical model":

| Work | Model/data | Capability demonstrated | Why it does not subsume HisFree |
|---|---|---|---|
| MacBERTh (2021) | BERT-style encoder, English 1450--1950 | Historical lexical/linguistic downstream tasks | Trained from scratch; not generative or multimodal |
| PHD (EMNLP 2023) | PIXEL-based encoder, synthetic scans plus 1700--1900 newspapers | OCR-free NLU, retrieval, and historical QA from rendered documents | Continued pixel pretraining; encoder-only; limited corpus/period coverage |
| CHURRO (EMNLP 2025) | 3B VLM, 99,491 pages from 155 corpora, 46 language clusters | Strong open-weight multilingual historical text recognition | Specialized training; primarily transcription rather than general historical reasoning |
| TypewriterLM (EMNLP 2026) | 7.24B decoder, 54B-token 1700--1913 corpus, trained for about 140B tokens | Temporally bounded generation, instruction following, cutoff evaluation | Expensive from-scratch pretraining; text-only; optimizes historical cutoff rather than adapting an existing modern model |
| AncientDoc (ACL Findings 2026) | ~3,000 pages, >100 books, 14 Chinese document types | Benchmark from page OCR through translation and reasoning QA | Benchmark, not an adaptation method; different language/script but valuable external transfer test |
| HisDoc-OCR (ACL Findings 2026) | Historical Chinese MLLM adaptation | Diagnoses language-prior domination and repairs visual grounding | Uses training-based layout injection, reweighting, and attention distillation |
| ProHist-Bench (ACL 2026) | 400 expert questions, 10,891 rubrics, 18 evaluated LLMs | Professional historical/evidentiary reasoning remains difficult even for strong LLMs | Knowledge- and reasoning-heavy Chinese benchmark, not a page-reading model |

TypewriterLM is particularly important because it weakens a simple "modern models have seen the archive, therefore no historical pretraining is necessary" claim. Its authors show that a strict temporal model requires aggressive filtering of modern metadata, a historical tokenizer, leakage-aware instruction tuning, and temporally aligned evaluation. Its general benchmark scores remain far below a modern Qwen3-4B baseline (for example, 43.3 versus 77.4 on ARC-Easy), illustrating a real trade-off between temporal grounding and modern general competence. HisFree should test whether a modern model can retain its general competence while acquiring **interpretive robustness**, not attempt to reproduce a strict historical knowledge cutoff through sparse editing.

CHURRO and recent image-informed post-OCR work also make an OCR-only paper too narrow. CHURRO already reports a 3B specialized VLM across 22 centuries and 46 language clusters; image-plus-noisy-OCR prompting can sharply reduce transcription error. A competitive HisFree paper therefore needs at least one capability above transcription: evidence-grounded QA, entity/event extraction, diachronic meaning, retrieval, or historical reasoning.

### 3.7 Evidence-based novelty verdict

As of 10 November 2026, the project is **potentially novel, but only under a stricter formulation**.

| Proposed claim | Verdict | Reason |
|---|---|---|
| "We build a historical language model" | Not novel enough | MacBERTh, PHD, CHURRO, TypewriterLM, and other History LMs already occupy this space |
| "We use ECCO/COHA/Gale for historical adaptation" | Not novel enough | Domain pretraining, fine-tuning, OCR correction, and retrieval over historical archives are established |
| "We apply WSP to historical text" | Weak novelty alone | This is a new application, but risks reading as technique transfer without a scientific insight |
| "We identify which historical failures are latent bottlenecks and repair them with sparse, gradient-free edits" | Stronger and currently open | No located work combines causal localization, WSP, historical data, and broad cross-corpus evaluation |
| "We show when sparse edits fail and training/retrieval is necessary" | Strong scientific contribution | It establishes the boundary between capability elicitation and domain learning |
| "One edit transfers across clean text, OCR, image, corpus, period, and task without general regression" | Potentially CCF B/A-level result | This would demonstrate a general principle rather than an archive-specific trick |

The most defensible paper thesis is:

> Modern foundation models possess uneven latent historical competence. Small, forward-validated weight interventions can repair some shared access and grounding failures, but cannot substitute for retrieval or pretraining when the underlying historical knowledge is absent.

This is falsifiable and remains valuable if the final result is a carefully delimited negative result.

### 3.8 Review-derived changes to the experimental design

The literature implies five non-negotiable design changes:

1. **Use free generation, never teacher forcing, for edit evaluation.** Otherwise edit success may be dramatically overstated.
2. **Validate base knowledge before editing.** A model-specific a priori probe, similar in spirit to TRACE, should distinguish "knows but cannot express" from "does not know." Only the first class is a plausible WSP target.
3. **Measure portability and ripple effects.** Every edit needs direct, paraphrase, derived-instance, neighboring-concept, and unrelated-task tests.
4. **Include a no-edit decision and multiple independent trajectories.** The WSP literature shows that refusing an unvalidated edit is often the correct result.
5. **Use matched non-parametric baselines.** Prompting, in-context correction, activation steering, and RAG sometimes beat parameter editing; the paper must show why a persistent weight change is warranted.

### 3.9 Review scope and uncertainty

This is a targeted state-of-the-field review, not a PRISMA-style systematic review. It covers the primary technical lines needed for the project: historical pretraining and continued adaptation, pixel/VLM historical document modeling, historical evaluation datasets, factual and conceptual model editing, activation/weight interventions, and the November 2026 WSP wave. Peer-reviewed results are treated as stronger evidence than arXiv preprints; the original WSP blog is treated as a motivating report whose headline claims require independent replication. Because this literature is changing within days, the WSP subsection should be refreshed before paper submission and all claimed reproductions should be tied to immutable code/checkpoint hashes.

## 4. Research questions and hypotheses

### RQ1: How much historical capability is already present?

Measure current open language and vision-language models on clean historical text, raw OCR, page images, and combined image-plus-text input.

**H1.** Modern models will perform well on clean historical text and common historical knowledge, but will show larger degradation on rare genres, raw pages, OCR noise, temporally shifted word meanings, and exact evidence localization.

### RQ2: Are failures localized bottlenecks or missing knowledge?

Use controlled input contrasts, activation probes, causal interventions, and edit attempts to distinguish representation failures from genuinely absent knowledge.

**H2.** A meaningful subset of historical failures will be attributable to localized bottlenecks in token embeddings, early/middle MLP directions, visual projection, or cross-modal alignment. Long-tail factual knowledge will be less editable and will remain better handled by retrieval.

### RQ3: Can bounded weight editing improve broad historical capability?

Ask an editor model to propose small, reversible changes to selected parameter regions and accept an edit only after held-out validation.

**H3.** Constrained edits can improve multiple historical tasks when they target shared failures such as orthographic fragmentation, OCR sensitivity, modernization bias, or page-to-language misalignment.

### RQ4: Do edits transfer?

Evaluate across corpora, time periods, genres, model families, and modalities.

**H4.** A genuine historicalization edit should transfer beyond the corpus and task used to propose it. Corpus-specific gains without cross-corpus transfer should be classified as overfitting, not historicalization.

### RQ5: What is the cost of historicalization?

Measure edit size, forward-pass budget, editor cost, variance across runs, and regression on modern tasks.

**H5.** The useful operating point is a Pareto frontier between historical gain, general-capability retention, edit sparsity, and inference cost rather than a single aggregate score.

## 5. Data strategy

### 5.1 Project corpora

The project expects access to combinations of:

- ECCO page images, OCR, metadata, and ECCO-TCP transcriptions where alignable;
- historical newspapers with page images, OCR, article boundaries, and dates;
- COHA or comparable clean, time-indexed text;
- Gale or other licensed collections where permitted.

Each item should be represented, when possible, as a multi-view document bundle:

```json
{
  "document_id": "...",
  "work_id": "...",
  "edition_family": "...",
  "image": "...",
  "raw_ocr": "...",
  "diplomatic_text": "...",
  "normalized_text": "...",
  "year": 1764,
  "genre": "sermon",
  "printer": "...",
  "source_collection": "..."
}
```

Licenses must be recorded per field and per collection. Restricted source pages should not be redistributed unless explicitly allowed; derived annotations, identifiers, hashes, evaluation code, or a controlled evaluation service may still be releasable.

### 5.2 Public benchmark suite

Public benchmarks are necessary for comparability. The following compact suite covers distinct abilities rather than treating OCR as the whole problem.

| Capability | Candidate benchmark | Views | Primary metrics |
|---|---|---|---|
| Historical QA | ChroniclingAmericaQA | corrected text, raw OCR, page image | EM, token F1, evidence localization |
| Entity recognition/linking | HIPE-2022 or HIPE-2026 | multilingual historical newspaper OCR | span F1, linking accuracy |
| Diachronic semantics | SemEval-2020 Task 1 | time-sliced English/German/Latin/Swedish | accuracy, Spearman correlation |
| Layout and reading order | PRImA ENP historical newspapers | page image and layout ground truth | mAP/IoU, reading-order score |
| Multilingual historical recognition | CHURRO-DS | page/region images across centuries and scripts | normalized edit similarity, CER/WER, historical preservation |
| OCR robustness | ECCO-TCP pairs, BLN600, or KB historical ground truth | OCR and human text | CER/WER, HCPR/AIR, downstream retention |
| Retrieval/layout scale | American Stories / Chronicling America | image, article regions, OCR, metadata | Recall@k, MRR, nDCG |
| Controlled visual robustness | PHD-style rendered GLUE/SQuAD/NER | clean and degraded renderings | task metric under controlled shifts |
| OCR-to-reasoning transfer | AncientDoc | historical page images, translation, QA | CER plus task-specific exact/semantic scores |
| Professional historical reasoning | ProHist-Bench | expert questions and rubric evidence | rubric-level accuracy and evidence quality |
| Temporal cutoff/knowledge | History-Event or TRACE-style probes | event chains and dates | recall, surprisingness, chronology consistency |
| General regression | selected modern NLU and DocVQA tasks | text and documents | original task metrics |

ChroniclingAmericaQA is especially useful because its records include questions, answers, cleaned context, raw OCR, dates, and source URLs. The same semantic task can therefore be evaluated under clean-text, noisy-text, image, and combined-input conditions.

HIPE provides multilingual historical entity recognition and linking with explicit dates and document types. SemEval-2020 Task 1 tests semantic change across time and helps prevent the benchmark from collapsing into document reading alone.

### 5.3 Contamination-aware challenge set

Public benchmarks may have appeared in pretraining or synthetic post-training data. Public results must therefore be supplemented by a blind challenge set drawn from project corpora.

Requirements:

- split by complete work and **edition family**, not random page;
- keep authors, printers, titles, and reprints from leaking across splits where feasible;
- include rare genres, low-quality scans, and long-tail entities;
- prefer recently digitized, unpublished, or access-controlled pages for the final blind test;
- construct synthetic but linguistically valid counterfactual sentences that cannot have been memorized;
- freeze test data before editing begins;
- permit the editor to see edit/dev data only, use validation to accept edits, and run test once at the end.

The benchmark should explicitly distinguish:

1. exact memorization or passage continuation;
2. historical linguistic generalization;
3. document visual understanding;
4. external retrieval ability;
5. diachronic semantic competence.

## 6. Proposed method

### 6.1 Stage A: capability audit before editing

Evaluate unmodified models under matched views:

```text
clean transcription
historical/diplomatic transcription
raw OCR
synthetically corrupted OCR
page image
page image + OCR
page image + clean text
```

Use paired examples so that the content is constant while spelling, noise, image quality, layout, or metadata changes. This identifies whether a failure belongs to the language path, visual path, or cross-modal interface.

Candidate model set:

- at least three open-weight families when compute permits;
- at least one small model suitable for rapid WSP replication;
- at least one stronger text model;
- at least one open vision-language model with accessible vision projector and language backbone.

Closed frontier models can be used as diagnostic baselines or editors, but the primary target models must be open-weight for reproducibility.

### 6.2 Stage B: constrained edit spaces

Do not initially expose every parameter. Define reversible edit spaces:

**Text models**

- selected token-embedding rows or low-dimensional embedding transforms;
- a small set of middle-layer MLP directions;
- attention output projections implicated by causal tests;
- an identity-initialized low-rank historical adapter that the editor writes directly.

**Vision-language models**

- visual patch embedding or normalization parameters;
- the vision-to-language projector;
- selected early language layers processing image tokens;
- low-rank adapters attached to the visual projector or cross-modal layers.

Editing an isolated adapter is safer and easier to distribute; editing original weights is a stronger scientific test. Both should be evaluated separately.

### 6.3 Stage C: editor harness

The editor receives:

- the target model architecture and tokenizer/processor;
- summaries or slices of permitted parameters;
- paired failure examples from the edit split;
- activation, attention, similarity, and error-cluster reports;
- a CPU/Python workspace with bounded read/write access;
- a limited forward-pass budget;
- an API that can propose, apply, revert, and hash candidate edits.

The editor must not receive test examples or unrestricted internet access during an edit trajectory.

A candidate edit \(\theta'\) is accepted only if it improves a predeclared historical validation objective while satisfying regression and sparsity constraints. A conceptual objective is:

```text
historical_gain(theta')
- lambda * general_capability_regression(theta')
- mu * edit_size(theta', theta)
- nu * newly_introduced_failures(theta')
```

The editor is allowed to return **no edit**. Refusal is a correct outcome when the evidence does not support a localized intervention.

### 6.4 Stage D: retrieval boundary

Weight editing should target reusable capabilities, not memorize the archive. Long-tail facts, page-specific claims, and bibliographic evidence should remain in an external retrieval layer. The intended system boundary is:

```text
edited historical model = better reader/reasoner
retrieval system         = historical memory and evidence
```

This separation makes it possible to test whether editing improves document interpretation without conflating the result with knowledge injection.

## 7. Evaluation protocol

### 7.1 Baselines

At minimum, compare against:

- original frozen model;
- prompting/in-context learning;
- RAG where the task permits retrieval;
- activation steering;
- in-context editing or an external correction memory;
- random edits matched by layer, rank, parameter count, and norm;
- a conventional model editor such as ROME/MEMIT/AlphaEdit on compatible targets;
- LoRA or another parameter-efficient trained adapter;
- continued pretraining/domain-adaptive pretraining where compute permits;
- historical pretrained models such as MacBERTh, PHD, CHURRO, or TypewriterLM on compatible tasks.

The comparison with trained baselines is necessary even though HisFree is training-free: the paper must quantify what quality is traded for the absence of backpropagation.

### 7.2 Metrics

Report task metrics separately and summarize results along four axes:

1. **Historical gain:** normalized improvement across historical tasks.
2. **General retention:** change on modern/general benchmarks and unrelated historical tasks.
3. **Edit efficiency:** parameters changed, rank, norm, forward passes, editor calls, wall time, and energy/compute where available.
4. **Transfer:** gain on unseen corpora, periods, genres, languages, and model families.

For transcription-like tasks, include historical-fidelity measures in addition to CER/WER, such as preservation of valid archaic forms and rates of anachronistic or invented characters. A lower CER is not sufficient if the model modernizes the evidence.

### 7.3 Statistical requirements

- run multiple independent editor trajectories;
- freeze decoding and evaluation settings before comparison;
- evaluate by unconstrained/free generation rather than teacher-forced target likelihood alone;
- use paired tests or paired bootstrap confidence intervals;
- report repairs and regressions, not only aggregate accuracy;
- report direct reliability, paraphrase generalization, derived-instance portability, semantic ripple effects, and locality separately;
- hash all base and edited parameter tensors;
- retain complete edit logs;
- include unsuccessful edits and no-edit outcomes.

### 7.4 Success criteria

An edit should count as successful historicalization only if it:

- improves at least two distinct historical capabilities;
- transfers to a corpus not used during editing;
- survives paraphrase or view changes;
- causes no material general-capability regression beyond a predefined tolerance;
- beats matched random edits;
- is stable across more than one run or has a documented variance profile.

## 8. Initial experiment plan

### Phase 0: repository and data contracts (week 1)

- create dataset manifests with provenance, license, modality, date range, and release constraints;
- define the multi-view document schema;
- select a small public benchmark subset for rapid iteration;
- implement immutable edit/dev/validation/test splits by work and edition family;
- establish experiment configuration, checkpoint hashing, and result logging.

**Deliverable:** reproducible data manifest and evaluation skeleton.

### Phase 1: baseline audit (weeks 2--3)

- evaluate 2--3 text models on clean versus historical versus OCR views;
- evaluate at least one VLM on image, OCR, and image-plus-OCR inputs;
- run memorization probes and synthetic counterfactual tests;
- cluster failures by orthography, temporal semantics, entity rarity, OCR pattern, layout, and evidence localization;
- identify a small number of repeated, plausibly localized failures.

**Deliverable:** capability audit report and a go/no-go decision for weight editing.

### Phase 2: reproduce a known WSP edit (week 3)

- reproduce or closely replicate a released small-model WSP experiment;
- verify the harness can read, patch, revert, hash, and evaluate weights;
- reproduce matched random-edit controls;
- document all deviations from the released setup.

**Deliverable:** validated editor harness independent of historical data.

### Phase 3: text-only historical editing pilot (weeks 4--6)

- choose one failure class with strong evidence of shared structure;
- compare embedding-row, low-rank adapter, MLP-direction, and activation-steering interventions;
- allow a strong editor model to propose bounded changes;
- validate on held-out corpora and modern regression tasks;
- test whether improvements persist under clean, OCR, and paraphrased views.

Promising first targets:

- orthographic fragmentation and archaic spelling invariance;
- modernization bias in diplomatic transcription;
- temporal word-sense calibration;
- named-entity robustness under OCR corruption.

**Deliverable:** first causal claim about when historical weight editing works or fails.

### Phase 4: multimodal pilot (weeks 7--9)

- align page-image, OCR, and clean-text representations for the same content;
- diagnose projector versus language-backbone errors;
- edit a projector or isolated cross-modal adapter first;
- evaluate on ChroniclingAmericaQA and a blind ECCO/newspaper subset;
- include layout and evidence-localization tasks, not only transcription.

**Deliverable:** multimodal historicalization result with image/text ablations.

### Phase 5: scale and paper decision (weeks 10--12)

- repeat the best method across model families;
- add multilingual or cross-domain transfer if targeting a general ML venue;
- complete baselines and statistical analysis;
- decide between two paper framings:
  - **NLP framing:** latent historical capability and contamination-aware evaluation;
  - **ML framing:** general gradient-free domain specialization through bounded weight editing.

**Deliverable:** paper-ready result table, limitations, and release plan.

## 9. Go/no-go gates

### Gate 1: latent capability exists

Proceed to editing only if the unmodified model is substantially better on clean historical text than on one or more corrupted/modal variants, or if a repeated historical failure can be localized. If performance is uniformly poor, the model likely lacks the capability and conventional training or retrieval is the more honest solution.

### Gate 2: edits transfer

Proceed beyond a pilot only if an edit improves a corpus or task not used to propose it. Improvements limited to the edit set are not evidence of historicalization.

### Gate 3: editing is competitive

Continue the training-free claim only if editing offers a meaningful advantage in compute, data, interpretability, modularity, or preservation relative to LoRA/continued pretraining. If it is simply worse and not cheaper or more interpretable, report the negative result rather than forcing the framing.

### Gate 4: multimodality adds evidence

Include the VLM contribution only if image inputs reveal or solve a failure not captured by text/OCR alone. Otherwise, keep the first paper text-focused and treat multimodality as follow-up work.

## 10. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Public benchmark contamination | Apparent capability may be memorization | blind/private test, post-cutoff scans, counterfactual examples, edition-family splits |
| Editor overfits validation feedback | "Training-free" becomes test optimization | strict split isolation, bounded calls, one-shot final test, complete logs |
| No localized historical failure | Weight editing cannot add missing ability | allow no-edit outcome; use retrieval or trained baseline |
| Catastrophic or silent regressions | Historical gain damages general model | regression gate, matched controls, reversible adapter, per-edit hashes |
| Proprietary editor dependence | Poor reproducibility | test multiple editors, include open editor if possible, release prompts/logs |
| Restricted ECCO/Gale licenses | Cannot release data or checkpoint | provenance manifest, derived annotations, IDs/hashes, public benchmarks, evaluation service |
| Historical modernization | Better surface metrics but damaged evidence | diplomatic/normalized outputs, historical preservation and anachronism metrics |
| Benchmark too OCR-centric | Weak foundation-model claim | require QA, IE, semantic change, retrieval, and document tasks |
| Compute cost of repeated VLM evaluation | Project stalls before method test | begin with small text model and fixed benchmark subsets |

## 11. Expected contributions

A strong paper should aim to contribute all of the following:

1. A contamination-aware audit of historical capability in modern open language and vision-language models.
2. A reproducible, bounded framework for gradient-free historicalization through weight editing.
3. Evidence identifying which historical failures behave like localized bottlenecks and which require training or retrieval.
4. A multi-view evaluation protocol spanning clean text, OCR, page images, and metadata.
5. A public benchmark configuration plus a blind challenge set protocol.
6. Full edit trajectories, negative results, matched random controls, and regression analysis.

The strongest scientific result may be positive or negative. A rigorous demonstration that broad historical ability cannot be reliably created through WSP, despite successful narrow repairs, would still clarify the boundary between capability elicitation and domain learning.

## 12. Venue strategy

The method should determine the venue, not the other way around.

- **ACL / EMNLP:** best fit for the historical capability audit, benchmark, and NLP-focused weight-editing study.
- **ICML / ICLR / NeurIPS:** plausible only if the method generalizes beyond historical text and establishes a general principle of gradient-free domain specialization.
- **CVPR:** plausible if the principal contribution is multimodal page representation or visual-projector editing.
- **ICDAR:** excellent topical fit for historical document understanding, though it is not CCF B or above in the current CCF list.

For CCF classification, verify the current official list and submission category. The current CCF policy counts main-conference full/regular papers, not Findings, short papers, demos, or workshops.

## 13. Suggested repository structure

```text
emnlp-hisfree/
├── README.md
├── BACKGROUND_AND_PLAN.md
├── configs/
│   ├── data/
│   ├── models/
│   ├── edits/
│   └── evaluations/
├── data/
│   ├── manifests/
│   └── README.md
├── src/
│   ├── datasets/
│   ├── evaluation/
│   ├── probes/
│   ├── editing/
│   └── logging/
├── tests/
├── scripts/
├── results/
│   └── README.md
└── docs/
    ├── licenses.md
    ├── benchmark_protocol.md
    └── experiment_log.md
```

## 14. Selected references and resources

### Historical language and document models

- Manjavacas Arevalo, E. and Fonteyn, L. (2021). [MacBERTh: Development and Evaluation of a Historically Pre-trained Language Model for English (1450--1950)](https://aclanthology.org/2021.nlp4dh-1.4/).
- Borenstein, N., Rust, P., Elliott, D., and Augenstein, I. (2023). [PHD: Pixel-Based Language Modeling of Historical Documents](https://aclanthology.org/2023.emnlp-main.7/).
- Semnani, S. et al. (2025). [CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition](https://aclanthology.org/2025.emnlp-main.1763/).
- Harju, A. and van der Goot, R. (2025). [How to age BERT Well: Continuous Training for Historical Language Adaptation](https://aclanthology.org/2025.loreslm-1.21/).
- Fittschen, E. et al. (2026). [Pretraining Language Models for Diachronic Linguistic Change Discovery](https://aclanthology.org/2026.findings-eacl.241/).
- Luo, X. et al. (2026). [A Language Model from 1913: Pretraining on Historical Text](https://aclanthology.org/2026.emnlp-main.2474/).
- Levchenko, M. A. (2025). [Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities](https://aclanthology.org/2025.lm4dh-1.7/).
- Kanerva, J. et al. (2025). [OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches](https://aclanthology.org/2025.resourceful-1.8/).
- Thomas, M. et al. (2026). [Image-Informed Post-OCR Correction with Vision-Language Models](https://aclanthology.org/2026.findings-emnlp.60/).
- Cao, J. et al. (2026). [HisDoc-OCR: Restoring Visual Grounding in MLLMs for Chinese Historical Document OCR](https://aclanthology.org/2026.findings-acl.301/).

### Weight and activation editing

- Meng, K. et al. (2022). [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262).
- Mitchell, E. et al. (2022). [Fast Model Editing at Scale (MEND)](https://openreview.net/forum?id=0DcZxeWfOPt).
- Mitchell, E. et al. (2022). [Memory-Based Model Editing at Scale (SERAC)](https://proceedings.mlr.press/v162/mitchell22a.html).
- Meng, K. et al. (2023). [Mass-Editing Memory in a Transformer (MEMIT)](https://arxiv.org/abs/2210.07229).
- Ilharco, G. et al. (2023). [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089).
- Turner, A. M. et al. (2023). [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248).
- Cohen, R. et al. (2024). [Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://aclanthology.org/2024.tacl-1.16/).
- Gupta, A. et al. (2024). [Model Editing at Scale Leads to Gradual and Catastrophic Forgetting](https://aclanthology.org/2024.findings-acl.902/).
- Fang, J. et al. (2024). [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355).
- Wang, P. et al. (2024). [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html).
- Yang, W. et al. (2025). [The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://aclanthology.org/2025.acl-long.745/).
- Takehi, R., Teoh, J., Zhang, A., and Khattab, O. (2026). [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
- Hu, J. et al. (2026). [Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations](https://arxiv.org/abs/2611.09382).
- Li, M. et al. (2026). [WeightLoop: Weight-Space Programming with Experimental Feedback](https://arxiv.org/abs/2611.07319).
- Zhang, J., Wang, K., and Lin, L. (2026). [Can LLMs Debug Robots by Editing Their Weights?](https://arxiv.org/abs/2611.09376).
- Ahmed, N. S. (2026). [Can an LLM Actually Reason and Program about Neural Network's Weights?](https://arxiv.org/abs/2611.08070).
- Lindholz, M. et al. (2026). [Weight-Space Programming for Medical Imaging: Do We Still Need Training Data?](https://arxiv.org/abs/2611.12066).
- Zhang, J. et al. (2026). [RULE-EDIT: Benchmarking Rule-Level Knowledge Editing in Large Language Models](https://aclanthology.org/2026.findings-emnlp.1924/).

### Public benchmarks and corpora

- Piryani, B., Mozafari, J., and Jatowt, A. (2024). [ChroniclingAmericaQA repository](https://github.com/datascienceUIBK/ChroniclingAmericaQA) and [paper](https://arxiv.org/abs/2403.17859).
- HIPE team. [HIPE-2022 multilingual historical NER and entity-linking data](https://github.com/hipe-eval/HIPE-2022-data).
- Schlechtweg, D. et al. (2020). [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://aclanthology.org/2020.semeval-1.1/).
- Dell, M. et al. (2023). [American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ffeb860479ccae44d84c0de32acd693d-Abstract-Datasets_and_Benchmarks.html).
- PRImA Research Lab. [ENP Image and Ground Truth Dataset of Historical Newspapers](https://www.primaresearch.org/datasets/ENP).
- KB National Library of the Netherlands. [Delpher Newspapers](https://www.kb.nl/en/research-find/datasets/delpher-newspapers) and [historical newspaper OCR ground truth](https://lab.kb.nl/dataset/historical-newspapers-ocr-ground-truth).
- Text Creation Partnership. [ECCO-TCP access and reuse FAQ](https://www.textpartnership.net/pages/faq.html).
- Yu, H. et al. (2026). [AncientDoc: Benchmarking Vision-Language Models on Chinese Ancient Documents](https://aclanthology.org/2026.findings-acl.1438/).
- Gao, L. et al. (2026). [ProHist-Bench: Can LLMs Act as Historians?](https://aclanthology.org/2026.acl-long.1378/).
- Piper, A. and Jiang, J. (2026). [Evaluating Multi-modal LLMs on Historical Book Illustrations](https://aclanthology.org/2026.findings-emnlp.1304/).

## 15. Immediate next decisions

Before implementation, decide:

1. Which open text model is small enough for the first reproducible WSP pilot?
2. Which open VLM exposes a sufficiently simple vision projector for controlled editing?
3. Which project corpora can be legally used for editing, blind evaluation, and public release?
4. What compute budget is available for repeated forward-only evaluation?
5. Is the first paper primarily an audit paper or a method paper?

The recommended default is to begin with the audit. It is useful regardless of whether weight editing succeeds, and it prevents the project from mistaking missing knowledge for an editable representational bottleneck.
