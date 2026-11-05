# November 5 experiment audit

Status snapshot: November 5, 2026. This records completed local measurements separately from work in progress. No new leaderboard gain is established by this report.

## What changed today

| Work | Evidence | Decision |
|---|---|---|
| Existing prediction audit | V34 sparse+8B: 1,442/2,699 correct; sparse+4B: 1,313. The 8B system corrects 264 errors and introduces 135. Gemini 3.7/3.8 have 92 mutual confusions across 300 passages. | Investigate that pair; retain the qualified 8B delivery candidate. These are component/blend results, not validation of the full .622 ensemble. |
| Fixed Gemini pair specialist | Four primary folds: 1,150→1,152 correct /2,159; 24 corrections, 22 regressions. Per-fold net changes −3,+1,0,+4. Pair accuracy 110→112 correct /240. Related split: 294→298 /540. | **Do not promote.** Primary improvement is only 0.093 percentage points and inconsistent. Related data were previously inspected too. |
| Token-rank transitions | Fixed protocol and exact GPT2-medium revision; causal-alignment, padding, ties, bucket/transition, serial-oracle and actual GPT2 prefix-parity tests passed, per research agent. A 100-row CPU throughput probe completed in 39.36 seconds; the projected full extraction is 17.7 minutes. Extraction of all 2,699 rows is running. | **In progress:** no measured accuracy improvement or full-extraction result yet. The below-45-minute runtime gate passed. No GPU job launched for this screen. |
| V39 delivery | Versions 2–3 lost input attachments; version 4 used CPU PyTorch; version 5 failed the transformers version guard. Version 6 preserves all attachments and the exact known-good V34 GPU image; fresh Kaggle metadata confirms its digest and session status is RUNNING. Latest locally inspected receipt is version 6; its verification records `execution_proven=false` and `submission_performed=false`. | Saved-version existence is not proof of inference completion or competition submission. Root must verify execution/output before submitting. |

The Gemini screen used each fold's own 8B encoder features, training-pair rows only, train-only standardization, and fixed binary logistic regression (C=0.02) separately on three layers. It changed a prediction only if the existing winner was Gemini 3.7 or 3.8. No search followed the results. All original training/validation prompt boundaries and saved feature hashes were checked. Pair selection used previously inspected OOF errors; this is exploratory evidence, not untouched confirmation.

Sources: `artifacts/nov5-error-audit/report.json`; `artifacts/nov5-gemini-pair-feasibility/report.json`; `artifacts/nov5-gemini-pair-screen/{plan,report}.json`; `scripts/{audit_errors_nov5,audit_gemini_pair_feasibility_nov5,run_gemini_pair_screen_nov5}.py`; `configs/name-that-llm/nov5-rank-transitions.json`; `scripts/rank_transitions.py`; `tests/test_rank_transitions.py`; `artifacts/improvement-v39/nov5-v34image-{sdk-run-receipt,private-run-verification}.json`. Raw rows and prediction files remain in ignored local artifacts, not this journal.

## What is missing for validation against the actual .622 ensemble?

The exact V21 recipe retains the credited V16 members and adds our 4B component. All weights multiply log probabilities before final softmax and 300-iteration equal-class-mass balancing:

| Member | Weight | Fold-safe cache needed |
|---|---:|---|
| Four-view TF-IDF LinearSVC | 1.0 | Held-out 18-class decision scores; vocabulary and fit must exclude held-out host prompts. Exact V16 external-pool specification required. |
| Qwen3-14B LoRA, plain head | 0.4 | Held-out pooled features plus plain-head scores, from a fold-safe adapter and heads. |
| Qwen3-14B centered twin | 0.1 | Same held-out features, centered-head weights, label-free cluster IDs/mask and scores. |
| Qwen2.5-14B LoRA, plain head | 0.2 | As above, independently for this encoder. |
| Qwen2.5-14B centered twin | 0.1 | As above. |
| DeBERTa-v3-large | 0.25 | Held-out 18-class scores from a fold-safe encoder/head. |
| Our Qwen3-4B SupCon addition | 0.15 | Already available per primary and related holdout through V34 comparison readouts. |
| Replacement Qwen3-8B CE addition | 0.15 | Already available through V34 candidate readouts. Compare replacement with the preceding row, retaining all other members. |

The twin clusters must be inferred from each held-out batch's text using the original distance threshold 0.85 and size range 2–18; do not substitute known prompt IDs. Preserve class order, clipping, weights and balancing exactly.

**Local inventory:** `artifacts/improvement-v21/inference-output/v21_scores.npz` contains only three visible test rows, not host OOF. Existing V34 caches provide both our 4B and 8B alternatives. A recursive filename inventory found no local upstream 14B/DeBERTa fold-safe checkpoints or prediction caches. The external `data/name-that-llm/external/pool_v4.parquet` exists. V32 sparse OOF uses a selected/weighted V17 pool recipe and cannot silently substitute for the exact V16 sparse member.

**Leakage constraint:** `configs/name-that-llm/v16.json` explicitly records that public pretrained heads saw host labels. Applying those full-host models to the 2,699 training rows cannot create honest OOF. Refitting only their output heads would also be insufficient if adapters/encoders already learned those labels. Every missing member needs auditable training boundaries.

**Smallest useful computation:** first obtain matching fold-safe upstream checkpoints and provenance, if available. Otherwise reproduce the three missing neural training recipes on one predeclared related-group holdout (2,159 training /540 validation), including exclusion of held-out prompts from all host adaptation and auditing external overlap. Refit the exact sparse member on that boundary; infer and cache the three neural members once. Then compare the fixed full incumbent against the fixed 8B replacement on CPU using the already-saved 4B/8B readouts. This establishes a real, but previously inspected, one-split marginal comparison; it is not full OOF or independent confirmation. Full OOF needs all five matching folds, and stronger confirmation needs genuinely new prompt groups excluded from all training and prior selection.

No trustworthy GPU time/cost estimate follows from the caches inspected here. Obtain the exact upstream training recipes and benchmark before allocating compute. No jobs or cloud changes were made for this feasibility audit.

Recipe sources: `notebooks/improvement-v16/name-that-llm-two-stage-v16.ipynb`; `scripts/build_notebook_v21.py`; `scripts/build_notebook_v39.py`; `scripts/run_reference_v32.py`; `artifacts/improvement-v34/{primary-oof.npz,related/related-predictions.npz}`. Original credited recipe: https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two and assets: https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend.

## Durable snapshot

Reviewed code, aggregate reports and launch receipts were archived privately at `s3://kaggle-featured-code-comps/name-that-llm/research/2026-11-05/research-snapshot-1.tar.gz`, version `G5t3Xoa7ewMXGTTNUtimbu82uou.M2.z`. A download verified 25,848 bytes and SHA-256 `3e1a664889dc8da3bebfdcd2fb896f58debefe5c28d169eb54baefefb95a5723`. Bucket public-access blocking, encryption and versioning were verified. An initial MCP byte-body upload failed checksum verification; the verified version replaced it and the invalid version was removed. This snapshot predates this journal addendum and later experiment results.

### Official upstream release check (November 6 continuation)

Live Kaggle MCP inspection found **no published OOF, fold IDs, validation-row IDs, fold-specific checkpoints or stage-one-only checkpoints** in the author's two datasets and two linked notebooks. `ntl-handwriting-blend` is dataset **12792418, version 1**, with 15 files: three inference model directories and the external pool. Its description explicitly states that every checkpoint underwent public-data training followed by competition-training adaptation. `ntl-two-stage-lora` is dataset **12693662, version 1**, with 11 files containing two final LoRA sets and the old pool. Both notebooks expose only a 676-byte `submission.parquet` output. No large weights were downloaded.

The current part-two source (notebook **140306245, version 2**) reports five-fold CV but explicitly excludes centered twins from those CV numbers; its reported 0.642 is therefore not validation of the complete submitted inference recipe. The retrieved source SHA-256 is `10ea965c83cf515a8ab98099f87bc6f41160fbf662e9edee6af84118405b6855`. Model weight checksums and fold boundaries are unavailable in the retrieved listings. Without additional author-released artifacts, exact full-baseline validation still requires retraining.

Evidence: `artifacts/nov5-upstream-validation-inventory/{public-api-evidence,part-one-outputs,report}.json`. Official author releases: https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend ; https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora ; https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two ; https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool .

## V39 submission accepted

Version 6 completed real dual-T4 execution in 635 seconds. All five required members and runtime version guards passed. Downloaded outputs passed schema, ID order, finite-score, weighted-vote arithmetic and prediction consistency checks. Kaggle accepted competition submission **58032484** using version 6 on November 5 UTC (November 6 IST). Hidden-test score remains pending. The full public baseline is credited to Nishant Dahal and its contributors; our addition is the fixed trained 8B component.

## Stronger public comparison launched

A separate exploratory candidate, `adithyagiri/ntl-credited-public-control-plus-ce8b-nov5` version 1, is running on Kaggle. It preserves Omer v2 and upstream credits, fixes verified global query indexing and a pathlib NameError, exercises real small-test inference, requires intended members, and preserves baseline outputs before adding our CE8B fixed 0.15 log-probability vote after public postprocessing. This fusion position differs from V39 and is not a validated improvement. Source SHA-256: `59639ba6b606adc88da876facab09ac9f3859fc984bc9eea7fb686e3e0a5955a`. Source, input identities, private access and pinned GPU image were read back. Explicit dataset versions were requested; exact Open Pool content is guarded at runtime. Notebook-output version resolution remains as documented in its manifest. No competition submission from this candidate yet.

## Token-rank result: reject

Frozen GPT2-medium token-rank/entropy/transition extraction completed all 2,699 passages on CPU in 1,034.6 seconds. Its fixed 10% probability blend with the existing raw V34 8B component changed balanced accuracy by -0.001484; prompt-bootstrap 95% interval [-0.004077, +0.000738]. Fold gains: +0.001852, -0.005556, -0.001852, -0.001852, 0. The promotion gate failed; no tuning, GPU training or submission follows this result. This was a model-likelihood feature hypothesis, not an attempt to emulate any target model’s tokenizer. Results and prediction arrays are preserved privately in `artifacts/nov5-rank-evaluation/`; feature SHA-256 is `08922c0d11078583ba9af12ffda08a176c63e0745164112db618b8062ae23a76`.

## Input snapshot correction

The public comparison’s first execution stopped at its Open Pool byte-size guard: requesting historical `/13` did not deliver that content. Version 2 replaces only the attachment with `melvindenishl/ntl-open-pool-v13-snapshot/1`, independently rechecked at 96,388,050 bytes and the same published SHA-256. Notebook source is unchanged. Kaggle save/readback verified version 2, source and attachment identities; GPU execution remains pending.

The completed rank experiment, private feature/prediction arrays, V39 submission/output evidence and comparison source were archived at `s3://kaggle-featured-code-comps/name-that-llm/research/2026-11-05/results-and-submission.tar.gz`, version `qtsMmHNMe2ES8Qelxd_E.20kw17_D2lJ`, 1,463,907 bytes, SHA-256 `cad0b630e23b2ebf51636ff1eb9c2a839f10eb1f372bfb5c7da8894ebfa2954d`. S3 verified the full-object SHA-256 and size. That snapshot predates the attachment correction.
