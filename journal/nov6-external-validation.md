# Frozen external validation — November 6 IST

The V39 competition submission is 58032484; its score is pending. A separate validation notebook now measures the fixed CE8B addition against its complete V16 baseline on newer external responses, with no model fitting or tuning.

## Cohort and limits

Open Pool versions 13 and 15 differ by 22,035 rows. Normalized-text exclusions against the older pool, host data and our pool_v4, old-prompt exclusions using the same pool15 canonical mapping, strong source prompt evidence and runtime-training-source exclusions leave 17,507 rows across 3,577 prompts. These rows match Kilupy's source data. Generation on November 2 is publisher-attested; per-row provider timestamps are unavailable. V39 fitted checkpoints predate that date (public models October19/23, our CE8B October27).

Gemini3.7's new responses reuse older prompts, so no clean 18-label cohort was established. Select 32 rows per remaining label by a frozen text-hash ordering: 544 rows, 381 prompts. No predictions were used. A prompt-disjoint reserve of 14,138 rows remains unmeasured. The source labels and distribution differ from the competition; this is not host OOF or an estimate of the leaderboard score.

## Frozen comparison

Preserve V39's original models, fixed weights, tokenizers, numerical settings, label-free clustering and centered twins. Compare baseline versus fixed 0.15 CE8B addition using raw 18-column scores as the primary measure. Report the original equal-mass 18-class Sinkhorn result only as a secondary stress check because the cohort lacks one class. A separate evaluation protocol explicitly amends the cohort manifest's earlier recommendation against centering/Sinkhorn; this decision preceded model results. No priors, weights or thresholds are fitted.

Labels are read only in the final evaluation cell, after predictions exist. The notebook cannot emit a competition `submission.parquet`; its outputs are external validation artifacts. It computes 17-observed-class balanced accuracy, corrections/regressions, and 5,000 prompt-cluster bootstrap draws.

## Exact artifacts

- Private dataset: `adithyagiri/name-that-llm-nov6-external-validation`, ID13117785, version1. All four uploaded files were downloaded and matched byte-for-byte; private visibility verified.
- Cohort manifest SHA256: `6d7f2cee9c6ec5a027b17f87ddd3f75a4d93f6b46d1d17d9a94e09f380df1443`.
- Evaluation protocol SHA256: `6f831e64e06a2b2d1181f4d6748417d1c7e767d340c6a76cb0aba8e87cf901d7`.
- Notebook: `adithyagiri/name-that-llm-external-validation-nov6`, version1, source SHA256 `1477374aa98fdebd9e10c8106c4ad4e8a314026e9823b1cc1d54d242d0f36e78`.
- Source and attachment identities read back. Session reports RUNNING; no validation result yet.
- Full audit, selection inputs and confirmation reserve remain private under `artifacts/nov6-external-holdout-audit/`.

Source generations belong to Kilupy (README CC BY4.0; Kaggle license field unset), through crtproduction Open Pool (CC BY-SA4.0). Baseline models and method retain Nishant Dahal and upstream credits; our contribution is the trained CE8B adapter/readout and this separate fixed evaluation.

The private dashboard's latest progress page is https://name-that-llm-lab-notebook.helpless-silly98.chatgpt.site/#progress . The deployment succeeded without changing its owner-only audience.

## Completed result and decision

Version 1 completed on November 5 UTC / November 6 IST. Its runtime audit records 913 seconds for model execution; all five mandatory members ran on all 544 rows, with the two original centered twins applied to 515 rows. The pinned CUDA runtime matched V39. Downloaded scores, predictions, report and runtime audit passed the independent verifier `scripts/verify_external_validation_nov6.py`, including a separate recalculation of the prompt-cluster bootstrap.

| Frozen comparison | Baseline | Fixed CE8B addition |
|---|---:|---:|
| Primary: raw score argmax | 259/544 (47.61%) | 253/544 (46.51%) |
| Secondary: original 18-class balancing | 271/544 (49.82%) | 271/544 (49.82%) |

The primary addition fixes 5 errors and introduces 11: **−1.10 percentage points**, with the predeclared 5,000-draw prompt-cluster bootstrap 95% interval **[−2.54, +0.32] percentage points**. This does not support promoting the fixed CE8B addition. The missing class and external generation differences still limit extrapolation to the competition; these numbers are not leaderboard scores. No weights were adjusted after seeing the result, and the prompt-disjoint confirmation reserve remains unmeasured.

V39 was already submitted before this evaluation completed; submission 58032484 remains pending at this snapshot. For the separately running stronger public comparison, prioritize its preserved **unaugmented** `submission-public-control.parquet` if execution and output checks pass, rather than automatically submitting the CE8B-augmented output. Credit remains with the original public authors. No original model gain is claimed.

Independent verification: `artifacts/nov6-external-validation/independent-verification.json`. Downloaded outputs: `artifacts/nov6-external-validation-output/`. Score-array SHA-256: `6a6e817091c65c58541c2c2db16d2feb40381174dbda3a66c48b873e1c13be06`.

Official resource-filtered Runpod billing now reports $4.1624992084 posted across the task's four pods and three volumes, leaving $5.8375007916 of the original $10 cap before future billing corrections. A $1 reserve would leave $4.8375007916 available. No paid compute was started in this resumed round. Sanitized evidence is in `artifacts/nov6-budget-reconciliation/`.
