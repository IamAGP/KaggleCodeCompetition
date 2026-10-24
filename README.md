# Name That LLM

Reproducible experiments and submissions for [Kaggle's Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm).
The classifier predicts one of 18 generating models from a passage.

Git tags preserve the source, notebooks, and tests for each recorded checkpoint.
`name-that-llm-v2` retains the 34.4% submission; `name-that-llm-v3` records 35.1%,
`name-that-llm-v4` records 36.6%, `name-that-llm-v7` records 38.2%, and
`name-that-llm-v10` records **42.7%**.
Full experiment evidence and data are archived privately in S3; the repository contains no
competition data or predictions.

## Frozen-feature follow-ups v11 and v12 (October 24)

Neither experiment qualified for submission. Both reused v10's frozen Qwen features,
fit preprocessing only on training folds, and compared against the same reconstructed
v10 classifier on the existing prompt-held-out development folds.

| Experiment | Best development balanced accuracy | Matched v10 | Difference |
| --- | ---: | ---: | ---: |
| v11: RBF SVM probe blend | 43.01% | 45.34% | −2.34 points |
| v12: shrinkage LDA probe blend | 45.71% | 45.34% | +0.37 points |

Each plan fixed two candidates before fitting and required at least +0.75 development
points before robustness evaluation. Neither passed; no robustness runs or Kaggle
submissions were made. These reused development labels are not an untouched test set.
The fixed historical Qwen members also share prior selection optimism with the comparator.
V12's reported four fold wins include a floating-point tie: materially it won three,
tied one, and lost one. The recorded raw results are preserved unchanged.

Plans are `configs/name-that-llm/v11.json` and `v12.json`; private artifacts retain
frozen plans, source/input hashes, fold audits, scores, and debug logs. This new
45.34% comparison does not replace the historical v10 CV figures below.

## Qwen3-4B probe blend v10

**V10 scored 42.7% public, up from v7's 38.2% (+4.5 percentage points); it is the new
scored best.** Submitted October 22 (kernel `adithyagiri/name-that-llm-v7-qwen-blend-v10`,
version 1). It reruns the unchanged v7 pipeline, adds the v9 frozen Qwen3-4B-Base
hidden-state probe (mean-pooled layers, standardized logistic regression, prompt-grouped
CV 39.64%), and averages row-standardized evidence 50/50. Out-of-fold it moved development
40.45% → 45.23% and related-prompt 40.86% → 44.89%, with 5/5 fold wins on each.

## Prompt assignment experiment v8 (failed)

**V8 scored 36.0% public, below v7's 38.2%; v7 was the scored best at the time.** Submission
on October 22 (kernel `adithyagiri/name-that-llm-prompt-assignment-v8`, version 1) kept
v7 evidence unchanged, recovered prompt groups from test text (TF-IDF cosine
clustering, `round(rows/18)` groups capped at 18) and assigned models one-to-one per
group. Offline it reached 52.49% development and 52.38% related-prompt versus 40.45% and
40.86%, but both partitions share train's complete 18-models-per-prompt structure, so
neither could falsify the assumption that the hidden test has it. The public notebook that
motivated it only matches clusters of 12–18 passages; v8 omitted that guard. The likely
failure is a wrong group count on the hidden test, merging unrelated prompts; this is an
inference, not verified. `configs/name-that-llm/v8.json` records the plan.

## StyleDistance experiment v7

**V7 scored 38.2% public balanced accuracy, up from v4's 36.6%
(+1.6 percentage points).** Submission `57491848` used notebook version 1 and
was submitted once after both frozen validation gates and Kaggle runtime checks
passed. The post-score snapshot on October 22 was **rank 1,682 of 2,477 teams**;
the field grew by 330 teams since the v4 snapshot, so the ranks are not directly
comparable.

Scored notebook: https://www.kaggle.com/code/adithyagiri/name-that-llm-styledistance-v7
It uses [StyleDistance](https://huggingface.co/StyleDistance/styledistance) at
revision `b7df5f0b0480773c097ba3121d83ca32b71015ca` with a fixed balanced
logistic classifier (`C=64`). The selected recipe averages its normalized log
probabilities with separately normalized v4 evidence. No LUAR predictions enter v7.

| Evaluation | Frozen v4 | V7 fixed half blend | Fold wins |
| --- | ---: | ---: | ---: |
| Original prompt groups | 37.8213% | **40.4509%** | 5 / 5 |
| Related-prompt groups | 39.5247% | **40.8588%** | 4 / 5 |
| Kaggle public leaderboard | 36.6% | **38.2%** | — |

The standalone StyleDistance classifier scored 29.5600% on the original folds.
The blend improves development by 2.6296 percentage points and the related-prompt
check by 1.3341 points. Descriptive paired-group bootstrap intervals are
1.2215–4.0031 points and -0.0738–2.8330 points respectively; the second includes
zero. These intervals do not account for the full adaptive search across earlier
experiments. Both partitions reuse inspected labels and are not untouched tests.

The [paper](https://arxiv.org/abs/2410.12757v2) trains style representations using
synthetic and natural examples; it does not demonstrate this competition's
18-class performance. The pinned release is standard RoBERTa with masked token
mean pooling, a 512-token input limit, and MIT license metadata. We preserve raw
text and use all consecutive 510-content-token windows plus two special tokens,
combine window means by nonpadding token count, then normalize the resulting
768-dimensional passage vector. Raw whitespace preservation differs from
SentenceTransformers' default stripping; long-text aggregation and final
normalization are explicit project choices. All training passages fit one window.

All vectors were extracted offline in 192 seconds on the local CPU; peak memory
was 1.48 GB. Source, model, input-row and cache checksums are bound to the frozen
plan in `configs/name-that-llm/v7.json`. The notebook verifies model files and
synthetic cross-runtime vectors before training, keeps each test row independent,
and validates runtime IDs, labels and the serialized Parquet output.

```sh
python scripts/extract_embeddings_v7.py smoke --plan configs/name-that-llm/v7.json
python scripts/extract_embeddings_v7.py extract --plan configs/name-that-llm/v7.json
python scripts/run_embeddings_v7.py development --output-dir artifacts/improvement-v7/development
python scripts/run_embeddings_v7.py robustness --output-dir artifacts/improvement-v7/robustness
python scripts/build_notebook_v7.py
```

Restore the pinned model, immutable baseline artifacts and synthetic runtime
fixture from private S3 before reproduction; use fresh output directories.
The notebook requires the private model dataset and competition data attached,
with internet disabled. It fits the fixed linear probe and v4 stacker on all
training rows at runtime.

The offline Kaggle CPU run completed in 2,088 seconds (34.8 minutes), using
Python 3.13.15, PyTorch 2.11.0+cpu and Transformers 5.16.1. All 14 pinned model
file hashes and 17 runtime source hashes matched; the synthetic encoder check
differed from the local reference by at most 2.39e-7, below its 1e-4 tolerance.
The three-row example Parquet passed ID, label and schema checks, with no
convergence warnings. This runtime check does not establish hidden-test accuracy.

## Frozen representation experiments v5 and v6

V6 followed the failed v5 probe with two separately preregistered classifiers on
the same immutable vectors: logistic regression with `C=64`, and RBF SVC with
`C=10, gamma="scale"`. Each was evaluated alone and in a fixed half blend with v4.
The logistic blend won development at **39.3381% versus 37.8213%**, improving all
five folds. On the related-prompt split it reached **39.8556% versus 39.5247%**,
a gain of only **0.3308 percentage points**, with three of five folds improved.
This failed the frozen 0.5-point/four-fold robustness gate. **V6 was not submitted**;
v4 remains the scored incumbent. `configs/name-that-llm/v6.json` records the
follow-up plan and its dependence on the inspected v5 result.

On October 21, v5 tested frozen [LUAR-CRUD](https://huggingface.co/rrivera1849/LUAR-CRUD)
authorship embeddings, revision `8b36ce7f289ee91900a7dff0eaccf5cdca745ab1`, with
`LogisticRegression(C=1, class_weight="balanced")`. The standalone probe scored
**26.8183%** and its fixed half blend with v4 scored **37.3751%**, below v4's
**37.8213%** on the same prompt folds. It failed the predeclared improvement
gate; no robustness fit or Kaggle submission was made for v5.

Each unchanged passage becomes one episode containing every consecutive
126-token window plus RoBERTa special tokens. Only equal-length episodes share
an inference batch. The 512-dimensional output is unit normalized. The encoder
uses no competition labels and never aggregates passages across test rows.
All 2,699 training vectors and three visible test vectors were extracted offline
in 72.4 seconds on the local CPU, with peak memory about 990 MiB.

The [author paper](https://arxiv.org/html/2401.06712v2) and its
[LLNL implementation](https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024)
motivate the representation experiment; their detection results do not establish
performance on this 18-model classification task. Two library-compatibility
repairs, empty-cache offline checks, exact repeated outputs, batch-equivalence
checks and all data/model/source checksums are retained in the private evidence.

`configs/name-that-llm/v5.json` preserves the frozen plan. Install the optional
`representations` dependencies into an isolated environment. With the audited
model package restored to `artifacts/improvement-v5/model/`:

```sh
python scripts/extract_embeddings_v5.py smoke --plan configs/name-that-llm/v5.json
python scripts/extract_embeddings_v5.py extract --plan configs/name-that-llm/v5.json
python scripts/run_embeddings_v5.py development --output-dir artifacts/improvement-v5/development
```

These commands require fresh output paths; completed caches and evaluation
records are deliberately not overwritten. Both validation partitions reuse
previously inspected development data, and no local result is a hidden-test score.

## Improvement v4

**V4 scored 36.6% public balanced accuracy, up from v3's 35.1%
(+1.5 percentage points).** Submission `57444632` used notebook version 1 and
was submitted exactly once after both predeclared validation gates passed.
The post-score snapshot was **rank 1,503 of 2,147 teams**, compared with
1,615 of 2,142 at the start of the experiment.

Scored notebook: https://www.kaggle.com/code/adithyagiri/name-that-llm-nested-stack-v4?scriptVersionId=360469832

V4 evaluates a learned combination of four frozen base classifiers. Its
development winner averages normalized stacked evidence and v3 evidence with
fixed 50/50 weights. The pure stack scored 37.1907%; the blend scored **37.8213%**
versus v3's **37.0067%**, passing the predeclared 0.75 percentage-point gate and
winning four of five folds. The frozen related-prompt comparison also passed:
**39.5247% versus 38.1186%**, a 1.4062-point gain with four of five folds improved.

| Evaluation | V3 | V4 fixed half blend |
| --- | ---: | ---: |
| Original prompt-grouped nested development | 37.0067% | 37.8213% |
| Fresh related-prompt split, seed 31415 | 38.1186% | 39.5247% |
| Kaggle public leaderboard | 35.1% | 36.6% |

The private offline CPU notebook completed on Kaggle in approximately 816 seconds,
with scikit-learn 1.6.1 and Python 3.13.15. Its four cells matched the local
notebook exactly, all seven runtime source hashes matched, and the generated
Parquet file passed the ID, label, row-count and schema checks before submission.
The hidden-data rerun subsequently completed and returned the public score above.

The descriptive paired-group bootstrap interval on the new split is
0.4144–2.3806 percentage points. This does not account for all development
selection or dependencies between overlapping training folds.

The four bases are the v3 character SVM, style histogram boosting and joint
multinomial text/style classifier, plus v2's character/style SVM. Each contributes
18 normalized scores. A training-only `StandardScaler` and
`LogisticRegression(C=0.05, solver="lbfgs")` combine the resulting 72 features.

Evaluation is genuinely nested by prompt group. For each outer fold, each
meta-training prediction comes from a base model trained without either the
outer fold or that row's own fold. Ten unordered excluded-fold pairs supply these
predictions without repeating equivalent fits. Historical base predictions are
used only for outer validation after checking their source, configuration,
training data, row order, class order and fold provenance. The new split refits
all inner and outer models and keeps seven related prompt pairs together.

The protocol is motivated by [Ting and Witten's stacking study](https://arxiv.org/abs/1105.5466)
and the [official scikit-learn stacking implementation](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/ensemble/_stacking.py).
The historical paper names no GitHub repository. This is a project-specific
logistic combiner, not a reproduction of the paper's model or a claim of
state-of-the-art attribution.

The available labels have already been used in earlier experiments. Nesting
prevents direct validation leakage but does not eliminate earlier model-selection
optimism or make the second split an untouched test set.

Reproduce in a fresh output directory with the v2/v3 artifacts restored:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/run_nested_v4.py --cached-outer-oof
python scripts/select_v4.py select --output-dir artifacts/improvement-v4/selection
python scripts/select_v4.py prepare-robustness --output-dir artifacts/improvement-v4/robustness
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/run_nested_v4.py --folds artifacts/improvement-v4/robustness/folds.csv --output-dir artifacts/improvement-v4/robustness/nested --refit-outer
python scripts/select_v4.py check --run artifacts/improvement-v4/robustness/nested --folds artifacts/improvement-v4/robustness/folds.csv --output-dir artifacts/improvement-v4/robustness/check
python scripts/build_notebook_v4.py
```

The notebook builder rejects a candidate that fails either gate or whose
validation inputs, source, evaluator, selector or frozen plan have changed.
The standalone notebook generates five-fold training predictions, fits the
combiner, and refits the four base models on all training data; no private cached
scores are needed on Kaggle.

All 114 software tests passed in a source-only temporary checkout containing no
private data or saved artifacts. They include actual fit spies for prompt
exclusion, routing, cache corruption, scaler boundaries, prediction batch
independence, fixed candidate selection and notebook provenance.

V4 logs, pair caches, rejected candidates, source snapshots and validation
evidence are retained under `artifacts/improvement-v4/` and archived privately in
`s3://kaggle-featured-code-comps/name-that-llm/experiments/improvement-v4/`.

## Improvement v3

**V3 scored 35.1% public balanced accuracy, up from v2's 34.4% (+0.7 percentage
points).** Submission `57440737` used notebook version 1, submitted once after
both predeclared validation gates passed. The final rank snapshot was
**1,591 of 2,121 teams**, versus 1,656 of 2,112 at the start of this experiment.
It adds a joint multinomial text/style classifier to the
two v2 components, with equal weights after passage-local score normalization:

- Character 1–5 gram TF-IDF with `LinearSVC(C=4)`.
- Histogram gradient boosting over the 465 fixed style measurements.
- Character 1–5 gram TF-IDF plus normalized style measurements, classified by
  `LogisticRegression(C=16, solver="lbfgs")`.

Two frozen experiment rounds evaluated ten new individual models and ten fixed
blends. The first round's best gain, 0.7400 percentage points, missed its
0.75-point gate and was not submitted. The second round passed without changing
that threshold. No validation-fold data enters vocabulary fitting, scaling,
kernel bandwidth selection, or classifier training.

| Evaluation | V2 | V3 candidate |
| --- | ---: | ---: |
| Original prompt-grouped development folds | 35.6366% | 37.0067% |
| Frozen related-prompt split, seed 2027 | 35.7472% | 36.7097% |
| Kaggle public leaderboard | 34.4% | 35.1% |

Scored notebook: https://www.kaggle.com/code/adithyagiri/name-that-llm-validated-ensemble-v3?scriptVersionId=360441703
The exact four-cell notebook was verified against the local source and completed
on Kaggle with scikit-learn 1.6.1, private visibility, CPU, and internet disabled.

On the second split, v3 won four folds and tied one. The pooled gain was
0.9625 percentage points; the descriptive paired-group bootstrap interval was
approximately 0.00–1.90 points. These results reuse development data, training
folds overlap, and selection across candidates introduces optimism. They are not
an untouched test estimate.

Sixty-five software tests passed in a clean temporary checkout containing no
private data or saved artifacts. They cover score and row alignment, training-only
transforms, prompt-group routing, inference batch invariance, kernel arithmetic,
and notebook source integrity.

For a fresh reproduction, first run or restore the v2 experiment and original
baseline folds as described below. Install experimental dependencies and copy the
frozen public configuration files into a new experiment directory:

```sh
python -m pip install -e '.[dev,experiments]'
mkdir -p artifacts/improvement-v3
cp configs/name-that-llm/v3-first-round.json artifacts/improvement-v3/plan.json
cp configs/name-that-llm/v3-second-round.json artifacts/improvement-v3/second-round-plan.json
python scripts/run_boosting_v3.py
python scripts/run_topic_v3.py
python scripts/select_v3.py
python scripts/run_kernel_v3.py
python scripts/run_linear_v3.py
python scripts/select_v3.py --plan artifacts/improvement-v3/second-round-plan.json --output-dir artifacts/improvement-v3/selection-second
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/check_v3.py --plan artifacts/improvement-v3/second-round-plan.json --selected artifacts/improvement-v3/selection-second/selected.json
python scripts/build_notebook_v3.py
python -m pytest -q
```

Use a fresh checkout/output directory for reproduction; completed experiment
directories refuse to be silently overwritten. The runnable submission notebook
embeds its source and fits on all available training rows. It needs no restored
experiment artifacts when run on Kaggle.

Rejected candidates, diagnostics, source snapshots, logs, and scores are retained
under `artifacts/improvement-v3/` and archived in
`s3://kaggle-featured-code-comps/name-that-llm/experiments/improvement-v3/`.

## Improvement v2

**Second Kaggle submission completed: public balanced accuracy 34.4%, up from
29.2% (+5.2 percentage points).** Submission `57439219` used notebook version 1.
The post-score leaderboard snapshot was rank 1,654 of 2,111 teams; the starting
snapshot was rank 2,030 of 2,110.

The second experiment compared 10 individual candidates and 9 fixed blends.
The selected model averages per-passage standardized evidence from:

- Case-sensitive character 1–5 gram TF-IDF with `LinearSVC(C=4)`.
- Histogram gradient boosting over 465 fixed writing-style measurements.

Style measurements cover punctuation, letter/function-word frequencies, word and
sentence length distributions, connectives, and paragraph openings/endings.
Learned preprocessing is fitted on training folds only. Neither classifier uses
IDs, prompt IDs, external data, or pretrained weights.

| Evaluation | Original baseline | Improvement v2 |
| --- | ---: | ---: |
| Original prompt-grouped development folds | 31.0410% | 35.6366% |
| Second split, seven related-prompt pairs grouped | 30.6714% | 36.1916% |
| Kaggle public leaderboard | 29.2% | 34.4% |

The frozen v2 recipe beat baseline in all five folds of the second split.
The paired related-prompt-group bootstrap interval for the gain is
4.04–7.03 percentage points. This is descriptive uncertainty: both split sets
reuse the available development data, and training folds overlap.

Reproduce the recorded experiment:

These commands require the competition files in `data/name-that-llm/` and the
original fold assignments in `artifacts/baseline-v1/folds.csv`. Restore that file
from `s3://kaggle-featured-code-comps/name-that-llm/baselines/v1/folds.csv`, or
generate the baseline once in a fresh checkout using the instructions below.
Run reproduction in a separate checkout so completed experiments stay intact.

```sh
python scripts/run_sparse_experiment.py
python scripts/run_style_experiment.py
python scripts/run_generative_experiment.py
python scripts/select_improvement.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/check_improvement.py
python scripts/build_improved_notebook.py
python -m pytest -q
```

The experiment and unsuccessful candidates are retained under
`artifacts/improvement-v2/`. Twenty-nine tests cover leakage-related contracts,
batch invariance, score alignment, and submission handling.

Notebook: https://www.kaggle.com/code/adithyagiri/name-that-llm-style-ensemble-v2

The scored notebook, all candidate results, validation reports and code are
archived under `s3://kaggle-featured-code-comps/name-that-llm/experiments/improvement-v2/`.

## First experiment

The two configurations were fixed before their scores were inspected. Both use
TF-IDF, L2 normalization and `LinearSVC(C=1, random_state=42)`.

| Candidate | Features | Prompt-grouped balanced accuracy |
| --- | --- | ---: |
| Selected: `char` | Case-sensitive character 3–5 grams | 31.0410% |
| `char_word` | Same character features plus lowercase word 1–2 grams | 30.6334% |

**First Kaggle submission completed: public balanced accuracy 29.2%.**
Submission `57438413` used notebook version 2, scored on October 20, 2026.
The public score is 1.841 percentage points below the selected model's local CV
score; the public and validation datasets differ.

Five folds share the same seed and prompt assignment. Every passage is predicted
once by a model that has not trained on its prompt. Each fold learns its own
vocabulary and IDF values. Only `text` enters the classifier; IDs and prompt IDs
are never predictive features. Scores are pooled out-of-fold balanced accuracy.

These are development scores. Related topics can occur under different prompt
IDs, the earlier qualitative review saw labels across the full dataset, and
selecting a candidate on CV introduces selection optimism. A Kaggle score is
recorded separately in `artifacts/baseline-v1/kaggle_submission.json`.

## Reproduce

Python 3.10+ is required. The local run used Python 3.12.8, NumPy 2.3.2,
pandas 2.3.1, SciPy 1.16.1, scikit-learn 1.7.1, and PyArrow 21.0.0.
The complete run configuration and versions are in `artifacts/baseline-v1/metrics.json`.

```sh
python -m pip install -e '.[dev]'
python scripts/run_baseline.py --data-dir data/name-that-llm
python scripts/build_notebook.py
python -m pytest -q
```

Put the competition's `train.csv`, `test.csv`, and `sample_submission.parquet`
in `data/name-that-llm/`. Raw data and generated run artifacts are ignored by Git.

The experiment saves fixed folds, out-of-fold predictions, per-model recall,
confusion matrices, metrics, and logs under `artifacts/baseline-v1/`.
The local `submission.parquet` covers the three visible test rows and is only a
format check.

Persistent artifacts are archived privately under
`s3://kaggle-featured-code-comps/name-that-llm/` in `ap-south-1`.
The bucket has public access blocked, SSE-S3 encryption, and versioning enabled.
`artifacts/storage/` contains the verified storage location and upload manifests.

## Kaggle notebook

`notebooks/name-that-llm-baseline.ipynb` embeds the shared implementation and
selected configuration. It trains on all 2,699 labeled rows and generates
predictions for every runtime test ID, retaining their order.

Attach `name-that-llm`, select CPU, disable internet, and save/run the notebook.
Submit `submission.parquet` from that saved version. Kaggle reruns it against the
hidden test set. No packages, external data, or pretrained weights are downloaded.

Notebook: https://www.kaggle.com/code/adithyagiri/name-that-llm-character-svm-baseline

Scored version: https://www.kaggle.com/code/adithyagiri/name-that-llm-character-svm-baseline?scriptVersionId=360422843

For version 1, Kaggle's MCP save operation omitted the requested competition data
attachment and the run failed before training. Pushing the same notebook through
the official CLI attached the source; version 2 completed and was scored.
The error and successful run logs are retained with the experiment artifacts.

The generated notebook should be rebuilt after changes to the shared code.
Its metadata records the embedded source hash and training-file hash.

## References

- [Competition evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
- [scikit-learn text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Topic Confusion paper](https://arxiv.org/abs/2104.08530) and
  [authors' implementation](https://github.com/malikaltakrori/Topic-Confusion-for-authorship-attribution-EMNLP-2021-).
  This human-authorship study motivates checking topic confounding; its scores
  are not predictions of performance on this competition.
- [Official Kaggle notebook submission workflow](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md#kaggle-competitions-submit)
