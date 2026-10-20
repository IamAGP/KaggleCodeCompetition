# Name That LLM

Reproducible experiments and submissions for [Kaggle's Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm).
The classifier predicts one of 18 generating models from a passage.

The Git tag `name-that-llm-v2` preserves the source, notebooks, and tests for the
34.4% public-score checkpoint. Full experiment evidence and data are archived
privately in S3; the repository contains no competition data or predictions.

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
