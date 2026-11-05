# Static execution review — November 6, 2026

Scope: local source inspection and non-GPU code checks. No notebook launch, weight download, training, or submission. This is a second-pass review by the preparing agent, not an independent empirical reproduction. Machine-readable checks and source hashes are in `static-review-checks.json`.

## Findings corrected before handoff

1. Upstream `infer.py` imports `pathlib as _V38Path` but subsequently calls `pathlib.Path` at verifier fusion. That is an unresolved global and would raise NameError. The builder now uses `_V38Path.Path`.
2. Required extra-passage batch filters previously caught missing columns and continued without applying the pin. They now fail instead. Required family gates likewise fail on missing scores or wrong shapes.
3. Generated ID arrays now use a Unicode dtype, so exported NPZs load with `allow_pickle=False`.
4. The README's stale claim that the metadata was unversioned and ambiguous CE8B absence wording were corrected.

## Member and ordering audit

Core required keys exactly match the three sparse assignments (`tfidf`, `svc4`, `distort`) and six active MODELS entries (`q14v5`, `q8v5`, `ll8v5`, `n14b`, `q25_14`, `deb`). Additional `*_twin`/`*_gauss` terms remain conditional on eligible prompt groups. Their absence or all-zero values on three unrelated rows is not treated as a missing backbone.

Complementary required keys match `add_member` paths: `stylegram`, `stylometry_lgb`, `ultrabert`, `debv3l`, `electl`, `desk`, `lora24_seed7`, `lora24_claude`, `lora20_q25_instyle`, `tokattn_deb`. The three encoder heads have positive weights, so their call paths are active. Zero-weight plain decoder heads are deliberately not required; those backbones still supply group/twin/Gaussian features.

The legacy `lora24_public` is explicitly optional because its source checkpoint is not among the supported attachments and upstream documents that it became unavailable. Its presence/absence is logged. `tokattn_q35` is explicitly optional because Qwen3.5 is not attached. DeBERTa token attention is required, even though upstream's generic availability filter could otherwise skip it. Successful family-gate application is enforced for Claude style, Claude LoRA, GPT, Gemini and GLM.

Execution order is: audited package setup; dataset/model discovery and GPU assertion; write Gaussian/sibling/pairwise scripts; run verifier; write main/complementary scripts; run complementary subprocess with checked exit; run main subprocess with checked exit; validate submission. Main checks complementary IDs/labels, all required LPs, ATLAS inference, sibling residual and verifier. The CE8B variant runs afterward in the parent process, verifies the existing private manifest/helper/adapter/readout, checks label/ID alignment, preserves baseline output and produces its new output.

All five extracted scripts and the combined ordinary notebook cells parse. Static global-symbol resolution reports no unresolved globals. Notebook format validates. Static symbol resolution does not prove runtime compatibility, branch reachability, GPU memory sufficiency, or file layout correctness.

## Fixed 0.15 contribution: interpretation and recommendation

The public core first sums weighted log scores. `--sk_div` defaults to 1, so it divides by the count of active LP arrays before softmax. It then mixes complementary probabilities (0.15) and ATLAS probabilities (0.20 on selected completed rows), performs soft marginal balancing, and adds typography/sibling/verifier corrections. Consequently its final logits are on an approximate log-probability scale, not the unnormalized sum with an SVC coefficient of 12.

The candidate adds `0.15 * log(P_CE8B)` **after** these public steps. This is mathematically a normalized product proportional to `P_public * P_CE8B**0.15`. It has a concrete, nonvanishing interpretation: CE8B odds of 10:1 multiply the public odds by about 1.41; 100:1 by about 2.00; 1000:1 by about 2.82. With probability clipping at 1e-9, the largest pairwise multiplier is about 22.39.

Recommendation: retain 0.15 as the already specified exploratory coefficient, with both outputs saved, rather than rescale it based on an imagined gain. Do not describe it as calibrated or transferred validation. V39 originally adds its vote to V16's raw weighted log sum **before** Sinkhorn, whose principal weights total 1.85 plus conditional twins. Therefore identical numeric weight does not imply identical relative influence or previously demonstrated performance. A later approved comparison can assess the saved component margins/calibration; no outcomes were inspected or used to choose this coefficient.

## Remaining execution risks

- Metadata pins dataset versions; notebook-output refs need resolved-version readback against `notebook-output-resolver.json`.
- The V34 CUDA image and V39 dependency guards are explicit, but this larger public recipe has not executed in that combined environment. No guard is removed to force it through.
- Upstream reserves most of the nine-hour runtime for public inference/ATLAS. CE8B adds runtime after that. No measured full-hidden runtime exists for this combination; inspect the real small-run timings before authorizing its hidden run. Baseline outputs remain available if the exploratory addition fails, but an errored notebook must not be represented as a successful combined submission.
- Mercury annotation metadata says Unknown license. Original attribution and that uncertainty remain visible.

No score, original research gain, full-ensemble cross-validation result, or successful GPU execution is claimed.
