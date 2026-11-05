# Credited public comparison — local preparation only

Upstream: Omer Faruk Merey, `0-668-verifier-matched-rows-pool-prompts`, version 2. Full upstream attribution remains in both notebooks. The public control makes engineering changes and claims no score or original research gain. The separately named CE8B variant adds our existing component, as described below.

Run `python scripts/build_nov5_public_control.py` to regenerate the draft. This only writes local artifacts; it does not start Kaggle or cloud jobs.

## Prepared

- Fixed global query indexing in the pairwise verifier after the first 128 rows.
- Removed placeholder commits; small test inputs exercise actual model inference.
- Added required-member checks and process exit checks, finite-score/ID audits, and intermediate/final score exports.
- Pinned dataset references and enforced Open Pool Kaggle version13's byte size, row count and published hash.
- Corrected upstream's undefined `pathlib` reference in final pairwise fusion; enforced extra-passage batch pins and required family gates instead of silently skipping them.
- Retained upstream training/inference recipe and contributor credit. Unknown annotation-license status is explicit.

## Checks completed

- Every generated Python cell and extracted script parses.
- Notebook format validates, with no outputs retained.
- A 257-row synthetic regression checks the actual verifier scoring loop at batch sizes 128 and 257. The repaired loop produces identical scores; the original differs (max absolute difference 0.08990449). See `index-regression-check.json`. This is a code-correctness check, not a model-quality result.

## Resolve before any launch

1. The live first attempt reportedly resolved the latest Open Pool despite requesting `/13`; the unchanged byte guard correctly stopped it. Both metadata files now attach the dedicated `melvindenishl/ntl-open-pool-v13-snapshot/1` dataset. Its current version1 and 96,388,050-byte main file were reconfirmed through Kaggle on November6. Existing row, byte and published SHA-256 guards remain unchanged. SDK acceptance of a historical suffix is not proof of backend pinning. Other data versions are requested at their observed current versions; read back resolved attachments.
2. The complementary `lora24_public` checkpoint (`**/lora24/final/COMPLETE.json`) is a declared optional legacy member because upstream states its training kernel became unavailable. Its absence is explicitly logged; other expected LoRA members fail if missing. The completion audit records whether it ran.
3. Qwen3.5 token-attention is explicitly optional because its backbone is not attached. DeBERTa token-attention is required and fails the completion check if absent. Availability and actual members are recorded in the complementary audit.
4. Mercury `ntllm-chatgpt-a-b` has an Unknown license in Kaggle metadata. This is not a clearance to redistribute it. Preserve original terms and resolve intended use before publication.
5. No GPU execution has validated this draft. Confirm environment compatibility and genuine small-test inference before submission.

## Later original contribution

`public-control-core.npz` exports aligned weighted core-member arrays. `public-control-final.npz` exports IDs, label order, pre-postprocessing logits, final logits and probabilities.

`inference-ce8b.ipynb` is a separately named executable candidate with `kernel-metadata-ce8b.json`. It preserves the full corrected public baseline outputs, then invokes the unchanged hash-verified V39 helper, adapter and readout, and adds `0.15 * log(P_CE8B)` to the final public logits. No further Sinkhorn is applied. This fusion position differs from V39's V16 experiment and is explicitly exploratory. The inherited public baseline's gain must not be described as our own. The helper/source hashes and GPU environment digest are in `build-manifest-ce8b.json`.

Both variants now establish the same audited V39 package environment before costly inference, using the existing private offline wheel mirror version1. Model/data methodology remains the credited public recipe; the private wheel mirror is only dependency packaging. The combined public recipe has not yet been demonstrated compatible with that environment on a GPU. Metadata specifies the successful V34 CUDA image; the launcher must preserve that explicit digest rather than override it with `latest`.

Notebook-output metadata only documents unversioned `owner/slug` refs; all seven Lucas outputs currently resolve to version1 and the vnton37nguyn output resolves to version2. The API validator permissively accepts more segments but does not document a kernel-output version syntax. Verify resolved attachments or use an explicitly authorized immutable snapshot; do not assume a guessed suffix pins outputs.
