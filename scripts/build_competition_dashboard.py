"""Build the private dashboard dataset from local evidence. Never publish raw data to GitHub."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/dashboard-site/dist'

# Editorial summaries are tied to README.md and the named experiment artifacts.
# Local results and Kaggle scores are deliberately separate fields.
EXPERIMENTS = [
 (1,'Character patterns','Submitted','Traditional ML','Count short sequences of characters and train a linear classifier.','The first reproducible baseline.', 'Prompt-group OOF balanced accuracy 0.3104.', 'Public 0.292; established the starting point.'),
 (2,'Writing-style features','Submitted','Traditional ML','Add measurements of punctuation, formatting and writing style to character patterns.','Test whether presentation carries model identity.','Development 0.3564; related-prompt check 0.3619.','Public 0.344.'),
 (3,'Combine text and style','Submitted','Traditional ML','Combine three classifiers that look at character patterns and style.','Different signals may correct different errors.','Development 0.3701; related-prompt check 0.3671.','Public 0.351.'),
 (4,'Learn the combination','Submitted','Ensemble','Train a regularized second classifier to combine evidence from four text/style models.','A learned combination may outperform fixed voting.','Development 0.3782; related-prompt check 0.3952.','Public 0.366.'),
 (5,'LUAR authorship embeddings','Did not qualify','Frozen representations','Use an existing authorship encoder to turn text into numerical vectors. Train a small classifier on those vectors.','Test whether human-authorship representations transfer to LLM attribution.','Standalone 0.2682; blend 0.3738 versus matched 0.3782.','Did not beat the comparison; no submission.'),
 (6,'New classifiers on LUAR','Did not qualify','Frozen representations','Try two fixed classifiers on the already extracted LUAR vectors.','Check whether the readout, rather than the representation, was limiting performance.','Development blend 0.3934; related-prompt 0.3986 versus 0.3952.','Failed the predeclared robustness requirement; no submission.'),
 (7,'StyleDistance representations','Submitted','Frozen representations','Use an encoder designed to represent writing style, then combine it with our text/style system.','Look for style information beyond character counts.','Development 0.4045; related-prompt 0.4086.','Public 0.382.'),
 (8,'Assign one author per prompt group','Submitted','Structured prediction','Infer which passages share a prompt and enforce one model per recovered group.','Exploit the nearly complete 18-model training groups.','Development about 0.5249; related-prompt about 0.5238.','Public fell to 0.360. The local structural assumptions did not transfer.'),
 (9,'Read Qwen3-4B hidden states','Support','Frozen representations','Extract numerical features from a frozen language model; fit a small classifier.','Test whether a language model can recognize source-specific patterns without fine-tuning.','Prompt-group probe balanced accuracy 0.3964.','Feature experiment used in V10; no separate submission.'),
 (10,'Add the Qwen probe','Submitted','Ensemble','Combine V7 and the frozen Qwen probe with equal standardized evidence.','Use a different representation to correct existing errors.','Development 0.4523; related-prompt 0.4489.','Public 0.427, our best before the public baseline reproduction.'),
 (11,'Nonlinear frozen-feature classifier','Did not qualify','Frozen representations','Fit an RBF classifier to existing Qwen features.','Test a curved decision boundary instead of a linear one.','Best development blend 0.4301 versus matched 0.4534.','Failed; no submission.'),
 (12,'Covariance-based classifier','Did not qualify','Frozen representations','Use shrinkage LDA on frozen Qwen features.','Model class distributions with a different classifier.','Best development 0.4571 versus 0.4534.','The small gain missed the fixed promotion threshold.'),
 (13,'Stronger covariance shrinkage','Did not qualify','Frozen representations','Increase regularization in a separately recorded LDA experiment.','Check whether noisy covariance estimates explain weak gains.','Development 0.4482 versus 0.4534.','Failed; no submission.'),
 (14,'Frozen 8B reader','Did not qualify','Frozen representations','Read features from a larger frozen Qwen model using full passages or prefixes.','Test larger representations before paying for fine-tuning.','Full-passage blend 0.4460; prefix blend 0.4275; reference 0.4534.','Neither passed; no submission.'),
 (15,'Exploratory LDA robustness check','Did not qualify','Frozen representations','Recheck the fixed V12 winner under a separately declared exploratory policy.','Test whether its small local gain survives related prompts.','Related-prompt 0.4538 versus 0.4523; only two material fold wins.','Failed even the relaxed policy; V12 was not relabeled as passed.'),
 (16,'Credited public ensemble','Submitted','Public reproduction','Reproduce Nishant Dahal’s public two-stage LoRA ensemble, preserving its models, inference and credit.','Establish a much stronger baseline using publicly shared competition work.','Upstream author reported 0.642 CV; this is not our own validation.','Public 0.620. This major jump is a credited reproduction, not our original improvement.'),
 (17,'External-data quality and weighting','Did not qualify','Data quality','Audit the public pool and compare uniform, source-balanced and source-plus-length weighting.','Better data weighting might transfer more reliably.','540-row pilot: host-only 0.2870; uniform pool 0.3944; weighted variants 0.3815.','Weighting failed. The uniform-pool model became a local comparison reference.'),
 (18,'Train for style across prompts','Support','Fine-tuning','Compare ordinary classification with classification plus supervised contrastive learning on Qwen3-4B.','Encourage texts from one author to have similar representations across topics.','540-row pilot heads: ordinary 0.2907; contrastive 0.3222.','Contrastive objective passed its pilot gate, but both heads were weak.'),
 (19,'Improve the readout','Support','Fine-tuning','Fit the same small classifier on features from three internal layers of the V18 models.','Check whether useful source information exists but the original head fails to extract it.','Pilot: frozen base 0.3722; ordinary-trained 0.3926; contrastive-trained 0.4204.','Contrastive readout qualified for further work; no direct submission.'),
 (20,'Five-fold contrastive confirmation','Planned','Validation','Plan independent host-fold training from external-only checkpoints.','Test the pilot gain across more held-out prompts.','Not run.','V21’s prospective submission took priority. No confirmation pass is claimed.'),
 (21,'Our contrastive addition','Submitted','Ensemble','Train the V19-selected component on all labeled rows and add it to the credited V16 ensemble at fixed weight 0.15.','Test whether our learned representation complements the stronger public ensemble.','The weaker sparse-pilot blend was 0.4741; no full-ensemble OOF claim.','Public 0.622, the current best: +0.002 over V16.'),
 (22,'Four times the external data','Support','Fine-tuning','Increase selected external passages from 4,500 to 18,000, retaining the paired training comparison.','Test whether more examples improve source recognition.','540-row pilot: contrastive readout 0.4426; sparse blend 0.5111.','Passed its local gate. More data also meant more training steps; effects are not isolated.'),
 (23,'Submit the larger-data component','Submitted','Ensemble','Replace V21’s added component with the V22 larger-data component at the same weight.','Test whether the better local model improves the actual competition ensemble.','No honest full-ensemble OOF score was available.','Public 0.620: down 0.002 from V21. The local gain did not transfer.'),
 (24,'8B training feasibility','Support','Runtime probe','Check whether 8B quantized adapter training fits available GPUs and the time budget.','Avoid launching a long run that cannot fit.','Numerical/reload checks passed; projected run 6.69 hours. No accuracy estimate.','Feasibility passed after setup repairs; supported V25.'),
 (25,'Larger fine-tuned reader','Did not qualify','Fine-tuning','Compare ordinary and contrastive training on Qwen3-8B with quantized weights and broader adapters.','Test a stronger trainable representation.','540-row pilot: ordinary readout 0.4833, blend 0.5074; contrastive readout 0.4722, blend 0.5185.','Both blends missed 0.5311. No submission. Size, quantization and adapter coverage changed together.'),
 (26,'Full-host 8B training','Planned','Fine-tuning','Prepare a full-data training route if V25 qualifies.','Turn a validated pilot into a deployable model.','Not run.','V25 did not qualify, so this route was not launched.'),
 (27,'Nonlinear adapted-feature readout','Did not qualify','Fine-tuning','Use one fixed RBF classifier on V22’s adapted features.','Check whether the adapted features need a nonlinear classifier.','Pilot readout 0.3963; blend 0.4611 versus V22 blend 0.5111.','Failed; no submission or retuning.'),
 (28,'Cross-fitted score calibration','Did not qualify','Ensemble','Learn regularized weights and class offsets across six prompt-disjoint calibration folds.','Different score scales may make a fixed combination inefficient.','Same 540-row pilot: 0.5259 versus 0.5111; gain interval −0.0093 to +0.0389.','Failed the fixed accuracy and positive lower-bound requirements.'),
 (29,'Source-likelihood feasibility','Support','Runtime probe','Measure one-GPU training and inference for a model conditioned on the candidate text author.','Explore a different objective from text classification.','Numerical checks passed; pilot runtime projected 31,601 seconds versus 24,300-second gate.','Single-GPU route rejected for runtime; no accuracy result.'),
 (30,'Source-conditioned likelihood','Running','Generative attribution','Train Qwen2.5-1.5B to model text under each of the 18 candidate sources, using source-specific adapters.','Ask which source condition best explains the passage, rather than only classifying an embedding.','Three external epochs, two host epochs; 540 held-out passages. Results pending.','Running at the recorded snapshot. Requires blend ≥0.5311 and positive bootstrap lower gain bound before broader validation.'),
 (31,'Two-GPU likelihood feasibility','Support','Runtime probe','Test distributed training, smaller loss scaling, checkpoint equivalence and inference time.','Make the new likelihood method feasible within the runtime limit.','Second version passed; pilot projection about 4.89 hours; no accuracy estimate.','Enabled V30. First version’s numerical failure remains recorded.'),
 (32,'Expand the sparse comparison','Support','Validation','Complete the fixed sparse comparison on the other four prompt folds; retain fold zero unchanged.','Prepare a consistent reference for broader confirmation.','Pooled OOF accuracy 0.3927. This is not a public score.','Reference computation only, not an improvement candidate.'),
 (33,'Conditional likelihood confirmation','Planned','Validation','Prepare four new primary folds and one related-prompt holdout, starting from V30’s external-only checkpoint.','A successful pilot must survive broader checks before full-data training.','Data preparation and package checks passed. No training launched.','Requires V30 to pass first. Code is prepared; independent cross-fold evaluation is still pending.')
]

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((ROOT / 'artifacts/dashboard-snapshot.json').read_text())
    host = pd.read_csv(ROOT / 'data/name-that-llm/train.csv', dtype={'id':str, 'prompt_id':str})
    test = pd.read_csv(ROOT / 'data/name-that-llm/test.csv', dtype={'id':str})
    pool = pd.read_parquet(ROOT / 'data/name-that-llm/external/pool_v4.parquet')
    records = host.to_dict(orient='records')
    for row in records:
        row['words'] = len(row['text'].split())
    keys = ['version','title','status','family','method','why','local','outcome']
    experiments = [dict(zip(keys, row)) for row in EXPERIMENTS]
    data = dict(snapshot, rows=records, visible_test=test.to_dict(orient='records'), experiments=experiments,
                model_counts=host.model.value_counts().sort_index().to_dict(),
                stats={'rows':len(host),'prompts':host.prompt_id.nunique(),'models':host.model.nunique(),
                       'median_words':int(host.text.str.split().str.len().median()),'external_rows':len(pool),
                       'external_sources':pool.source.nunique()},
                provenance={'host_sha256':hashlib.sha256((ROOT/'data/name-that-llm/train.csv').read_bytes()).hexdigest(),
                            'sources':['Kaggle competition metadata and submission API, 2026-10-26',
                                       'Local train.csv and test.csv supplied by Kaggle',
                                       'README.md and versioned experiment artifacts',
                                       'Public external pool: Nishant Dahal and original contributors, CC BY-SA 4.0']})
    (DEST / 'data.js').write_text('window.NTL_DATA = ' + json.dumps(data, ensure_ascii=False).replace('</',r'<\/') + ';\n')
    for name in ['index.html','app.js','style.css','hypotheses.js','research-history-2026-10-26.txt']:
        shutil.copyfile(ROOT/'dashboard'/name, DEST/name)
    shutil.copyfile(ROOT/'README.md',DEST/'technical-history.txt')
    manifest = {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(DEST.iterdir()) if p.is_file()}
    (ROOT/'artifacts/dashboard-build-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'rows':len(records),'experiments':len(experiments),'submissions':len(data['submissions']),
                      'output':str(DEST),'data_bytes':(DEST/'data.js').stat().st_size}))

if __name__ == '__main__':
    main()
