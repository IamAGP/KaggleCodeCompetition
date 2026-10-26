# Private competition notebook

The static dashboard introduces the competition, lets the owner read all 2,699
training rows, compares models on the same prompt, and separates ten actual
public submissions from 33 experiment records. It is a dated snapshot, not a
live monitor or a training control panel.

Build from the project root:

```sh
.venv/v5/bin/python scripts/build_competition_dashboard.py
node --check dashboard/app.js
NODE_PATH=/path/to/jsdom/node_modules node dashboard/check.cjs artifacts/dashboard-site/dist
```

The builder reads the sanitized official API snapshot at
`artifacts/dashboard-snapshot.json`, the local competition files, the public
external-pool metadata, and README.md. Experiment summaries are editorial and
must be updated against recorded evidence when new results arrive.

Generated data, deployment archives and Site repository state stay under the
ignored artifacts directory. Do not add generated data.js to the public Git
repository. The Site is owner-private. Use its existing project identity in
`artifacts/dashboard-site/.openai/hosting.json` for future publication.

DOM interaction checks passed on the first build. No connected browser was
available for rendered visual QA. CSS includes desktop and narrow-screen layouts,
but that is not a claim of screenshot verification.

Sources and attribution are visible in the dashboard. Dataset passages are
unaltered, attributed to Kaggle, and retain the CC BY-SA 4.0 license. The private
S3 archive records the complete first snapshot, checksums and version IDs.
