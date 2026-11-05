# AWS V46 feasibility — read-only, November 6

Decision: the full configured V46 round does not fit its $1.25 new-spend cap at current observed AWS prices. A strictly bounded CUDA NF4 replay/throughput trial is feasible in principle after reconciling the outstanding Runpod bill and provisioning a dedicated access/watchdog arrangement. Nothing was launched or modified.

Official AWS MCP queried DescribeRegions, regional GetServiceQuota, Pricing GetProducts, DescribeSpotPriceHistory, DescribeInstanceTypes, DescribeInstances, SSM DescribeInstanceInformation, and IAM profile/policy reads. All returned successful API receipts. Evidence: `artifacts/nov6-aws-preflight/report.json`.

## Quotas and current prices

Only Mumbai (`ap-south-1`) among 18 enabled regions has nonzero G-family quota: 48 on-demand vCPUs and 8 Spot vCPUs. Its present instances consume no G quota. Offerings/prices do not guarantee launch capacity.

| Type | GPU | RAM / vCPU | Linux on-demand | Current Spot observations |
|---|---|---|---:|---:|
| g6.xlarge | L4, 22,888 MiB usable reported VRAM | 16 GiB / 4 | $0.9664/hour | $0.4545/hour in 1b; $0.5406 in 1a |
| g5.xlarge | A10G, 22,888 MiB reported VRAM | 16 GiB / 4 | $1.208/hour | $0.5481/hour in 1b; $0.6036 in 1a |

On-demand SKUs: `TUVAGNTZU3PYSBYD` (g6), `JRSA88BXM3KNDEGZ` (g5), publication 2026-11-05. Spot observations are timestamped November 5 15:00–17:00 UTC; they are variable rates, not a guaranteed quote. Mumbai gp3 is $0.0912/GB-month, SKU `D7RMW443YWTMZ4V5`; a 30 GB root disk is roughly $0.0038/hour before other costs. No reserved-instance purchase is proposed.

The V46 configuration allocates 12,600 training seconds (3.5 hours). Even $0.4545/hour Spot is **$1.59075 compute alone**, excluding setup, backup and storage. On-demand g6 is **$3.3824**. The configured 1,500-second per-fit number is not a measurement on L4/A10G. GPU memory looks plausible for 4B NF4; actual CUDA replay, RAM footprint and speed still require testing.

## Bounded trial proposal, not a launch instruction

Use one g6.xlarge in Mumbai, preferably on-demand for a short deterministic preflight; a 35-minute total lifetime costs about $0.564 compute. At most 20 minutes for CUDA replay and throughput, then at least 10 minutes for export/verification, with a hard shutdown by minute 35. Treat startup/download time as part of lifetime. No full arm launch unless measured projection plus backup reserve fits remaining cap. Stop immediately on CUDA replay mismatch, OOM, or runtime projection failure.

Before launch, verify a dedicated instance profile with SSM core access and read/write limited to the competition prefix, IMDSv2, encrypted root disk, no inbound SSH, and a prebuilt verified CUDA environment to avoid spending the trial installing packages. Five existing profiles have SSM core access, but all six profiles belong to other projects and none has identity-policy access to the competition S3 prefix. Two unrelated SSM nodes are online; do not reuse or modify them. Caller RunInstances/PassRole/SendCommand permissions, bucket policy/SCP effects and a suitable image/network path remain unverified: IAM feasibility is **partial**, not confirmed.

Install an OS shutdown timer before training; configure instance-initiated shutdown to **terminate**, not its default stop. Add an independent provider-side scheduled termination targeting only the new instance, with the schedule verified before training. Set all experiment EBS volumes DeleteOnTermination. No detached disks, Elastic IP, NAT gateway or other persistent billable resources should be added. Verify the instance reaches terminated after the trial, and check its volumes are deleted.

Back up code/config hashes, environment, numerical replay results, metrics, logs and any checkpoint to private `s3://kaggle-featured-code-comps/name-that-llm/` in the same region, with per-file size/SHA-256 manifest and read-back verification. Stream small logs/results during work; do not rely on a final upload after a Spot interruption. Preserve existing encryption/versioning; no credentials or signed URLs in archives. The hard cost watchdog remains active even if a final upload fails.

Budget: prior recorded spend is $3.5691351859, but the latest short Runpod rental is unresolved. Do not assume the remaining $10 budget is fully available. Reconcile it before any trial.

Official references: [EC2 lifecycle and deletion behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html), [instance-initiated shutdown](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html), [SSM prerequisites](https://docs.aws.amazon.com/systems-manager/latest/userguide/fleet-manager-default-host-management-configuration.html). Prices above come from live official AWS Pricing/EC2 APIs, not third-party pricing sites.
