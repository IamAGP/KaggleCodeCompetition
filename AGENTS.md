# Project instructions

The user's durable storage preference is AWS S3.

- Save reusable competition data, reviews, code snapshots, notebooks, logs,
  evaluation results, and submission records under
  `s3://kaggle-featured-code-comps/name-that-llm/` in `ap-south-1`.
- Keep local working copies as needed; S3 is the persistent archive.
- Preserve the bucket's private access, encryption, and versioning.
- Verify uploaded sizes and SHA-256 checksums and retain an artifact manifest.
- Never include credentials, authentication files, or presigned URLs in archives.
- Keep raw competition data and generated predictions out of the public Git repo.
- Use the official AWS MCP tools for AWS operations.
- The first baseline's fixed comparison is complete. Do not silently tune or
  rerun the same experiment; record new experiments separately.
