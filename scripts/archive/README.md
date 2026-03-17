# Archived Helpers

This directory keeps one-off or operational helpers that were useful during data recovery, migration, backup, sync, or incremental corrections, but are not part of the normal reproducible analysis path.

The top-level `scripts/` directory is reserved for the reusable workflow that can be run from scratch on the current sample. Archived helpers stay here as reference material and for provenance, not as part of the standard pipeline.

Typical archived categories:

- recovery-only HST download helpers
- shortlist-specific search/download scripts
- post-hoc correction scripts whose logic was folded into the main pipeline
- backup, rsync, and Overleaf sync helpers
- QA/audit utilities used to validate intermediate cleanup work

If one of these archived scripts becomes necessary again, prefer checking first whether the required logic should be integrated into the main pipeline rather than running another one-off correction.
