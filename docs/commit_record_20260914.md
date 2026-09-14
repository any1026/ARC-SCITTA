# Git Commit Record

Date: 2026-09-14

## Scope

This commit freezes the Stage9 anatomy-aware continuous fusion configuration and adds the paper-scale Fundus-doFE validation queue. It includes the derived evaluation scripts, a strict post-run aggregator, a non-overwriting screen queue, and the audit/metric disclosures.

## Verification before commit

- `py -3.9 -m py_compile` passed for all newly queued Python entry points.
- `git diff --check` passed.
- Runtime unit tests could not run on this workstation because the global Python 3.9 installation lacks the server dependencies (`torch`, `scipy`, and the project packages); the same tests are required in the server Conda environment before the queue starts.
- No checkpoints, datasets, raw images, server outputs, or logs are included in the commit.

## Commit

Primary freeze commit:

```text
66a5fc2 Freeze Stage9 anatomy fusion and add paper-table queue
```

The commit was created only in the nested `SCITTA-pro` Git repository and pushed to its existing `origin/main`. The parent project directory is not part of this Git history. Follow-up audit hardening, if any, is committed separately rather than rewriting the published commit.
