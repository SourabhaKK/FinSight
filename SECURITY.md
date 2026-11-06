# Security policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue.

Use GitHub's private reporting: open the repository's **Security** tab and choose **Report a vulnerability**. If that option is not shown, open a public issue that says only that you have a security concern, without details, and I will arrange a private channel.

This is a personal project, so there is no guaranteed response time. I aim to acknowledge reports within a week.

## Supported versions

Only the latest commit on `main` is supported.

## Scope notes

- Secrets are never committed to this repository. If you find one, please report it privately.
- Model files (`artefacts/*.pt`, `*.joblib`) are loaded with pickle-based loaders. Only load model files you trust.
- Known dependency advisories in the RAG evaluation tooling (the `ragas` and langchain dev packages) are tracked in `docs/suggested-tickets.txt`.
