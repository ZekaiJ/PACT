# Contributing

Focused bug reports, reproducibility checks, and small documentation improvements are welcome.

## Before opening an issue

1. Install the package with `python -m pip install -e .`.
2. Re-run the smallest command that demonstrates the problem.
3. Record the operating system, Python version, command, expected behavior, and observed output.
4. Use the reproducibility issue template when a reported result cannot be reproduced.

Do not post credentials, private records, provider-restricted media, model weights, or unlicensed datasets in an issue.

## Pull requests

Keep each pull request focused on one scientific or software concern. Before submitting, run:

```bash
python examples/pact_quickstart.py
python -m unittest discover -s tests -v
python analysis/verify_release.py
```

Changes to metrics, denominators, study protocols, reference values, or released result files must include regenerated outputs and a concise explanation of the scientific effect. Documentation-only changes should not alter numerical claims.

## Scope

PACT is a research implementation for provenance-aware evidence accounting and typed action admission. Proposals that broaden the public API or add new experimental dependencies should first be discussed in an issue.
