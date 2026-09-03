# Documentation guide

This guide provides the shortest route from the paper's central idea to the
implementation, experiments, and results.

## Start here

| Question | Read or run |
|---|---|
| Where can I read the paper? | [arXiv:2609.01662](https://arxiv.org/abs/2609.01662) ([PDF](https://arxiv.org/pdf/2609.01662)) |
| What problem does PACT solve? | [Why PACT](../README.md#why-pact) |
| How does the conservation operator work? | [PACT operator](PACT.md) |
| How do I run one example? | `python examples/pact_quickstart.py` |
| Why does copy assignment matter? | `python examples/copy_assignment_demo.py` |
| Where is each reported result reproduced? | [Paper-to-code map](PAPER_TO_CODE.md) and [numerical-result index](CLAIM_ARTIFACT_MAP.md) |
| How do I check the complete repository? | [Reproducibility notes](REPRODUCIBILITY.md) |
| Where are the released results? | [Result index](../results/INDEX.md) |

## Method and implementation

- [PACT operator](PACT.md): provenance components, within-component meet, and
  cross-component accumulation.
- [`src/action_admission/pact.py`](../src/action_admission/pact.py): public PACT
  interface.
- [`src/action_admission/admission.py`](../src/action_admission/admission.py):
  public typed-admission interface.
- [Paper-to-code map](PAPER_TO_CODE.md): manuscript concepts and their executable
  counterparts.

## Data, results, and scope

- [Data notes](DATA.md): released inputs and data provenance.
- [Third-party assets](THIRD_PARTY_ASSETS.md): external sources and redistribution
  constraints.
- [Release scope](RELEASE_SCOPE.md): what the repository supports and what remains
  outside its scope.
- [Numerical-result index](CLAIM_ARTIFACT_MAP.md): principal numerical results,
  statistical units, and their repository locations.
