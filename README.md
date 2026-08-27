<div align="center">

<img src="assets/pact-logo.png" width="220" alt="PACT logo" />

# PACT

**P**rovenance-**A**ware evidence **C**onservation and **T**yped action admission

Reference implementation and released experiments for<br />
**Not All Agreement Counts as Corroboration: Provenance-Conserving Multi-View Fusion for Action Admission in Human–Robot Collaboration**

Zekai Jin · Hanrong Zhang · Yihong Tang · Fei Hu · Zhen Dong · Yi Shao

[Why PACT](#why-pact) · [Quick start](#quick-start) · [Method](#method) · [Results](#results) · [Reproduce](#reproduce) · [Citation](#citation)

[![Tests](https://github.com/ZekaiJ/provenance-aware-action-admission/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ZekaiJ/provenance-aware-action-admission/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-1f5a94.svg)](https://www.python.org/)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-6f42c1.svg)](CITATION.cff)

</div>

> **PACT asks a question that ordinary fusion leaves implicit: which outputs are allowed to count separately?**

Prompt variation, repeated sampling, self-consistency, and related checkpoints can multiply agreement without adding a new observation. PACT uses provenance to distinguish repeated computation from separately countable support before evidence is combined or an action is admitted.

<p align="center">
  <img src="assets/pact_overview.png" width="100%" alt="PACT motivation: repeated computation can multiply agreeing outputs without adding evidence, whereas support from separately countable provenance components can corroborate a candidate action." />
</p>

## Why PACT

Reliability and conflict describe the quality of individual source values. PACT addresses a different question: how many evidential contributions those values represent.

| Situation | PACT interpretation | Decision consequence |
|---|---|---|
| Several outputs descend from one acquisition | One provenance component | Agreement is retained, but it does not become additional evidence |
| Outputs belong to separately countable components | One budget per component | Common support is accumulated across components |
| A candidate scores highly | Candidate selection | Permission still depends on ordered, reason-specific admission checks |

This separation gives PACT three useful properties: copies retained within a component do not amplify support; false splitting can inflate the evidence budget; and over-broad merging can suppress complementary evidence.

## Quick start

```bash
git clone https://github.com/ZekaiJ/provenance-aware-action-admission.git
cd provenance-aware-action-admission
python -m pip install -e .
python examples/pact_quickstart.py
```

Python 3.10 or later is required. The example constructs three source opinions: geometry and risk share a scene parent, while language forms a separate component. It prints the posterior, provenance components, and selection score:

```text
posterior: [0.5917, 0.1286, 0.1059, 0.0907, 0.0831]
provenance components: (('geometry', 'risk'), ('language',))
selection score: 0.7364
```

<details>
<summary><strong>Minimal Python example</strong></summary>

```python
import numpy as np

from action_admission import SourceEvidence, pact_fuse


def source(name, probabilities, parents):
    return SourceEvidence(
        source_id=name,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        quality=0.9,
        conflict=0.0,
        missing=False,
        parents=parents,
    )


sources = [
    source("language", [0.78, 0.08, 0.06, 0.04, 0.04], ("command",)),
    source("geometry", [0.70, 0.12, 0.08, 0.06, 0.04], ("scene",)),
    source("risk", [0.64, 0.14, 0.10, 0.08, 0.04], ("scene",)),
]

result = pact_fuse(sources, concentration=8.0)
print(result.posterior, result.selection_score, result.group_ids)
```

</details>

## Method

<p align="center">
  <img src="assets/pact_method_overview.png" width="100%" alt="PACT method overview from simulation-grounded source representation through provenance topology, coordinatewise conservation, cross-component accumulation, posterior projection, score selection, and typed admission." />
</p>

For source evidence $e_i \in \mathbb{R}_{\geq 0}^{K}$ with parent sets $P_i$, PACT connects sources whose parent sets overlap,

$$
(i,j) \in E_{\mathrm{prov}} \quad \Longleftrightarrow \quad P_i \cap P_j \neq \varnothing.
$$

The connected components form the provenance partition $\Pi_P$. Within each component, PACT retains the coordinatewise common support; across separately countable components, it adds the resulting budgets:

$$
b_C = \bigwedge_{i\in C} e_i,
\qquad
E_{\Pi} = \sum_{C\in\Pi_P} b_C.
$$

Posterior projection and non-vacuity rank candidate contracts. Typed admission then evaluates command consistency, source validity, risk support, and component corroboration in order, preserving the reason for hold, confirmation, or fallback. An admitted action is eligible for downstream execution; admission is not execution itself.

See [`docs/PACT.md`](docs/PACT.md) for the complete operator and [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) for its implementation map.

## Results

The released studies are organized around three questions.

| Question | Evidence in this repository | Scope |
|---|---|---|
| **Do copies create evidence?** | Exact copies retained within their component leave PACT unchanged; false splitting increases the counted budget. [`Topology and multiplicity`](results/topology_multiplicity/README.md) · [`Operator characterization`](results/operator_characterization/README.md) | Algebraic properties and controlled provenance interventions |
| **How do partition errors change support?** | Across all 856 single-merge cover relations among 203 six-view partitions, coarsening never increases the PACT budget, although selective ordering may improve or worsen. [`Partition coarsening surface`](results/partition_coarsening_surface/README.md) | Budget monotonicity does not imply utility monotonicity |
| **Are accounting, ranking, and permission interchangeable?** | Under the stated admission rule and method-specific scores, complete PACT attains ncsAURC 0.0861 over $[0.10,0.39]$, compared with 0.1479 for nested Dirichlet composition, across 31,200 evaluations from 48 scene clusters. [`Reference comparison`](results/reference/) · [`Fusion and admission attribution`](results/minimal_attribution_2x2/README.md) | Benchmark-, score-, and rule-specific comparison |

Further studies transfer the accounting mechanism to public feature-view predictions, multi-camera VLM outputs, and checkpoint families. In the downstream 720-case HRC evaluation, target-identity and event-proximity evidence retains 51 of 53 episode-reference-consistent Qwen3-VL-8B candidates while withholding 90 of 91 episode-reference-inconsistent admissions. See the [`result index`](results/INDEX.md) for the complete set of analyses.

## Reproduce

Choose the shortest path for the question you want to inspect:

| Goal | Command |
|---|---|
| Run one PACT decision | `python examples/pact_quickstart.py` |
| Compare same-parent and false-split copies | `python examples/copy_assignment_demo.py` |
| Check the core operator and admission properties | `python -m unittest discover -s tests -v` |
| Run the main controlled study | `python experiments/pcecf_study.py --output outputs/pcecf_study` |
| Verify the complete release | `python analysis/verify_release.py` |

The verification command checks the package tests, file manifest, released analyses, and numerical-result bindings. For focused instructions, use the [`documentation guide`](docs/index.md), [`reproducibility notes`](docs/REPRODUCIBILITY.md), and [`result index`](results/INDEX.md).

<details>
<summary><strong>Paper-to-repository guide</strong></summary>

| Paper component | Entry point or result |
|---|---|
| Fusion properties and boundary cases | `python -m unittest discover -s tests -v` |
| Fusion and admission comparison | `python analysis/minimal_attribution_2x2.py --output outputs/minimal_attribution_2x2` |
| Topology and multiplicity interventions | [`results/topology_multiplicity/`](results/topology_multiplicity/README.md) |
| Equal-cardinality regrouping | [`results/equal_cardinality_topology/`](results/equal_cardinality_topology/README.md) |
| Public multi-view transfer | [`results/public_outcome_closure/`](results/public_outcome_closure/README.md) |
| Exhaustive partition coarsening | [`results/partition_coarsening_surface/`](results/partition_coarsening_surface/README.md) |
| Joint scene–configuration holdout | [`results/joint_scene_configuration_holdout/`](results/joint_scene_configuration_holdout/README.md) |
| Multi-camera VLM transfer | [`results/native_view_fm_provenance_transfer/`](results/native_view_fm_provenance_transfer/README.md) |
| Checkpoint-family analysis | [`results/balanced_fm_panel/`](results/balanced_fm_panel/README.md) |
| HRC admission studies | [`results/habit_fixed_image_admission/`](results/habit_fixed_image_admission/README.md) |

</details>

<details>
<summary><strong>Repository layout</strong></summary>

```text
.
|-- src/action_admission/          # PACT and typed admission
|-- examples/                      # small executable examples
|-- experiments/                   # controlled studies and interventions
|-- analysis/                      # statistical analyses and release checks
|-- configs/                       # study specifications
|-- data/controlled/               # released inputs and separate labels
|-- results/                       # tables, summaries, and no-media outputs
|-- docs/                          # method, data, scope, and reproduction notes
`-- tests/                         # deterministic properties and boundary cases
```

</details>

## Data and scope

The repository does not redistribute provider-restricted media, model weights, or licensed datasets. Acquisition requirements and redistribution boundaries are documented in [`docs/DATA.md`](docs/DATA.md), [`docs/THIRD_PARTY_ASSETS.md`](docs/THIRD_PARTY_ASSETS.md), and [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).

PACT is a research implementation of decision-level evidence accounting and typed action admission. It is not a certified robot safety controller and should not serve as the sole execution authority around people.

## License

PACT software, scripts, and tests are released under the [MIT License](LICENSE). Original documentation, controlled data, result tables, and figures owned by the PACT contributors are released under [CC BY 4.0](LICENSE-DATA). These grants do not relicense third-party datasets, model weights, source media, or implementations. HABIT attribution and modification notices are provided in [NOTICE.md](NOTICE.md), with the complete redistribution boundary in [docs/THIRD_PARTY_ASSETS.md](docs/THIRD_PARTY_ASSETS.md).

## Contributing

Focused bug reports and reproducibility improvements are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md) before opening an issue or pull request.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A publication DOI will be added after assignment.
