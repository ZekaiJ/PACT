# PACT operator

Let each typed source slot provide nonnegative evidence
\(e_i \in \mathbb{R}_{\ge 0}^{K}\) and a provenance parent set \(P_i\).
PACT forms the connected components of the graph

\[
(i,j) \in E \quad \Longleftrightarrow \quad P_i \cap P_j \ne \varnothing.
\]

For a component \(C\), its evidence is the coordinatewise lower envelope

\[
e_C(k)=\min_{i\in C}e_i(k).
\]

An unavailable or structurally invalid slot contributes the zero vector while its
slot and provenance parent set remain observable. Under this fixed-catalog
component-completeness convention, the zero participates in the component
envelope; an available categorical vector must instead have positive unit mass and is rejected if it is malformed.

Evidence is then accumulated across components,

\[
e_{\mathrm{PACT}}(k)=\sum_C e_C(k), \qquad
p(k)=\frac{e_{\mathrm{PACT}}(k)+1}
{\sum_j e_{\mathrm{PACT}}(j)+K}.
\]

This construction is invariant to an exact same-parent copy within an existing
component. For a fixed component partition, coordinatewise perturbations of the
source evidence bound the perturbation of the component envelope, which gives
the near-copy stability tested in `tests/test_pact.py`.

A parent identifier denotes the source event or intermediate evidence object whose
redistribution constitutes evidence reuse. A common physical origin alone does not
merge complementary transforms. This granularity is fixed before evaluation.

The provenance policy also determines which component budgets may accumulate:
components kept separate by that policy are separately countable in the common
evidence unit. This accounting permission does not certify statistical independence.

The operator assumes that the typed catalog and provenance parent sets supplied to
it are observable. It does not authenticate those relations, infer hidden
common ancestry, or guarantee calibration or physical safety. The typed-admission
stage in `src/action_admission/admission.py` addresses release
eligibility after candidate selection; it is not part of the conservation
operator.
