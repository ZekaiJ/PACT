# Decision-cost sensitivity

This analysis evaluates prespecified admission boundaries on the same held-out
PACT candidates and fold-specific thresholds used in the controlled study. For
wrong-admission to non-release cost ratio `lambda`, the reported relative cost
is

`(lambda * wrong admissions + non-releases) / all decisions`.

The always-withhold policy therefore has relative cost 1. Correct admissions
have zero cost. Hold, confirm, and fallback receive the same non-release cost,
so this analysis compares admission boundaries rather than response-specific
utilities.

Regenerate the outputs from the repository root:

```bash
python analysis/policy_cost_sensitivity.py
```

The analysis uses 31,200 held-out decisions from 48 scenes, a target coverage
of 0.13, and 2,000 scene-cluster bootstrap replicates with seed 2052. The
prespecified policies and both limiting strategies remain in the output; the
analysis does not select or promote a policy from these results.
