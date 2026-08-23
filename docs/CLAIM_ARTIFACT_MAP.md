# Claim-to-artifact map

`CLAIM_ARTIFACT_MAP.csv` binds the manuscript's main quantitative claims to a
specific released artifact, an exact row or JSON locator, the artifact hash,
the generating or archival analysis script, the denominator, statistical unit,
and the claim boundary.

The map is deliberately narrower than a project history. It covers the claims
that carry the abstract, main results, robustness section, external-transfer
section, and computational-cost statement. `expected_artifact_tokens` provides
a lightweight check that the claim-bearing values remain present in the bound
file; `analysis/verify_release.py` verifies those tokens, paths, and hashes.

Regenerate the map after an intentional artifact change:

```bash
python analysis/build_partition_disclosure_summary.py
python analysis/policy_cost_sensitivity.py
python analysis/build_claim_artifact_map.py
```

Regenerating hashes does not validate a scientific claim by itself. Scientific
validity still depends on the frozen protocols, denominators, statistical
units, and interpretation boundaries named in each row.
