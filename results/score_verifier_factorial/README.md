# Score and verifier-policy factorial

This snapshot contains the controlled-study score crossover and verifier
policy sensitivity. Policy A requires three-source unanimity and three
registered components. Policy B broadens the trigger to at least two
supporting sources while retaining three components. Policy C requires
language plus a physical role and two components.

Regenerate after producing the controlled PACT output:

```bash
python experiments/pcecf_study.py
python analysis/score_verifier_factorial.py \
  --pcecf-output outputs/pcecf_study \
  --output outputs/score_verifier_factorial
```

Verify the released snapshot:

```bash
python analysis/verify_secondary_results.py
```

