# Frozen shared-score transport

A logistic score is fitted only on the multiplicity-one outer-training folds, then transported without coefficient refitting to multiplicities 2, 4, 8, 16, and 32. Thresholds are recomputed from outer-training score quantiles without labels.

- [`full/`](full/) uses posterior features and observed-opinion count.
- [`no_count/`](no_count/) removes only observed-opinion count.

The full score changes the PACT-minus-nested ordering at multiplicity 2. Without the count feature, the change occurs at multiplicity 4. The PACT fusion operator remains numerically invariant in the registered-copy arm of both analyses, so the result cannot be reduced to an observed-source counter alone.
