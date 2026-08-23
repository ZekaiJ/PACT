# Scene15--RCML native-protocol gate

## Verdict

`HOLD_UNRESOLVED_RELEASE_PROTOCOL`

Scene15--RCML is not eligible for the public topology experiment. The released
payload and the two available split protocols reproduce accuracies of 0.6972
and 0.6944, respectively, rather than the published 0.7619. No Scene15
topology result is integrated into the manuscript.

## Evidence

- RCML repository commit: `c9c5ab41e6fe62a85e5f6441a4dc7b568e1fa421`.
- Released `main.py` invokes `HandWritten()` in its normal-data entry point and
  `PIE()` in its conflict-data entry point. Although `data.py` defines
  `Scene()`, the release exposes no Scene15 training invocation or split file.
- The paper reports Scene15 accuracy `76.19 +/- 0.12`.
- Five-seed official-style 80/20 probe: mean `0.6972129319955407`, standard
  deviation `0.0130886600860104`.
- Five-seed 60/20/20 probe: mean `0.6944196428571429`, standard deviation
  `0.01555148522174141`.
- The supplementary material documents the shared hyperparameter setting but
  does not identify a Scene15 split seed/file or a separate released training
  command that closes the discrepancy.

## Frozen identities

| Object | SHA-256 |
|---|---|
| `main.py` | `f85751fe7d589bc4341f0f40984e6a908f7b7dacfd3a4f908ad85d3598a01020` |
| `data.py` | `9be7f75e53e3a0534e1cf3556eca666cac8833817768f6ba3c46a86601c0ab3a` |
| `model.py` | `42bf3fb0d877c7c55f6391035f530ae5a5accf1b8a84aed1b8f2577fe1ebc4ab` |
| `loss_function.py` | `2044693be41b481716ee831b8ba01b43e9da4856c5b47455ca532a63eeeb3bed` |
| `scene15_mtv.mat` | `52a71c6c675d5b0c1e07189778716bc52b972b17b58d8156c01422bcce2e4442` |
| RCML paper PDF | `061119a7fd82f7e3d6f81f854ee26445225be44c626422638a7a6abdc722aa2e` |
| RCML supplement PDF | `3aa34587a3c9b7c230980a6cd1b900cb48121bec817469096a43dc94a7bf72ee` |

## Claim boundary

This gate records a reproduction mismatch, not evidence that the published
number is incorrect. Resumption requires an author-supplied Scene15 split,
configuration, or training entry point that reproduces the native result
without outcome-guided tuning. Until then, the manuscript's explicit exclusion
of Scene15 is the final P1 result.
