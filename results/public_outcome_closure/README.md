# Donor-complete public outcome readout

The snapshot evaluates frozen per-view evidence from TMC and RCML on HandWritten/Mfeat. Every native view serves once as the duplicated donor; no emitter is retrained.

| Pair | False split minus registered copy, ncsAURC [0.10, 0.90] | Posterior L1 drift |
|---|---:|---:|
| TMC | +0.004914 [0.000418, 0.010787] | +0.246273 [0.243159, 0.249355] |
| RCML | +0.002442 [0.000848, 0.004408] | +0.214278 [0.208977, 0.220037] |

Intervals use 2,000 class-stratified record-paired bootstrap draws. Each sampled record carries all five frozen training realizations, all six donors, and every intervention arm. Same-parent registered copies pass exact invariance before outcome inference.

`PAIRED_CONTRASTS.csv` is the primary statistical record. `PAIR_OUTCOMES.csv` contains point estimates; `INVARIANCE_GATE.json`, `PAIR_SUPPORT.json`, and `FINAL_AUDIT.json` record the integrity and support checks. PIE/RCML is explicitly blocked from this outcome analysis because donor-level per-view evidence was not retained.

NLL, Brier score, and ECE move in the opposite donor-macro direction from ncsAURC, while worst-donor diagnostics reverse NLL and Brier. The result separates average probability quality from selective ordering; it does not license a universal calibration claim.
