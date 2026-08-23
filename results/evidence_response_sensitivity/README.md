# Evidence-response sensitivity

This snapshot separates a 256-setting opinion-parameter nuisance sweep from 64
matched occlusion-response sweeps evaluated at three occlusion levels. In the
paired response study, direct unoccluded outputs remain unchanged and
partial-occlusion coverage decreases monotonically in all 64 sweeps. The
maximum all-case wrong-admission rate is 0.0146.

The readout tests whether weaker controlled evidence changes selective
admission in the expected direction. It is not a raw-sensor, physical-robot, or
participant-facing validation. `report.json` is the primary aggregate record;
the CSV files expose setting-level and response-level values. `run.py` is an
archival generation script with dependencies outside this compact release.
