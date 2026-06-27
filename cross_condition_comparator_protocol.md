# Cross-Condition Comparator Analysis: Pre-Registered Protocol

Timestamped via git commit.

## Hypothesis

spTRIO's three deviation metrics (basin shift, extremum displacement,
Jaccard distance), computed between rest and task channel selections,
will discriminate rest-from-task more sensitively than the same three
metrics computed between rest and task ICA-selected channels.

This tests spTRIO's actual claim: that the multi-view + locked-
parameter design produces a cross-condition deviation signal that
ICA-based selection does not.

## Test

Subject: S001R03 (motor execution). This subject has rest periods
(T0) and task periods (T1+T2) within one recording, allowing a
within-recording cross-condition comparison without confounds from
across-session variability.

Band: β (13–30 Hz). The band where motor execution produces the
strongest expected change relative to rest, per Pfurtscheller &
Lopes da Silva (1999). Also the band where the single-band
comparator showed ICA stronger than spTRIO, so this is a genuine
discriminating test rather than a softball.

## Procedure (locked)

1. Segment S001R03 into rest periods (T0) and task periods (T1+T2).
2. For each segment-condition (rest, task):
   a. Run spTRIO calibration → channel set `S_sp_rest`, `S_sp_task`.
   b. Run sklearn FastICA (max_iter=5000, tol=1e-5, random_state=42,
      n_components=full). Select IC with highest |corr| of mixing-
      column topography to all-channel β-power topomap. Take K
      highest-magnitude channels of that IC, where K matches the
      spTRIO K for the same condition.
      → channel set `S_ica_rest`, `S_ica_task`.
3. Compute the three deviation metrics for each method's rest→task
   transition:
   a. spTRIO: basin shift, extremum displacement, Jaccard distance
      between `S_sp_rest` and `S_sp_task`.
   b. ICA: same three metrics between `S_ica_rest` and `S_ica_task`,
      using identical metric implementations.
4. Report all six values plus the comparison.

## What gets reported (committed in advance)

A single table with six numbers and a one-line interpretation:

  Metric                     spTRIO   ICA     Δ(sp − ICA)
  Basin shift                  ...    ...     ...
  Extremum displacement        ...    ...     ...
  Jaccard distance             ...    ...     ...

## Decision rule (locked before running)

- If spTRIO's metric magnitudes exceed ICA's on **at least 2 of 3
  metrics** by a clear margin: "spTRIO's multi-view selection
  produces a stronger cross-condition deviation signal than ICA-
  based selection on the same metric formulae, on this subject."

- If results are mixed (1 of 3 wins, or magnitudes similar): "On
  this single subject, spTRIO and ICA produce comparable cross-
  condition deviation signatures. Population-level testing is
  required."

- If ICA's metric magnitudes exceed spTRIO's on most metrics:
  "ICA-based selection is competitive with spTRIO for cross-
  condition deviation as well. The differentiator of spTRIO lies
  elsewhere — in pre-registered reproducibility and lack of
  manual judgement — rather than in deviation-signal magnitude."

The third outcome would substantially weaken the paper. Acknowledge
this possibility now, in writing, so we report it honestly if it
happens.

## Convergence policy

If FastICA fails to converge within 5000 iterations on either
rest or task, raise to 20000 and re-run. If it still does not
converge, report the result with explicit notation that ICA
did not fully converge.

## What we will NOT do

- Add or change subjects mid-analysis.
- Change the band tested.
- Re-define the deviation metrics.
- Adjust the IC-selection rule after seeing results.
- Add additional comparators (sLORETA, beamforming) post hoc.