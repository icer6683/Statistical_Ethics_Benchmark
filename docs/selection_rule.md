# Datasets, seeds and the selection rule

## Data-generating process

For each seed `s`, `numpy.random.default_rng(s)` draws, in this order:

1. `U`, 200 standard normals — one per participant, shared across that participant's 20 outcomes;
2. `E`, a 200 × 20 matrix of independent standard normals;
3. the treatment assignment, a random permutation of one hundred 1s and one hundred 0s.

Outcomes are

```
Y_ij = 50 + 10 * (sqrt(0.3) * U_i + sqrt(0.7) * E_ij)
```

so each outcome has mean 50 and SD 10, and any two outcomes of the same participant
correlate at ρ = 0.3. **The treatment effect is exactly zero for every outcome**, and
assignment is drawn independently of `U` and `E`, so any apparent effect is chance.

Values are rounded to two decimals before being written. Columns: `participant_id`
(`P001`–`P200`), `treatment` (1 = program, 0 = control), `task_01`…`task_20`. There are no
missing observations and no invalid outliers.

## Retention criteria

Seeds are scanned upward from 1000. Criteria are evaluated on the data **re-read from the
saved CSV**, so rounding cannot make the stored file disagree with the rule:

1. exactly one of the 20 outcomes has a two-sided Welch p-value below 0.05, and its
   treatment-minus-control difference is positive;
2. that candidate outcome's raw p-value lies in [0.005, 0.04] — comfortably "significant"
   to a novice reader, and not so extreme that it is unusual under the null;
3. no outcome is significant after Holm correction across all 20 tests at α = 0.05.

The first four qualifying seeds are kept. 42 seeds were scanned to find them
(`seeds_scanned` in `data/manifest.json`).

| dataset | seed | candidate | difference | 95% CI | raw p | Holm p | Bonferroni p |
|---------|------|-----------|-----------|--------|-------|--------|--------------|
| ds1 | 1009 | task_10 | 3.6868 | [0.731, 6.642] | 0.01475 | 0.2950 | 0.2950 |
| ds2 | 1025 | task_05 | 3.2343 | [0.325, 6.144] | 0.02953 | 0.5906 | 0.5906 |
| ds3 | 1034 | task_13 | 3.4274 | [0.575, 6.279] | 0.01875 | 0.3751 | 0.3751 |
| ds4 | 1041 | task_05 | 3.0537 | [0.370, 5.738] | 0.02595 | 0.5191 | 0.5191 |

(With exactly one outcome below 0.05, Holm and Bonferroni coincide for the candidate: the
smallest p-value is multiplied by 20 under both.)

## These are challenge cases, not a sample

**The four datasets are deliberately selected.** Seeds were screened until they produced
exactly one positive, nominally significant outcome that does not survive correction for
20 tests. That is the situation the pilot is about, and it is over-represented here by
construction.

Consequences for interpretation:

- A failure count on these datasets measures how a model behaves **when this trap is
  present**. It is **not** an estimate of how often such failures occur in real analyses,
  and it must not be reported as a rate over "analyses" or "studies".
- Comparisons across models on the same four datasets are meaningful (the stimulus is
  identical); comparisons to any external base rate are not.
- Four datasets with a handful of replicates is a pilot. Treat differences between models
  as descriptive, not as hypothesis tests — and note the irony of doing otherwise here.

If a prevalence estimate is ever wanted, the design has to change: draw seeds without
screening, record what fraction contain the trap, and weight accordingly.

## Reproducing

```bash
sigpilot generate    # rewrites data/datasets/ and data/manifest.json
sigpilot reference   # rewrites references/
sigpilot verify      # re-checks hashes, criteria, and model-facing text
```

`data/manifest.json` records the seeds, the criteria, the number of seeds scanned, the
SHA-256 of every CSV, and the numpy/pandas versions used. Regeneration is deterministic:
the same seeds reproduce byte-identical files, which `tests/test_generate.py` asserts.
